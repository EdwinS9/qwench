"""H2H — Full-FT SFT vs SDFT forgetting comparison, single GPU, per-epoch MMLU.

Both on Qwen3-4B, lr=5e-5, full-parameter, same data, same intensity. The SFT sweep
already showed a 22-pt MMLU drop here. Question: does SDFT forget less?

    modal run training/forget_h2h.py   (30 epochs, ~2h)
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers>=4.51", "trl>=1.0", "peft", "accelerate",
                 "datasets", "wandb", "torch")
    .add_local_dir(str(REPO / "qwench"), remote_path="/root/qwench")
    .add_local_dir(str(REPO / "schemas"), remote_path="/root/schemas")
    .add_local_dir(str(REPO / "training"), remote_path="/root/training")
    .add_local_dir(str(REPO / "data"), remote_path="/root/data")
)

app = modal.App("qwench-h2h", image=image)
hf_cache = modal.Volume.from_name("qwench-hf-cache", create_if_missing=True)


@app.function(gpu="A100-80GB", timeout=8 * 60 * 60,
              volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_name("wandb-secret")])
def h2h(model_name: str, epochs: int, lr: float, mmlu_n: int):
    import copy
    import json
    import random
    import sys
    sys.path.insert(0, "/root")

    import torch
    import wandb
    from datasets import Dataset, load_dataset
    from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments
    from trl import SFTConfig, SFTTrainer

    from qwench.prompts import pick_demo, student_messages, teacher_messages
    from qwench.sdft_loss import analytic_token_kl
    from training.common import (
        TrainConfig,
        evaluate,
        load_examples,
        load_tokenizer,
        response_logits,
    )
    from training.general_eval import mmlu_accuracy

    # --- shared setup -----------------------------------------------------------

    def render(msg):
        return tok.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    cfg = TrainConfig(method="sdft", model=model_name, epochs=epochs, lr=lr,
                      use_lora=False, batch_size=2, ema_alpha=0.02)
    tok = load_tokenizer(model_name)
    train_rows = load_examples("train")
    heldout = load_examples("heldout")
    rng = random.Random(cfg.seed)
    spe = max(1, len(train_rows) // cfg.batch_size)

    mmlu_ds = load_dataset("cais/mmlu", "all", split="test")
    mmlu_ds = mmlu_ds.shuffle(seed=0).select(range(mmlu_n))
    mmlu_items = [{"question": r["question"], "choices": r["choices"],
                    "answer": r["answer"]} for r in mmlu_ds]

    def snap(m, step, tag):
        acc = mmlu_accuracy(m, tok, mmlu_items)
        plan = evaluate(m, tok, heldout, train_rows, cfg, rng)[0]["eval/plan_success"]
        print(f"[{tag}] step={step} mmlu={acc:.3f} plan_success={plan:.3f}")
        return acc, plan

    def fresh_model():
        m = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to("cuda")
        m.config.use_cache = False
        return m

    @torch.no_grad()
    def _gen_batch(model_obj, prompts, max_new, do_samp, temp=1.0, top_p=0.95):
        device = next(model_obj.parameters()).device
        eos = tok.eos_token_id
        gk: dict = {"max_new_tokens": max_new, "use_cache": True,
                    "pad_token_id": tok.pad_token_id}
        if do_samp:
            gk.update(do_sample=True, temperature=temp, top_p=top_p)
        else:
            gk["do_sample"] = False
        prev = tok.padding_side
        tok.padding_side = "left"
        outs = []
        try:
            for k in range(0, len(prompts), 16):
                enc = tok(prompts[k:k + 16], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(device)
                gen_rows = model_obj.generate(**enc, **gk)
                for row in gen_rows[:, enc.input_ids.shape[1]:].tolist():
                    cont = []
                    for t in row:
                        cont.append(t)
                        if t == eos:
                            break
                    outs.append(cont)
            return outs
        finally:
            tok.padding_side = prev

    # --- base ------------------------------------------------------------------

    bm = fresh_model().eval()
    base_mmlu, _ = snap(bm, 0, "base")
    del bm
    torch.cuda.empty_cache()

    results = {}
    train_pool = load_examples("train")

    # ============================================================================
    # SFT
    # ============================================================================
    print("\n=== SFT ===\n")
    wandb.init(project="qwench-h2h", name="sft", config=cfg.__dict__,
               tags=["sft", "fullft"], reinit=True)
    m = fresh_model()
    ds = Dataset.from_list([
        {"prompt": render(student_messages(r)),
         "completion": json.dumps(r["target"]) + tok.eos_token}
        for r in train_rows
    ])

    class SftSnap(TrainerCallback):
        def on_step_end(self, args, state, control, model=None, **kw):
            if state.global_step % spe == 0:
                snap(model, state.global_step, "sft")

    SFTTrainer(model=m, args=SFTConfig(
        output_dir="/tmp/sft", num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size, gradient_accumulation_steps=2,
        learning_rate=cfg.lr, lr_scheduler_type="constant", warmup_ratio=0.0,
        logging_steps=10, bf16=True, max_length=cfg.max_len, gradient_checkpointing=True,
        report_to="wandb", run_name="sft-h2h", save_strategy="no",
        remove_unused_columns=False,
    ), train_dataset=ds, processing_class=tok, callbacks=[SftSnap()]).train()
    sft_mmlu, _ = snap(m, cfg.epochs * spe, "sft-final")
    results["sft"] = sft_mmlu
    wandb.finish()
    del m
    torch.cuda.empty_cache()

    # ============================================================================
    # SDFT
    # ============================================================================
    print("\n=== SDFT ===\n")
    wandb.init(project="qwench-h2h", name="sdft", config=cfg.__dict__,
               tags=["sdft", "fullft"], reinit=True)
    model = fresh_model()
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    rng2 = random.Random(cfg.seed)

    class SdftSnap(TrainerCallback):
        def on_step_end(self, args, state, control, model=None, **kw):
            if state.global_step % spe == 0:
                snap(model, state.global_step, "sdft")

    class SDFTTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            batch = inputs["batch"]
            losses, clens = [], []

            # 1. rollout (batched, on-policy)
            prompts = [render(student_messages(ex)) for ex in batch]
            conts = _gen_batch(model, prompts, cfg.max_new_tokens, True)
            items = [(tok(p, add_special_tokens=False).input_ids, c, ex)
                     for p, c, ex in zip(prompts, conts, batch, strict=True) if c]

            # 2. student scoring
            s_l = [(response_logits(model, pid, c, cfg.max_len), c)
                   for pid, c, _ in items]

            # 3. teacher scoring (deepcopy EMA teacher, no grad)
            with torch.no_grad():
                t_pids = [
                    tok(render(teacher_messages(ex, pick_demo(ex, train_pool, rng2))),
                        add_special_tokens=False).input_ids
                    for _, _, ex in items
                ]
                t_l = [(response_logits(teacher, tp, c, cfg.max_len), c)
                       for tp, (_, c) in zip(t_pids, s_l, strict=True)]

            for (sl, c), (tl, _) in zip(s_l, t_l, strict=True):
                if sl is None or tl is None:
                    continue
                mask = torch.ones(1, len(c), device=sl.device)
                losses.append(analytic_token_kl(sl.unsqueeze(0), tl.unsqueeze(0), mask))
                clens.append(len(c))

            if losses:
                loss = torch.stack(losses).mean()
            else:
                loss = sum(p.sum() for p in model.parameters() if p.requires_grad) * 0.0
            self._kl = loss.item()
            self._diag = {
                "rollout/usable_frac": len(losses) / max(len(batch), 1),
                "rollout/cont_len_mean": (sum(clens) / len(clens)) if clens else 0.0,
            }
            return loss

        def log(self, logs, *a, **k):
            if getattr(self, "_kl", None) is not None:
                logs["train/sdft_reverse_kl"] = self._kl
            logs.update(getattr(self, "_diag", {}))
            super().log(logs, *a, **k)

    class EmaCb(TrainerCallback):
        def on_step_end(self, args, state, control, model=None, **kw):
            a = cfg.ema_alpha
            for pt, ps in zip(teacher.parameters(), model.parameters(), strict=True):
                pt.data.mul_(1 - a).add_(ps.data.detach(), alpha=a)

    def collate(f):
        return {"batch": f}

    SDFTTrainer(model=model, args=TrainingArguments(
        output_dir="/tmp/sdft", num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size, gradient_accumulation_steps=4,
        learning_rate=cfg.lr, lr_scheduler_type="constant", warmup_ratio=0.0,
        logging_steps=10, bf16=True, report_to="wandb", run_name="sdft-h2h",
        save_strategy="no", remove_unused_columns=False,
    ), data_collator=collate, train_dataset=train_rows,
        callbacks=[EmaCb(), SdftSnap()]).train()
    sdft_mmlu, _ = snap(model, cfg.epochs * spe, "sdft-final")
    results["sdft"] = sdft_mmlu
    wandb.finish()

    # --- report -----------------------------------------------------------------
#    ... (SDFT region calls _gen_batch, a local helper defined below this block) ...
    print("\n=== H2H FORGETTING ===")
    sft_drop = results["sft"] - base_mmlu
    sdft_drop = results["sdft"] - base_mmlu
    print(f"model: {model_name}  lr={lr}  epochs={epochs}")
    print(f"  base  mmlu={base_mmlu:.3f}")
    print(f"  sft   mmlu={results['sft']:.3f}  ({sft_drop:+.3f})")
    print(f"  sdft  mmlu={results['sdft']:.3f}  ({sdft_drop:+.3f})")
    return results


@app.local_entrypoint()
def main(model_name: str = "Qwen/Qwen3-4B", epochs: int = 30, lr: float = 5e-5,
         mmlu_n: int = 200):
    print(h2h.remote(model_name, epochs, lr, mmlu_n))
