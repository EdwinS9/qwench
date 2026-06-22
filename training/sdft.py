"""Phase 4 — SDFT trainer on Modal, live-logged to W&B.

Implements the paper's loop with a custom Trainer:
  1. ROLLOUT  — student generates its own plan for each instruction (on-policy).
  2. DUAL FWD — score those response tokens under the student (with grad) and under
                the EMA teacher (= same model + the in-context demonstration, no grad).
  3. LOSS     — analytic per-token reverse KL (qwench/sdft_loss.py).
  4. EMA      — after each optimizer step, teacher_weights ← (1-α)·teacher + α·student.

Teacher = EMA copy of the student's LoRA adapter over the shared frozen base, with the
demonstration in its prompt. LoRA keeps student + EMA teacher on one 80GB GPU.

    modal run training/sdft.py                          # Qwen3-8B, LoRA, 2 epochs
    modal run training/sdft.py --limit 128 --epochs 1   # quick smoke

NOTE: this trainer has not yet been validated on a GPU — the first Modal run is its
integration test. Watch the W&B loss + KL-gap curves on the smoke run before a full one.
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers>=4.51", "peft", "accelerate", "datasets", "wandb", "torch")
    .add_local_dir(str(REPO / "qwench"), remote_path="/root/qwench")
    .add_local_dir(str(REPO / "schemas"), remote_path="/root/schemas")
    .add_local_dir(str(REPO / "training"), remote_path="/root/training")
    .add_local_dir(str(REPO / "data"), remote_path="/root/data")
)

app = modal.App("qwench-sdft", image=image)
hf_cache = modal.Volume.from_name("qwench-hf-cache", create_if_missing=True)


@app.function(gpu="A100-80GB", timeout=6 * 60 * 60,
              volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_name("wandb-secret")])
def train(limit: int, epochs: int, lr: float, ema_alpha: float):
    import copy
    import json
    import random
    import sys

    sys.path.insert(0, "/root")

    import torch
    import wandb
    from transformers import Trainer, TrainingArguments, TrainerCallback

    from qwench.prompts import pick_demo, student_messages, teacher_messages
    from qwench.sdft_loss import analytic_token_kl
    from training.common import (TrainConfig, PlanEvalCallback, load_examples,
                                 load_model, load_tokenizer)

    cfg = TrainConfig(method="sdft", epochs=epochs, lr=lr, ema_alpha=ema_alpha,
                      run_name="sdft", batch_size=4)
    wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=cfg.__dict__,
               tags=["sdft"])

    tok = load_tokenizer(cfg.model)
    train_rows = load_examples("train")
    heldout = load_examples("heldout")
    if limit:
        train_rows = train_rows[:limit]
    train_pool = load_examples("train")
    model = load_model(cfg)
    device = next(model.parameters()).device

    # EMA teacher: frozen copy of the (LoRA) student, tracks it slowly.
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    rng = random.Random(cfg.seed)

    def response_logits(m, prompt_ids, cont_ids):
        """Logits predicting each continuation token, given a prompt."""
        ids = torch.tensor([prompt_ids + cont_ids], device=device)[:, :cfg.max_len]
        logits = m(ids).logits[0]
        start = len(prompt_ids) - 1
        return logits[start:start + len(cont_ids)]  # [len(cont), V]

    class SDFTTrainer(Trainer):
        def _prepare_inputs(self, inputs):
            return inputs  # raw example dicts, not tensors

        @torch.no_grad()
        def _rollout(self, ex):
            prompt = tok.apply_chat_template(student_messages(ex), tokenize=False,
                                            add_generation_prompt=True)
            pids = tok(prompt, add_special_tokens=False).input_ids
            inp = torch.tensor([pids], device=device)
            gen = self.model.generate(inp, max_new_tokens=cfg.max_new_tokens, do_sample=True,
                                      temperature=1.0, top_p=0.95, pad_token_id=tok.pad_token_id)
            cont_ids = gen[0, len(pids):].tolist()
            return pids, cont_ids

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            losses = []
            for ex in inputs["batch"]:
                s_pids, cont_ids = self._rollout(ex)
                if not cont_ids:
                    continue
                demo = pick_demo(ex, train_pool, rng)
                t_prompt = tok.apply_chat_template(teacher_messages(ex, demo), tokenize=False,
                                                  add_generation_prompt=True)
                t_pids = tok(t_prompt, add_special_tokens=False).input_ids

                s_logits = response_logits(model, s_pids, cont_ids)          # grad
                with torch.no_grad():
                    t_logits = response_logits(ema, t_pids, cont_ids)        # EMA teacher
                n = min(s_logits.size(0), t_logits.size(0))
                mask = torch.ones(1, n, device=device)
                losses.append(analytic_token_kl(s_logits[:n].unsqueeze(0),
                                                t_logits[:n].unsqueeze(0), mask))
            loss = torch.stack(losses).mean() if losses else torch.zeros(1, device=device, requires_grad=True).sum()
            wandb.log({"train/sdft_reverse_kl": loss.item()}, step=self.state.global_step)
            return (loss, None) if return_outputs else loss

    @torch.no_grad()
    def ema_update():
        a = cfg.ema_alpha
        for p_ema, p in zip(ema.parameters(), model.parameters()):
            if p.requires_grad:  # only the trainable (LoRA) params move
                p_ema.mul_(1 - a).add_(p.detach(), alpha=a)

    class EMACallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            ema_update()

    def collate(features):
        return {"batch": features}

    args = TrainingArguments(
        output_dir="/root/checkpoints/sdft", num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size, gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=1, bf16=True, report_to="wandb", run_name=cfg.run_name,
        save_strategy="no", remove_unused_columns=False,
    )
    trainer = SDFTTrainer(
        model=model, args=args, data_collator=collate,
        train_dataset=train_rows,  # plain list of example dicts; collate() batches them
        callbacks=[EMACallback(),
                   PlanEvalCallback(tok, heldout[:cfg.eval_examples], train_pool, cfg)],
    )
    trainer.train()
    print(f"W&B run: {wandb.run.url}")
    wandb.finish()


@app.local_entrypoint()
def main(limit: int = 0, epochs: int = 2, lr: float = 1e-5, ema_alpha: float = 0.02,
         gpu: str = ""):
    # --gpu overrides the default (A100-80GB), e.g. --gpu L40S or --gpu H100.
    fn = train.with_options(gpu=gpu) if gpu else train
    fn.remote(limit, epochs, lr, ema_alpha)
