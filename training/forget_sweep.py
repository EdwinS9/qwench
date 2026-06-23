"""Push SFT hard until it forgets: full fine-tune Qwen3-4B aggressively and track MMLU.

Single GPU, no FSDP. Trains full-parameter SFT for many epochs at a high learning rate
and logs MMLU (general capability) + plan-success ONCE PER EPOCH, so we can see whether —
and when — general ability starts to decay under heavy fine-tuning. This finds the regime
where plain SFT forgets, which is the prerequisite for a meaningful SFT-vs-SDFT comparison.

    modal run training/forget_sweep.py                       # 4B, 30 epochs, lr 5e-5
    modal run training/forget_sweep.py --epochs 6 --lr 1e-4  # shorter/hotter
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers>=4.51", "trl>=1.0", "peft", "accelerate",
                 "datasets", "wandb", "torch", "bitsandbytes")
    .add_local_dir(str(REPO / "qwench"), remote_path="/root/qwench")
    .add_local_dir(str(REPO / "schemas"), remote_path="/root/schemas")
    .add_local_dir(str(REPO / "training"), remote_path="/root/training")
    .add_local_dir(str(REPO / "data"), remote_path="/root/data")
)

app = modal.App("qwench-forget-sweep", image=image)
hf_cache = modal.Volume.from_name("qwench-hf-cache", create_if_missing=True)


@app.function(gpu="A100-80GB", timeout=5 * 60 * 60,
              volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_name("wandb-secret")])
def run(model_name: str, epochs: int, lr: float, mmlu_n: int, eval_n: int):
    import random
    import sys

    sys.path.insert(0, "/root")

    import torch
    import wandb
    from datasets import Dataset, load_dataset
    from transformers import AutoModelForCausalLM, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    from training.common import (
        TrainConfig,
        evaluate,
        load_examples,
        load_model,
        load_tokenizer,
        to_prompt_completion,
    )
    from training.general_eval import mmlu_accuracy

    cfg = TrainConfig(method="sft", model=model_name, use_lora=False, epochs=epochs,
                      lr=lr, batch_size=4)
    wandb.init(project="qwench-forget-sweep",
               name=f"fullft-{model_name.split('/')[-1]}-lr{lr}-e{epochs}", config=cfg.__dict__)
    tok = load_tokenizer(model_name)
    train_rows = load_examples("train")
    heldout = load_examples("heldout")
    probe = heldout[:eval_n] if eval_n else heldout
    rng = random.Random(cfg.seed)

    mmlu = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=0).select(range(mmlu_n))
    mmlu_items = [{"question": r["question"], "choices": r["choices"], "answer": r["answer"]}
                  for r in mmlu]
    steps_per_epoch = max(1, len(train_rows) // cfg.batch_size)

    def snapshot(model, step, tag):
        acc = mmlu_accuracy(model, tok, mmlu_items)
        plan = evaluate(model, tok, probe, train_rows, cfg, rng)[0]["eval/plan_success"]
        wandb.log({"forget/mmlu": acc, "forget/plan_success": plan}, step=step)
        print(f"[{tag}] step={step} mmlu={acc:.3f} plan_success={plan:.3f}")
        return acc, plan

    # base reference (step 0)
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda").eval()
    base_mmlu, _ = snapshot(base, 0, "base")
    del base
    torch.cuda.empty_cache()

    class ForgetCallback(TrainerCallback):
        """Once per epoch, log MMLU + plan-success to trace the forgetting curve."""
        def on_step_end(self, args, state, control, model=None, **kw):
            if state.global_step % steps_per_epoch == 0:
                snapshot(model, state.global_step, f"epoch~{state.global_step // steps_per_epoch}")

    model = load_model(cfg)  # use_lora=False -> full-parameter SFT
    ds = Dataset.from_list([to_prompt_completion(tok, r) for r in train_rows])
    sft_cfg = SFTConfig(
        output_dir="/root/ckpt", num_train_epochs=cfg.epochs,
        # Proven 4B full-FT config (same as the premise run that ran cleanly): normal
        # AdamW + grad checkpointing, batch 4. No 8-bit optimizer — that combo is what
        # hung the 8B run. 4B full-FT (~48GB) fits one 80GB GPU comfortably.
        per_device_train_batch_size=cfg.batch_size, gradient_accumulation_steps=1,
        learning_rate=cfg.lr, lr_scheduler_type="constant", warmup_ratio=0.0,
        logging_steps=10, bf16=True, max_length=cfg.max_len, gradient_checkpointing=True,
        report_to="wandb", run_name="forget-sweep", save_strategy="no",
        remove_unused_columns=False,
    )
    trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=ds, processing_class=tok,
                         callbacks=[ForgetCallback()])
    trainer.train()
    final_mmlu, final_plan = snapshot(model, trainer.state.global_step, "final")

    drop = round(base_mmlu - final_mmlu, 4)
    wandb.log({"forget/base_mmlu": base_mmlu, "forget/final_mmlu": final_mmlu,
               "forget/total_mmlu_drop": drop})
    print("\n=== FORGET SWEEP ===")
    print(f"model {model_name}  lr {lr}  epochs {epochs}")
    print(f"base MMLU {base_mmlu:.3f} -> final MMLU {final_mmlu:.3f}  (drop {drop:+.3f})")
    print(f"final plan_success {final_plan:.3f}")
    print(f"verdict: {'SFT FORGETS (MMLU dropped)' if drop > 0.03 else 'still no forgetting'}")
    print(f"W&B: {wandb.run.url}")
    wandb.finish()
    return {"base_mmlu": base_mmlu, "final_mmlu": final_mmlu, "drop": drop}


@app.local_entrypoint()
def main(model_name: str = "Qwen/Qwen3-4B", epochs: int = 30, lr: float = 5e-5,
         mmlu_n: int = 200, eval_n: int = 48):
    print(run.remote(model_name, epochs, lr, mmlu_n, eval_n))
