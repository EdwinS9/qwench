"""Phase 3 — SFT baseline on Modal, live-logged to W&B.

Standard supervised fine-tuning on (instruction+scene -> gold plan JSON). This is the
forgetting-prone baseline SDFT is measured against. Logs the SAME metrics to the SAME
W&B project as SDFT so curves overlay directly.

    modal run training/sft.py                       # Qwen3-8B, LoRA, 2 epochs
    modal run training/sft.py --limit 128 --epochs 1   # quick smoke

Needs Modal auth (profile build-small-hackathon) + a `wandb-secret` Modal secret
holding WANDB_API_KEY (see README "Live dashboard").
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers>=4.51", "trl>=0.12", "peft", "accelerate",
                 "datasets", "wandb", "torch")
    .add_local_dir(str(REPO / "qwench"), remote_path="/root/qwench")
    .add_local_dir(str(REPO / "schemas"), remote_path="/root/schemas")
    .add_local_dir(str(REPO / "training"), remote_path="/root/training")
    .add_local_dir(str(REPO / "data"), remote_path="/root/data")
)

app = modal.App("qwench-sft", image=image)
hf_cache = modal.Volume.from_name("qwench-hf-cache", create_if_missing=True)


@app.function(gpu="A100-80GB", timeout=4 * 60 * 60,
              volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_name("wandb-secret")])
def train(limit: int, epochs: int, lr: float):
    import sys
    sys.path.insert(0, "/root")

    import wandb
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    from training.common import (
        PlanEvalCallback,
        TrainConfig,
        define_wandb_metrics,
        load_examples,
        load_model,
        load_tokenizer,
        to_prompt_completion,
    )

    cfg = TrainConfig(method="sft", epochs=epochs, lr=lr, run_name="sft-baseline")
    wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=cfg.__dict__,
               tags=["sft", "baseline"])
    define_wandb_metrics()

    tok = load_tokenizer(cfg.model)
    train_rows = load_examples("train")
    heldout = load_examples("heldout")
    if limit:
        train_rows = train_rows[:limit]
    ds = Dataset.from_list([to_prompt_completion(tok, r) for r in train_rows])

    model = load_model(cfg)
    sft_cfg = SFTConfig(
        output_dir="/root/checkpoints/sft", num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size, gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=1, bf16=True, max_seq_length=cfg.max_len,
        report_to="wandb", run_name=cfg.run_name, save_strategy="no",
    )
    trainer = SFTTrainer(
        model=model, args=sft_cfg, train_dataset=ds,
        callbacks=[PlanEvalCallback(tok, heldout[:cfg.eval_examples], load_examples("train"), cfg)],
    )
    trainer.train()
    print(f"W&B run: {wandb.run.url}")
    wandb.finish()


@app.local_entrypoint()
def main(limit: int = 0, epochs: int = 2, lr: float = 1e-5, gpu: str = ""):
    # --gpu overrides the default (A100-80GB); SFT holds one model so 40GB-class works too.
    fn = train.with_options(gpu=gpu) if gpu else train
    fn.remote(limit, epochs, lr)
