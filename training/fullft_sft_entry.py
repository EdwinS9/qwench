"""torchrun entry script: full fine-tuning of Qwen3-8B under FSDP (SFT baseline).

Launched by training/fullft.py via `torchrun --nproc_per_node=N`. Not a Modal function —
it runs once per GPU process. Full-parameter SFT (no LoRA), FSDP full-shard so the 8B fits
across the GPUs, gradient checkpointing for activation memory. Rank 0 logs to W&B and saves
the gathered full-state model to the checkpoints volume.

    torchrun --nproc_per_node=4 training/fullft_sft_entry.py --epochs 2
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, "/root")

import torch
import wandb
from datasets import Dataset
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from training.common import load_examples, load_tokenizer, to_prompt_completion

CKPT = "/root/checkpoints/sft-fullft"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()
    rank = int(os.environ.get("RANK", "0"))

    tok = load_tokenizer(args.model)
    rows = load_examples("train")
    if args.limit:
        rows = rows[:args.limit]
    ds = Dataset.from_list([to_prompt_completion(tok, r) for r in rows])

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)

    if rank == 0:
        wandb.init(project="qwench-fullft", name="sft-fullft",
                   config={"model": args.model, "epochs": args.epochs, "method": "sft-fullft"})

    cfg = SFTConfig(
        output_dir=CKPT, num_train_epochs=args.epochs,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=1, bf16=True, max_length=2048, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fsdp="full_shard auto_wrap",
        fsdp_config={"transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"],
                     "activation_checkpointing": False},
        report_to=("wandb" if rank == 0 else "none"),
        save_strategy="no", remove_unused_columns=False,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()
    # Trainer.save_model gathers the full FSDP state dict on rank 0.
    trainer.save_model(f"{CKPT}-final")
    if rank == 0:
        tok.save_pretrained(f"{CKPT}-final")
        print(f"saved full-FT SFT model to {CKPT}-final")
        wandb.finish()


if __name__ == "__main__":
    main()
