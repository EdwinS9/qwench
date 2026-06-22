"""Cheap test of the forgetting PREMISE: does full fine-tuning on our (tiny) data
actually damage general ability?

Single-GPU full fine-tune (no LoRA, no FSDP) of a model that fits on one 80GB GPU
(default Qwen3-4B), measuring MMLU and plan-success before vs. after. If full-FT SFT
clearly drops MMLU here, forgetting is real on our data and the expensive 8B-FSDP
SFT-vs-SDFT comparison is justified. If it doesn't drop, our data is too small to
induce forgetting — and we've learned that for ~1 cheap run instead of many 4-GPU ones.

    modal run training/forgetting_premise.py                       # Qwen3-4B, 3 epochs
    modal run training/forgetting_premise.py --mmlu-n 64 --sft-epochs 1   # smoke
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

app = modal.App("qwench-premise", image=image)
hf_cache = modal.Volume.from_name("qwench-hf-cache", create_if_missing=True)


@app.function(gpu="A100-80GB", timeout=4 * 60 * 60,
              volumes={"/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_name("wandb-secret")])
def run(model_name: str, sft_epochs: int, mmlu_n: int, eval_n: int):
    import random
    import sys

    sys.path.insert(0, "/root")

    import torch
    import wandb
    from datasets import Dataset, load_dataset
    from transformers import AutoModelForCausalLM
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

    cfg = TrainConfig(method="sft", model=model_name, use_lora=False, epochs=sft_epochs,
                      batch_size=4)
    wandb.init(project="qwench-premise", name=f"fullft-{model_name.split('/')[-1]}",
               config=cfg.__dict__)
    tok = load_tokenizer(model_name)
    train_rows = load_examples("train")
    heldout = load_examples("heldout")
    if eval_n:
        heldout = heldout[:eval_n]
    rng = random.Random(cfg.seed)

    mmlu = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=0).select(range(mmlu_n))
    mmlu_items = [{"question": r["question"], "choices": r["choices"], "answer": r["answer"]}
                  for r in mmlu]

    def measure(m, label):
        plan = evaluate(m, tok, heldout, train_rows, cfg, rng)[0]["eval/plan_success"]
        acc = mmlu_accuracy(m, tok, mmlu_items)
        print(f"[{label}] plan_success={plan:.3f}  mmlu={acc:.3f}")
        return {"plan_success": round(plan, 4), "mmlu": round(acc, 4)}

    # --- base (before fine-tuning) ---
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda").eval()
    base_r = measure(base, "base")
    del base
    torch.cuda.empty_cache()

    # --- full fine-tuning (no LoRA, single GPU) ---
    model = load_model(cfg)  # use_lora=False -> plain model, full-parameter training
    ds = Dataset.from_list([to_prompt_completion(tok, r) for r in train_rows])
    sft_cfg = SFTConfig(
        output_dir="/root/ckpt", num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size, gradient_accumulation_steps=2,
        learning_rate=cfg.lr, lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=2,
        bf16=True, max_length=cfg.max_len, gradient_checkpointing=True,
        report_to="wandb", run_name="premise-sft", save_strategy="no",
        remove_unused_columns=False,
    )
    SFTTrainer(model=model, args=sft_cfg, train_dataset=ds, processing_class=tok).train()
    sft_r = measure(model, "full-ft-sft")

    drop = round(base_r["mmlu"] - sft_r["mmlu"], 4)
    wandb.log({"premise/base_mmlu": base_r["mmlu"], "premise/sft_mmlu": sft_r["mmlu"],
               "premise/mmlu_drop": drop,
               "premise/base_plan": base_r["plan_success"],
               "premise/sft_plan": sft_r["plan_success"]})
    print("\n=== FORGETTING PREMISE (full fine-tuning) ===")
    print(f"model: {model_name}")
    print(f"{'':>12} {'plan':>7} {'mmlu':>7}")
    print(f"{'base':>12} {base_r['plan_success']:>7.3f} {base_r['mmlu']:>7.3f}")
    print(f"{'full-ft-sft':>12} {sft_r['plan_success']:>7.3f} {sft_r['mmlu']:>7.3f}")
    print(f"MMLU drop from full-FT SFT: {drop:+.3f}  "
          f"({'forgetting IS measurable' if drop > 0.02 else 'little/no forgetting on this data'})")
    print(f"W&B: {wandb.run.url}")
    wandb.finish()
    return {"base": base_r, "sft": sft_r, "mmlu_drop": drop}


@app.local_entrypoint()
def main(model_name: str = "Qwen/Qwen3-4B", sft_epochs: int = 3, mmlu_n: int = 500,
         eval_n: int = 0):
    print(run.remote(model_name, sft_epochs, mmlu_n, eval_n))
