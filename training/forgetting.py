"""Phase 5 — the forgetting comparison (SFT vs SDFT), the experiment that justifies SDFT.

One unattended Modal run that:
  1. trains an SFT baseline on the full data and saves its best adapter,
  2. evaluates base Qwen3-8B, the SFT adapter, and the existing SDFT adapter (sdft-best)
     on BOTH the planning task (plan-success) and a general-capability probe (MMLU),
  3. reports each model's plan-success and MMLU, and the MMLU drop vs. base (= forgetting).

The hypothesis: SFT and SDFT both reach high plan-success, but SFT degrades MMLU (forgets)
while SDFT preserves it. That preserved general capability is SDFT's reason to exist.

    modal run training/forgetting.py                     # full: SFT 3 epochs, 800 MMLU items
    modal run training/forgetting.py --mmlu-n 24 --sft-epochs 1   # quick smoke
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

app = modal.App("qwench-forgetting", image=image)
hf_cache = modal.Volume.from_name("qwench-hf-cache", create_if_missing=True)
ckpts = modal.Volume.from_name("qwench-checkpoints", create_if_missing=True)
CKPT_DIR = "/root/checkpoints"


@app.function(gpu="A100-80GB", timeout=6 * 60 * 60,
              volumes={"/root/.cache/huggingface": hf_cache, CKPT_DIR: ckpts},
              secrets=[modal.Secret.from_name("wandb-secret")])
def run(mmlu_n: int, sft_epochs: int, eval_n: int):
    import gc
    import random
    import sys

    sys.path.insert(0, "/root")

    import torch
    import wandb
    from datasets import Dataset, load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer

    from training.common import (
        PlanEvalCallback,
        TrainConfig,
        define_wandb_metrics,
        evaluate,
        load_examples,
        load_model,
        load_tokenizer,
        to_prompt_completion,
    )
    from training.general_eval import mmlu_accuracy

    cfg = TrainConfig(method="sft", epochs=sft_epochs, run_name="forgetting")
    wandb.init(project="qwench-forgetting", name="sft-vs-sdft", config=cfg.__dict__)
    define_wandb_metrics()
    tok = load_tokenizer(cfg.model)
    train_rows = load_examples("train")
    heldout = load_examples("heldout")
    if eval_n:
        heldout = heldout[:eval_n]
    rng = random.Random(cfg.seed)

    # --- 1. Train the SFT baseline on the full data, save the best adapter ----------
    sft_best = f"{CKPT_DIR}/sft-best"
    ds = Dataset.from_list([to_prompt_completion(tok, r) for r in train_rows])
    model = load_model(cfg)
    sft_cfg = SFTConfig(
        output_dir=f"{CKPT_DIR}/sft", num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size, learning_rate=cfg.lr,
        lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=1, bf16=True,
        max_length=cfg.max_len, report_to="wandb", run_name="sft-forgetting",
        save_strategy="no", remove_unused_columns=False,
    )
    SFTTrainer(
        model=model, args=sft_cfg, train_dataset=ds, processing_class=tok,
        callbacks=[PlanEvalCallback(tok, heldout[:cfg.eval_examples], train_rows, cfg,
                                    save_dir=sft_best, save_adapter="default")],
    ).train()
    model.set_adapter("default")
    model.save_pretrained(sft_best, selected_adapters=["default"])  # ensure a saved adapter
    ckpts.commit()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # --- 2. Build the MMLU probe set ------------------------------------------------
    mmlu = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=0).select(range(mmlu_n))
    mmlu_items = [{"question": r["question"], "choices": r["choices"], "answer": r["answer"]}
                  for r in mmlu]

    # --- 3. Evaluate base, SFT, SDFT on plan-success AND MMLU -----------------------
    def fresh_base():
        return AutoModelForCausalLM.from_pretrained(
            cfg.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to("cuda").eval()

    targets = {
        "base": None,
        "sft": sft_best,
        "sdft": f"{CKPT_DIR}/sdft-best",
    }
    results = {}
    for name, adapter in targets.items():
        m = fresh_base()
        if adapter is not None:
            m = PeftModel.from_pretrained(m, adapter).eval()
        plan = evaluate(m, tok, heldout, train_rows, cfg, rng)[0]["eval/plan_success"]
        mmlu_acc = mmlu_accuracy(m, tok, mmlu_items)
        results[name] = {"plan_success": round(plan, 4), "mmlu": round(mmlu_acc, 4)}
        print(f"[{name}] plan_success={plan:.3f}  mmlu={mmlu_acc:.3f}")
        del m
        gc.collect()
        torch.cuda.empty_cache()

    # --- 4. Report: forgetting = base MMLU - fine-tuned MMLU ------------------------
    base_mmlu = results["base"]["mmlu"]
    table = wandb.Table(columns=["model", "plan_success", "mmlu", "mmlu_drop_vs_base"])
    for name in ("base", "sft", "sdft"):
        r = results[name]
        drop = round(base_mmlu - r["mmlu"], 4)
        table.add_data(name, r["plan_success"], r["mmlu"], drop)
        wandb.log({f"forgetting/{name}_plan_success": r["plan_success"],
                   f"forgetting/{name}_mmlu": r["mmlu"],
                   f"forgetting/{name}_mmlu_drop": drop})
    wandb.log({"forgetting/summary": table})
    print("\n=== FORGETTING COMPARISON ===")
    print(f"{'model':>6} {'plan':>7} {'mmlu':>7} {'mmlu_drop':>10}")
    for name in ("base", "sft", "sdft"):
        r = results[name]
        print(f"{name:>6} {r['plan_success']:>7.3f} {r['mmlu']:>7.3f} "
              f"{base_mmlu - r['mmlu']:>10.3f}")
    print(f"\nW&B: {wandb.run.url}")
    wandb.finish()
    return results


@app.local_entrypoint()
def main(mmlu_n: int = 800, sft_epochs: int = 3, eval_n: int = 0, gpu: str = ""):
    # eval_n=0 -> full heldout; set small (e.g. 24) for a fast smoke.
    fn = run.with_options(gpu=gpu) if gpu else run
    print(fn.remote(mmlu_n, sft_epochs, eval_n))
