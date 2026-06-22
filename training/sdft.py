"""Phase 4 — SDFT trainer on Modal, live-logged to W&B.

Implements the paper's loop with a custom Trainer:
  1. ROLLOUT  — student generates its own plan for each instruction (on-policy).
  2. DUAL FWD — score those response tokens under the student (with grad) and under
                the EMA teacher (= same model + the in-context demonstration, no grad).
  3. LOSS     — analytic per-token reverse KL (qwench/sdft_loss.py).
  4. EMA      — after each optimizer step, teacher_weights ← (1-α)·teacher + α·student.

Teacher = a second frozen LoRA adapter (the EMA of the student adapter) over the SAME
shared frozen base, with the demonstration in its prompt — so no full second model copy
is held, and SDFT fits a 40-48GB GPU.

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
# Persists trained adapters across runs (the previous run saved nothing and was lost).
ckpts = modal.Volume.from_name("qwench-checkpoints", create_if_missing=True)
CKPT_DIR = "/root/checkpoints"


@app.function(gpu="A100-80GB", timeout=6 * 60 * 60,
              volumes={"/root/.cache/huggingface": hf_cache, CKPT_DIR: ckpts},
              secrets=[modal.Secret.from_name("wandb-secret")])
def train(limit: int, epochs: int, lr: float, ema_alpha: float):
    import logging
    import random
    import sys
    import warnings

    sys.path.insert(0, "/root")

    import torch
    import wandb
    from transformers import Trainer, TrainerCallback, TrainingArguments

    from qwench.prompts import pick_demo, render_chat, student_messages, teacher_messages
    from qwench.sdft_loss import analytic_token_kl
    from training.common import (
        STUDENT_ADAPTER,
        TEACHER_ADAPTER,
        PlanEvalCallback,
        TrainConfig,
        add_ema_teacher_adapter,
        define_wandb_metrics,
        generate_batched,
        load_examples,
        load_model,
        load_tokenizer,
        response_logits,
        sync_teacher_from_student,
    )

    cfg = TrainConfig(method="sdft", epochs=epochs, lr=lr, ema_alpha=ema_alpha,
                      run_name="sdft", batch_size=4)
    wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=cfg.__dict__,
               tags=["sdft"])
    define_wandb_metrics()

    tok = load_tokenizer(cfg.model)
    train_rows = load_examples("train")
    heldout = load_examples("heldout")
    if limit:
        train_rows = train_rows[:limit]
    train_pool = load_examples("train")
    model = load_model(cfg)
    # EMA teacher = a second frozen LoRA adapter on the SAME base (no full model copy).
    add_ema_teacher_adapter(model, cfg)
    rng = random.Random(cfg.seed)
    log = logging.getLogger("qwench.sdft")

    class SDFTTrainer(Trainer):
        def _prepare_inputs(self, inputs):
            return inputs  # raw example dicts, not tensors

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            # Run rollout AND student scoring with dropout disabled (eval mode), so the
            # optimized distribution matches the on-policy sampling distribution and the
            # teacher targets. eval() disables LoRA dropout without blocking gradients.
            was_training = model.training
            model.eval()
            batch = inputs["batch"]
            losses, cont_lens = [], []
            try:
                # 1. ROLLOUT — sample all plans in the micro-batch at once (student adapter).
                model.set_adapter(STUDENT_ADAPTER)
                prompts = [render_chat(tok, student_messages(ex)) for ex in batch]
                conts = generate_batched(model, tok, prompts, max_new_tokens=cfg.max_new_tokens,
                                         do_sample=True, temperature=1.0, top_p=0.95,
                                         batch_size=len(batch))
                # keep only examples that produced a non-empty rollout
                items = [(tok(p, add_special_tokens=False).input_ids, c, ex)
                         for p, c, ex in zip(prompts, conts, batch, strict=True) if c]

                # 2. STUDENT scoring (grad flows) — student adapter is active.
                s_logits = [response_logits(model, pids, c, cfg.max_len) for pids, c, _ in items]

                # 3. TEACHER scoring (no grad) — one adapter swap for the whole batch.
                t_pids = []
                for _, _, ex in items:
                    demo = pick_demo(ex, train_pool, rng)
                    t_prompt = render_chat(tok, teacher_messages(ex, demo))
                    t_pids.append(tok(t_prompt, add_special_tokens=False).input_ids)
                try:
                    model.set_adapter(TEACHER_ADAPTER)
                    with torch.no_grad():
                        t_logits = [response_logits(model, tp, c, cfg.max_len)
                                    for tp, (_, c, _) in zip(t_pids, items, strict=True)]
                finally:
                    model.set_adapter(STUDENT_ADAPTER)

                # 4. LOSS — analytic per-token reverse KL, per example.
                for (_, c, _), sl, tl in zip(items, s_logits, t_logits, strict=True):
                    if sl is None or tl is None:
                        warnings.warn(f"skipped example: continuation ({len(c)} toks) "
                                      f"exceeds max_len={cfg.max_len}", stacklevel=2)
                        continue
                    mask = torch.ones(1, len(c), device=sl.device)
                    losses.append(analytic_token_kl(sl.unsqueeze(0), tl.unsqueeze(0), mask))
                    cont_lens.append(len(c))
            finally:
                if was_training:
                    model.train()

            if losses:
                loss = torch.stack(losses).mean()
            else:
                log.warning("SDFT step %s produced no usable continuations; "
                            "contributing zero gradient.", self.state.global_step)
                # graph-connected zero so backward runs cleanly (zero grad, not a no-op leaf)
                loss = sum(p.sum() for p in model.parameters() if p.requires_grad) * 0.0
            self._last_kl = loss.item()
            # Cheap per-step diagnostics so the dashboard has smooth curves between evals.
            self._step_metrics = {
                "rollout/usable_frac": len(losses) / max(len(batch), 1),
                "rollout/cont_len_mean": (sum(cont_lens) / len(cont_lens)) if cont_lens else 0.0,
            }
            return (loss, None) if return_outputs else loss

        def log(self, logs, *args, **kwargs):
            # Ride the Trainer's own monotonic step counter instead of a manual wandb.log,
            # so these sit alongside loss/lr/eval without step-collision warnings.
            if getattr(self, "_last_kl", None) is not None:
                logs["train/sdft_reverse_kl"] = self._last_kl
            logs.update(getattr(self, "_step_metrics", {}))
            return super().log(logs, *args, **kwargs)

    class EMACallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            sync_teacher_from_student(model, alpha=cfg.ema_alpha)

    def collate(features):
        return {"batch": features}

    best_dir = f"{CKPT_DIR}/sdft-best"
    args = TrainingArguments(
        output_dir=f"{CKPT_DIR}/sdft", num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size, gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=1, bf16=True, report_to="wandb", run_name=cfg.run_name,
        save_strategy="no", remove_unused_columns=False,  # we save the best adapter ourselves
    )
    # Save the STUDENT adapter whenever heldout plan-success hits a new best (strict >),
    # so we keep the earliest, least-over-trained checkpoint at peak performance.
    trainer = SDFTTrainer(
        model=model, args=args, data_collator=collate,
        train_dataset=train_rows,  # plain list of example dicts; collate() batches them
        callbacks=[EMACallback(),
                   PlanEvalCallback(tok, heldout[:cfg.eval_examples], train_pool, cfg,
                                    save_dir=best_dir, save_adapter=STUDENT_ADAPTER)],
    )
    trainer.train()

    # Always also save the final adapter, then persist the volume.
    model.set_adapter(STUDENT_ADAPTER)
    model.save_pretrained(f"{CKPT_DIR}/sdft-final", selected_adapters=[STUDENT_ADAPTER])
    tok.save_pretrained(f"{CKPT_DIR}/sdft-final")
    ckpts.commit()
    print(f"W&B run: {wandb.run.url}")
    print("adapters saved to Modal volume qwench-checkpoints: sdft-best/ and sdft-final/")
    wandb.finish()


@app.local_entrypoint()
def main(limit: int = 0, epochs: int = 2, lr: float = 1e-5, ema_alpha: float = 0.02,
         gpu: str = ""):
    # --gpu overrides the default (A100-80GB), e.g. --gpu L40S or --gpu H100.
    fn = train.with_options(gpu=gpu) if gpu else train
    fn.remote(limit, epochs, lr, ema_alpha)
