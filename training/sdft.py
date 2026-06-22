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


@app.function(gpu="A100-80GB", timeout=6 * 60 * 60,
              volumes={"/root/.cache/huggingface": hf_cache},
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

    from qwench.prompts import pick_demo, student_messages, teacher_messages
    from qwench.sdft_loss import analytic_token_kl
    from training.common import (
        STUDENT_ADAPTER,
        TEACHER_ADAPTER,
        PlanEvalCallback,
        TrainConfig,
        add_ema_teacher_adapter,
        define_wandb_metrics,
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

        @torch.no_grad()
        def _rollout(self, ex):
            # On-policy sampling must come from the student; assert it self-sufficiently
            # rather than rely on whatever adapter the previous step left active.
            self.model.set_adapter(STUDENT_ADAPTER)
            prompt = tok.apply_chat_template(student_messages(ex), tokenize=False,
                                            add_generation_prompt=True)
            pids = tok(prompt, add_special_tokens=False).input_ids
            inp = torch.tensor([pids], device=next(self.model.parameters()).device)
            # use_cache is False during training; enable it for generation or each rollout
            # recomputes full attention per token (orders of magnitude slower).
            gen = self.model.generate(inp, max_new_tokens=cfg.max_new_tokens, do_sample=True,
                                      temperature=1.0, top_p=0.95, use_cache=True,
                                      pad_token_id=tok.pad_token_id)
            return pids, gen[0, len(pids):].tolist()

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            # Run rollout AND student scoring with dropout disabled (eval mode), so the
            # optimized distribution matches the on-policy sampling distribution and the
            # teacher targets. eval() disables LoRA dropout without blocking gradients.
            was_training = model.training
            model.eval()
            losses, cont_lens = [], []
            n_batch = len(inputs["batch"])
            try:
                for ex in inputs["batch"]:
                    s_pids, cont_ids = self._rollout(ex)
                    if not cont_ids:
                        continue
                    demo = pick_demo(ex, train_pool, rng)
                    t_prompt = tok.apply_chat_template(teacher_messages(ex, demo), tokenize=False,
                                                      add_generation_prompt=True)
                    t_pids = tok(t_prompt, add_special_tokens=False).input_ids

                    s_logits = response_logits(model, s_pids, cont_ids, cfg.max_len)  # grad flows
                    try:
                        model.set_adapter(TEACHER_ADAPTER)                # activate EMA teacher
                        with torch.no_grad():
                            t_logits = response_logits(model, t_pids, cont_ids, cfg.max_len)
                    finally:
                        model.set_adapter(STUDENT_ADAPTER)                # always restore student
                    if s_logits is None or t_logits is None:
                        warnings.warn(f"skipped example: continuation ({len(cont_ids)} toks) "
                                      f"exceeds max_len={cfg.max_len}", stacklevel=2)
                        continue
                    mask = torch.ones(1, len(cont_ids), device=s_logits.device)
                    losses.append(analytic_token_kl(s_logits.unsqueeze(0),
                                                    t_logits.unsqueeze(0), mask))
                    cont_lens.append(len(cont_ids))
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
                "rollout/usable_frac": len(losses) / max(n_batch, 1),
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
