"""Shared training scaffold for SFT (Phase 3) and SDFT (Phase 4).

Holds everything the two trainers have in common so their metrics are directly
comparable on the same W&B dashboard:
  - config, data loading, prompt/target formatting
  - LoRA model loading (student + EMA-teacher adapters share one frozen base)
  - the realtime eval callback: plan-success %, failure-stage breakdown,
    teacher-student KL gap, and a table of sample generations

All heavy imports (torch/transformers/wandb) resolve inside the Modal image.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import wandb
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

from qwench.grade import grade
from qwench.prompts import pick_demo, student_messages, teacher_messages
from qwench.sdft_loss import analytic_token_kl

REPO_REMOTE = "/root"  # where the repo is mounted inside the Modal image


@dataclass
class TrainConfig:
    method: str = "sft"                      # "sft" | "sdft"
    model: str = "Qwen/Qwen3-8B"
    lr: float = 1e-5
    epochs: int = 2
    batch_size: int = 16
    grad_accum: int = 1
    max_len: int = 2048
    max_new_tokens: int = 512
    # LoRA (default, fits one 80GB GPU). Set use_lora=False for full FT (needs more GPU).
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    # SDFT-only
    ema_alpha: float = 0.02                  # teacher EMA rate (paper: {0.01,0.02,0.05})
    # eval / logging
    eval_steps: int = 25
    eval_examples: int = 64
    n_samples_logged: int = 6
    seed: int = 0
    wandb_project: str = "qwench-sdft"
    run_name: str = ""


def load_examples(split: str) -> list[dict[str, Any]]:
    path = Path(REPO_REMOTE) / "data" / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_tokenizer(model: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def lora_config(cfg: TrainConfig) -> LoraConfig:
    return LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )


def load_model(cfg: TrainConfig):
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    if cfg.use_lora:
        model = get_peft_model(model, lora_config(cfg))  # creates the "default" (student) adapter
        model.print_trainable_parameters()
    model.config.use_cache = False
    return model


STUDENT_ADAPTER = "default"
TEACHER_ADAPTER = "teacher"


def add_ema_teacher_adapter(model, cfg: TrainConfig):
    """Add a second, frozen LoRA adapter on the SAME frozen base as the EMA teacher.

    The student and teacher share the 16GB base weights — only their small LoRA
    adapters differ — so SDFT no longer needs a full second model copy (~32GB of
    weights -> ~17GB), letting it fit on a 40-48GB GPU. The teacher adapter is the
    EMA of the student adapter; the SDFT trainer activates it (no grad) for the
    teacher forward pass and updates it after each step.
    """
    if not cfg.use_lora:
        raise ValueError("dual-adapter EMA teacher requires use_lora=True")
    model.add_adapter(TEACHER_ADAPTER, lora_config(cfg))
    # Only the student adapter trains; the teacher adapter is frozen and EMA-updated.
    # add_adapter can re-mark trainability, so set both sides explicitly.
    for name, p in model.named_parameters():
        if "lora_" in name and f".{TEACHER_ADAPTER}." in name:
            p.requires_grad_(False)
        elif "lora_" in name and f".{STUDENT_ADAPTER}." in name:
            p.requires_grad_(True)
    model.set_adapter(STUDENT_ADAPTER)
    sync_teacher_from_student(model, alpha=1.0)  # start teacher == student
    return model


@torch.no_grad()
def sync_teacher_from_student(model, alpha: float):
    """teacher ← (1-alpha)·teacher + alpha·student over the LoRA params. alpha=1 hard-copies."""
    params = dict(model.named_parameters())
    for name, p in params.items():
        if "lora_" in name and f".{STUDENT_ADAPTER}." in name:
            tname = name.replace(f".{STUDENT_ADAPTER}.", f".{TEACHER_ADAPTER}.")
            if tname in params:
                params[tname].mul_(1 - alpha).add_(p.detach(), alpha=alpha)


# --- scoring -----------------------------------------------------------------
def response_logits(model, prompt_ids: list[int], cont_ids: list[int], max_len: int):
    """Logits predicting each continuation token, given a prompt.

    Left-truncates the PROMPT (never the continuation) when prompt+continuation exceed
    `max_len`, so the returned logits always cover the full continuation — both roles
    score exactly the same tokens. Returns None if the continuation alone exceeds
    max_len (caller must skip, never contribute a spurious zero). Reads the device live
    (the Trainer moves the model to GPU only at train() time).
    """
    keep = max_len - len(cont_ids)
    if keep <= 0:
        return None
    pids = prompt_ids[-keep:] if len(prompt_ids) > keep else prompt_ids
    device = next(model.parameters()).device
    ids = torch.tensor([pids + cont_ids], device=device)
    logits = model(ids).logits[0]
    start = len(pids) - 1
    out = logits[start:start + len(cont_ids)]
    assert out.size(0) == len(cont_ids), (out.size(0), len(cont_ids))
    return out


# --- SFT data formatting ---------------------------------------------------
def to_prompt_completion(tok, example: dict[str, Any]) -> dict[str, str]:
    """Prompt-completion pair for TRL SFTTrainer (prompt tokens are auto-masked)."""
    prompt = tok.apply_chat_template(
        student_messages(example), tokenize=False, add_generation_prompt=True
    )
    completion = json.dumps(example["target"]) + tok.eos_token
    return {"prompt": prompt, "completion": completion}


# --- realtime metrics ------------------------------------------------------
@torch.no_grad()
def _gold_continuation_kl_gap(model, tok, examples, train_pool, rng, max_len):
    """Mean token KL( student(·|x) ‖ teacher(·|x,c) ) over gold continuations.

    The paper's health signal: a demonstration-conditioned teacher should stay *close*
    to the student (low gap) while being more correct. SDFT should shrink this gap over
    training; SFT need not. Computed with the *current* model in both roles, through the
    SAME analytic_token_kl estimator as the training loss, so the number is comparable
    across SFT and SDFT.
    """
    def render(messages):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    gaps = []
    for ex in examples:
        demo = pick_demo(ex, train_pool, rng)
        cont_ids = tok(json.dumps(ex["target"]) + tok.eos_token, add_special_tokens=False).input_ids
        s_pids = tok(render(student_messages(ex)), add_special_tokens=False).input_ids
        t_pids = tok(render(teacher_messages(ex, demo)), add_special_tokens=False).input_ids

        s = response_logits(model, s_pids, cont_ids, max_len)
        t = response_logits(model, t_pids, cont_ids, max_len)
        if s is None or t is None:
            continue  # continuation exceeds budget — skip, do not record a spurious 0.0
        mask = torch.ones(1, len(cont_ids), device=s.device)
        gaps.append(analytic_token_kl(s.unsqueeze(0), t.unsqueeze(0), mask).item())
    return sum(gaps) / max(len(gaps), 1)


def define_wandb_metrics():
    """Tidy the W&B dashboard: track best/last per panel and group by prefix.

    Metric prefixes (`train/`, `eval/`, `eval/success/`, `eval/fail/`, `rollout/`) render
    as collapsible sections; the summaries surface the headline numbers in the run table.
    """
    wandb.define_metric("eval/plan_success", summary="max")
    wandb.define_metric("eval/teacher_student_kl_gap", summary="min")
    wandb.define_metric("eval/success/*", summary="max")
    wandb.define_metric("train/sdft_reverse_kl", summary="min")


@torch.no_grad()
def evaluate(model, tok, examples, train_pool, cfg, rng):
    """Generate plans on a heldout slice, grade them, and gather sample rows.

    Returns (metrics, samples). Metrics include the aggregate plan-success, a per-family
    breakdown (`eval/success/<family>`), the failure-stage mix (`eval/fail/<stage>`), and
    the teacher-student KL gap.
    """
    from collections import Counter

    device = next(model.parameters()).device
    succ, stages = 0, Counter()
    fam_succ, fam_total = Counter(), Counter()
    samples = []
    model.eval()
    for i, ex in enumerate(examples):
        prompt = tok.apply_chat_template(
            student_messages(ex), tokenize=False, add_generation_prompt=True
        )
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        gen = model.generate(**ids, max_new_tokens=cfg.max_new_tokens, do_sample=False,
                             use_cache=True, pad_token_id=tok.pad_token_id)
        text = tok.decode(gen[0, ids.input_ids.size(1):], skip_special_tokens=True)
        v = grade(ex, text)
        fam = ex["task_family"]
        succ += int(v["success"])
        stages[v["stage"]] += 1
        fam_total[fam] += 1
        fam_succ[fam] += int(v["success"])
        if i < cfg.n_samples_logged:
            samples.append([fam, ex["instruction"], text[:600], v["success"], v["stage"]])
    kl_gap = _gold_continuation_kl_gap(model, tok, examples[:cfg.n_samples_logged * 4],
                                       train_pool, rng, cfg.max_len)
    model.train()
    n = len(examples)
    return {
        "eval/plan_success": succ / n,
        "eval/teacher_student_kl_gap": kl_gap,
        **{f"eval/success/{fam}": fam_succ[fam] / fam_total[fam] for fam in sorted(fam_total)},
        **{f"eval/fail/{k}": c / n for k, c in stages.items() if k != "ok"},
    }, samples


class PlanEvalCallback(TrainerCallback):
    """Every cfg.eval_steps, log plan-success / KL-gap / sample generations to W&B."""

    def __init__(self, tok, eval_examples, train_pool, cfg: TrainConfig):
        import random
        self.tok = tok
        self.eval_examples = eval_examples
        self.train_pool = train_pool
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

    def on_step_end(self, args, state, control, model=None, **kw):
        if state.global_step == 0 or state.global_step % self.cfg.eval_steps != 0:
            return
        metrics, samples = evaluate(model, self.tok, self.eval_examples, self.train_pool,
                                    self.cfg, self.rng)
        step = state.global_step
        table = wandb.Table(
            columns=["step", "family", "instruction", "generated_plan", "success", "stage"],
            data=[[step, *row] for row in samples],
        )
        wandb.log({**metrics, "eval/samples": table}, step=step)
