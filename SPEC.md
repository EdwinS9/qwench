# qwench — project spec & reproduction guide

Plan-of-record, design decisions, hyperparameters, and run commands. For the conceptual
overview of SDFT and the repo layout, see [README.md](README.md).

## Plan of record

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 0 | Skill API JSON schemas (`schemas/`) | done |
| 1 | Procedural data generation → `(instruction, scene) → gold plan`, every plan executed and goal-verified (`qwench/`, `python -m qwench.generate`). Symbolic executor today; ManiSkill executor is the planned drop-in. | done (symbolic) |
| 2 | Teacher-beats-student gate — base Qwen3-8B with/without demo on heldout (`eval/gate.py`); passes if the teacher leads by ≥ 15%. | built |
| 3 | SFT baseline (`training/sft.py`), live W&B logging. | built |
| 4 | SDFT trainer (`training/sdft.py`): on-policy rollout → student & EMA-teacher dual forward → analytic per-token reverse-KL → EMA update. | built, pending GPU validation |
| 5 | Compare SDFT vs SFT on (a) plan accuracy, (b) forgetting. | todo |

## Decisions

- **Base model: Qwen3-8B.** The paper's headline model is Qwen2.5-7B and it notes SDFT
  degrades below ~7B, since the method leans entirely on in-context-learning ability —
  the teacher is only worth distilling if the demonstration genuinely improves the model.
  8B clears that threshold while still fitting LoRA fine-tuning on a single GPU. (An
  earlier draft targeted Qwen3-4B; bumped to 8B to stay in the paper's working regime.)
- **Task = high-level skill planning, not low-level control.** SDFT distills tokens the
  base model can already produce with a demo; it cannot bridge dramatic behavioral shifts
  (the paper's stated limitation), which rules out VLA / continuous-action prediction.
- **Grader = symbolic world model now, ManiSkill later.** The plan format and goal checks
  are executor-agnostic, so the ManiSkill executor is a localized swap.
- **Data = procedural + templated** (deterministic, no API key). Add a one-shot LLM
  paraphrase pass over instructions only if the gate shows the model over-fits phrasing.
- **Trainer = custom `transformers.Trainer` subclass** (not a TRL `GKDTrainer` fork); the
  SDFT loss/rollout/EMA are hand-written in `training/sdft.py` + `qwench/sdft_loss.py`.

## The go/no-go gate (Phase 2)

Run before any training to confirm there is a teacher signal to distill:

```bash
pip install modal
modal run eval/gate.py --limit 32     # quick smoke
modal run eval/gate.py                 # full heldout on Qwen3-8B
```

It runs base Qwen3-8B over the heldout set twice (student vs. teacher-with-demo), grades
both, and reports the plan-success gap. If the teacher does not clearly beat the student,
stop — the model can't exploit the demonstration at this scale and SDFT won't help.

## Reference hyperparameters (paper)

- Optimizer AdamW, cosine LR schedule with warmup.
- LR ∈ {5e-6, 1e-5, 5e-5}; batch ∈ {16, 32, 64}; ~2 epochs for skill learning.
- EMA rate α ∈ {0.01, 0.02, 0.05}.
- On-policy generation makes SDFT ≈ 2.5× FLOPs / ~4× wall-clock vs. SFT.
- This repo defaults to **LoRA** (`TrainConfig.use_lora=True`) so the student and EMA
  teacher share one frozen base; set `use_lora=False` for full fine-tuning (more GPU).

## Training with the live W&B dashboard (Phases 3–4)

Both trainers stream to one W&B project (`qwench-sdft`) so SFT and SDFT curves overlay.

One-time setup — put your key in `.env`, then push it to a Modal secret the trainers read:

```bash
cp .env.example .env          # paste your key from https://wandb.ai/authorize
./scripts/push-secret.sh      # creates Modal secret `wandb-secret` from .env
```

`.env` is gitignored; the key is injected into the GPU container at runtime, never baked
into code or the image.

```bash
# smoke runs first (watch the W&B URL each prints)
modal run training/sft.py  --limit 128 --epochs 1
modal run training/sdft.py --limit 128 --epochs 1
# full runs
modal run training/sft.py
modal run training/sdft.py
```

Logged in realtime, grouped into dashboard sections by prefix:

- `train/` — loss (SDFT reverse-KL), LR, grad-norm, every step.
- `rollout/` — per-step SDFT diagnostics: usable-example fraction and mean continuation
  length, so the curves stay smooth between the (costlier) evals.
- `eval/` — every `eval_steps`: aggregate `plan_success`, the teacher–student KL gap
  (the paper's health signal, expected to shrink under SDFT), a per-family success
  breakdown (`eval/success/<family>`), a failure-stage mix (`eval/fail/<stage>`), and a
  `samples` table of generated plans (with a step column to scrub over time).

`define_wandb_metrics()` marks `plan_success`/`success/*` as `max` and the KL metrics as
`min`, so the run table surfaces best values automatically.

## Compute

- **Auth:** needs a Modal account — run `modal token new` once (or
  `modal profile activate <name>` for a named profile).
- **GPU:** defaults to `A100-80GB`; override per run with `--gpu` (e.g. `--gpu L40S`).
  SDFT holds the student and the EMA teacher as two LoRA adapters over one shared frozen
  base (~17GB of weights, not ~32GB), so it fits a 40–48GB GPU. SFT holds a single model.
  A LoRA run over the 800-example dataset is well under 5h on any of these.

## Status / caveats

- The SDFT trainer has not yet been validated on a GPU; its first Modal run is the
  integration test. Start with the `--limit 128 --epochs 1` smoke and watch the loss and
  KL-gap curves before a full run.
- Grading is against the symbolic world model, not ManiSkill yet — fine for measuring the
  relative SDFT-vs-SFT gap; physical fidelity comes with the ManiSkill executor.
- Coverage is bounded by the four task families and the 8-skill vocabulary: a method
  study, not an open-world planner.
