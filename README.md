# qwench

**Self-Distillation Fine-Tuning (SDFT) of Qwen3-4B for robot skill planning.**

A from-the-paper reproduction of SDFT ([arXiv:2601.19897](https://arxiv.org/html/2601.19897v1)),
applied to a high-level robot **skill-planning** task and evaluated in
[ManiSkill](https://github.com/haosulab/ManiSkill). The goal is to learn the SDFT method
on something useful — an LLM planner that maps natural-language instructions to a sequence
of robot skill calls — that transfers to building real LLM-as-planner agents later.

---

## What SDFT is (and why it fits this task)

Plain supervised fine-tuning (SFT) on new demonstrations causes **catastrophic forgetting**:
the model learns the new skill but degrades on everything else. On-policy RL avoids this but
needs a reward function you usually don't have. **SDFT gets on-policy-style learning from
demonstrations alone, no reward model**, by making the model its own teacher.

- **Student** — the model answering the query alone: `π_θ(y | x)`
- **Teacher** — the *same* model, but with a demonstration `c` placed in-context: `π(y | x, c)`

The teacher is better only because it can see an example. SDFT distills that in-context-boosted
behavior into the weights so the student no longer needs the example.

### The loop (paper §3)

1. **Rollout** — student generates its own plan for instruction `x` (on-policy).
2. **Dual scoring** — score those student-generated tokens under both the student `π_θ(·|x)`
   and the teacher `π(·|x,c)` (teacher = same model + demo in prompt).
3. **Loss** — minimize **reverse KL** `D_KL(π_θ(·|x) ‖ π(·|x,c))` over the response tokens,
   using the **analytic per-token KL estimator** (marginalize over the vocab at each position —
   the paper found this most stable):

   ```
   L = Σ_t  softmax(student_logits_t) · (log_softmax(student_logits_t) − log_softmax(teacher_logits_t))
   ```
   masked to response tokens only.
4. **EMA teacher** — the teacher uses an exponential-moving-average copy of the student weights
   (`θ_ema ← (1−α)·θ_ema + α·θ`, with `α ∈ {0.01, 0.02, 0.05}`), not frozen and not live.

### Teacher prompt template (paper)

> `<Instruction>` This is an example response to a similar instruction: `<Demonstration>`
> Now answer with a response of your own, including the thinking process:

---

## ⚠️ The 4B caveat — the go/no-go gate

The paper's headline model is **Qwen2.5-7B** and it explicitly notes SDFT **degrades below ~7B**,
because the whole method leans on the model's in-context-learning ability: the teacher is only
worth distilling if the demonstration genuinely makes the model better.

**Before any training**, run the *teacher-beats-student* check: compare base Qwen3-4B planning
*with* a demo vs. *without* on a held-out set. If the demo-conditioned model isn't clearly more
accurate, **SDFT has nothing to distill — stop here.** This is the project's go/no-go gate.

Also avoid asking for behaviors the base model can't partially do with a demo (the paper notes
SDFT can't handle *dramatic* behavioral shifts) — which is exactly why this project targets
**high-level skill planning** (tokens the model can already produce) and **not** low-level
control / VLA action prediction.

---

## Task framing: LLM-as-planner

- **Input:** a natural-language instruction + a text/JSON description of the scene state.
- **Output:** a sequence of **skill calls** drawn from a fixed API (see
  [`schemas/skills.json`](schemas/skills.json)).
- **Executor / grader:** ManiSkill executes the plan; success is checked automatically
  (does the plan parse → valid skills → valid args → goal reached in sim).

This is the paper's strongest result (tool use), retargeted to robot primitives. It's
auto-gradable, the teacher boost is easy to verify, and it's the canonical pattern used by
SayCan / Code-as-Policies.

---

## Plan of record

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 0 | Skill API JSON schemas | ✅ `schemas/` |
| 1 | Templated data-gen from ManiSkill task definitions → `(instruction, state) → gold plan` pairs + sim validator | ⬜ |
| 2 | **Teacher-beats-student gate** — base Qwen3-4B with/without demo on held-out set | ⬜ (go/no-go) |
| 3 | SFT baseline + general-capability eval (measure forgetting) | ⬜ |
| 4 | SDFT trainer — fork TRL `GKDTrainer`: student rollouts → student & teacher (EMA + demo-in-prompt) forward passes → analytic per-token reverse-KL → EMA update | ⬜ |
| 5 | Compare SDFT vs SFT on (a) plan accuracy, (b) forgetting | ⬜ |

### Decisions locked in
- **Base model:** Qwen3-4B
- **Sim / grader:** ManiSkill
- **Data:** templated from ManiSkill task definitions (deterministic, no API key)
- **Trainer:** fork of TRL `GKDTrainer` (closest existing on-policy KD trainer)

### Reference hyperparameters (paper)
- Full-parameter fine-tune, AdamW, cosine LR + warmup
- LR ∈ {5e-6, 1e-5, 5e-5}; batch ∈ {16, 32, 64}; 2 epochs (skills)
- EMA rate α ∈ {0.01, 0.02, 0.05}
- Cost ≈ **2.5× FLOPs / ~4× wall-clock vs SFT** (on-policy generation each step)
- Hardware: full FT of 4B in bf16 + Adam ≈ one 80GB GPU (or FSDP across 2×40GB); LoRA to iterate cheaply

---

## Layout

```
schemas/        Skill API + plan + scene-state JSON schemas (Phase 0)
data/           (Phase 1) generated instruction→plan pairs + validator
eval/           (Phase 2/5) teacher-beats-student gate + forgetting eval
training/       (Phase 3/4) SFT baseline + SDFT (TRL GKDTrainer fork)
```

Nothing under `data/`, `eval/`, `training/` exists yet — this first pass is **plan + schemas only**.
