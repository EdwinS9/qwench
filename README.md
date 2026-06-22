# qwench

**Self-Distillation Fine-Tuning (SDFT) of Qwen3-8B for robot skill planning.**

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

## The go/no-go gate (now a formality, but still run it)

The paper's headline model is **Qwen2.5-7B** and it notes SDFT **degrades below ~7B**, because
the whole method leans on the model's in-context-learning ability: the teacher is only worth
distilling if the demonstration genuinely makes the model better. **Qwen3-8B sits just above
that threshold**, which is exactly why it was chosen — this de-risks the project relative to a
4B base.

It's still cheap and worth confirming. **Before any training**, run the *teacher-beats-student*
check: compare base Qwen3-8B planning
*with* a demo vs. *without* on a held-out set. If the demo-conditioned model isn't clearly more
accurate, **SDFT has nothing to distill — stop here.** This is the project's go/no-go gate.

Also avoid asking for behaviors the base model can't partially do with a demo (the paper notes
SDFT can't handle *dramatic* behavioral shifts) — which is exactly why this project targets
**high-level skill planning** (tokens the model can already produce) and **not** low-level
control / VLA action prediction.

> **Why 8B and not 4B:** an earlier draft targeted Qwen3-4B, but the paper warns SDFT weakens
> below ~7B. Bumping to **Qwen3-8B** clears the threshold while still fitting full fine-tuning
> on a single 80GB GPU, so we get the paper's regime without a multi-GPU project.

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

## Where the prompts come from

Each training example is `(instruction, scene_state) → gold_plan`. The input is the prompt; the
gold plan is the target. None of it is hand-labeled and there is no API dependency — **ManiSkill
is both the source and the grader.**

- **`scene_state`** — read from ManiSkill ground truth (object poses, on-relations, articulation,
  gripper, reachability) by a small **state extractor**, serialized to `schemas/scene_state.json`.
- **`gold_plan`** — **constructed procedurally** from each task's `(initial state → goal condition)`
  by a tiny per-family solver, then **executed in the sim and kept only if the goal is reached.**
  We do not ask a model to guess plans.
- **`instruction`** — **templated paraphrases** of the task goal (e.g. "put the red cube in the
  bin" / "drop the cube into the bin"), templates × task object names for language variety.

**Task families (all four):** PickAndPlace · Open/Close articulated · Push · Stack/multi-object.
**Language:** pure templates first (deterministic, no API); add a one-shot LLM paraphrase pass
later *only if* Phase 2 shows the model is over-fitting template phrasing.

> Known limits: template phrasing can be rigid (mitigation: optional LLM paraphrase pass), and
> coverage is bounded by ManiSkill tasks + the 8-skill vocabulary. Fine for learning the method;
> not an open-world planner.

## Plan of record

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 0 | Skill API JSON schemas | ✅ `schemas/` |
| 1 | Templated data-gen → `(instruction, state) → gold plan` pairs, every plan executed & goal-verified. 800 examples in `data/` (`qwench/`, `python -m qwench.generate`). Uses a symbolic executor now; ManiSkill executor swaps into step 4 for the GPU pass. | ✅ (symbolic) |
| 2 | **Teacher-beats-student gate** — base Qwen3-8B with/without demo on heldout (`modal run eval/gate.py`). Reports student vs teacher plan-success; passes if teacher leads by ≥15%. | ✅ built (run on Modal) |
| 3 | SFT baseline + general-capability eval (measure forgetting) | ⬜ |
| 4 | SDFT trainer — fork TRL `GKDTrainer`: student rollouts → student & teacher (EMA + demo-in-prompt) forward passes → analytic per-token reverse-KL → EMA update | ⬜ |
| 5 | Compare SDFT vs SFT on (a) plan accuracy, (b) forgetting | ⬜ |

### Decisions locked in
- **Base model:** Qwen3-8B
- **Sim / grader:** ManiSkill
- **Data:** templated from ManiSkill task definitions (deterministic, no API key)
- **Trainer:** fork of TRL `GKDTrainer` (closest existing on-policy KD trainer)
- **Compute:** rented GPU on **Modal** (single 80GB H100/A100 for full FT)

### Reference hyperparameters (paper)
- Full-parameter fine-tune, AdamW, cosine LR + warmup
- LR ∈ {5e-6, 1e-5, 5e-5}; batch ∈ {16, 32, 64}; 2 epochs (skills)
- EMA rate α ∈ {0.01, 0.02, 0.05}
- Cost ≈ **2.5× FLOPs / ~4× wall-clock vs SFT** (on-policy generation each step)
- Hardware: full FT of 8B in bf16 + Adam ≈ one 80GB GPU (Modal H100/A100); LoRA to iterate cheaply during dev

---

## Layout

```
schemas/        Skill API + plan + scene-state JSON schemas (Phase 0) ✅
qwench/         data-gen + skills/world/grader + shared prompts (Phase 1) ✅
data/           generated train/heldout JSONL (800 verified examples) ✅
tests/          solve+verify, validator, grader-plumbing tests ✅
eval/gate.py    Phase 2 teacher-beats-student gate (Modal) ✅
training/       (Phase 3/4) SFT baseline + SDFT (TRL GKDTrainer fork) ⬜
```

### Running the gate (Phase 2)
```
pip install modal && modal run eval/gate.py            # full heldout on Qwen3-8B
modal run eval/gate.py --limit 32                       # quick smoke
```
Auth: profile `build-small-hackathon` is already active in `~/.modal.toml`.
