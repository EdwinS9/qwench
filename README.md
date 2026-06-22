# qwench

**Self-Distillation Fine-Tuning (SDFT) of Qwen3-8B for robot skill planning.**

`qwench` is a faithful, from-the-paper reimplementation of SDFT — *Self-Distillation
Enables Continual Learning* (Shenfeld, Damani, Hübotter & Agrawal,
[arXiv:2601.19897](https://arxiv.org/abs/2601.19897)) — applied to a high-level robot
**skill-planning** task: an LLM that maps a natural-language instruction and a scene
description to a sequence of robot skill calls. It exists to learn the method on a
useful, auto-gradable task and to study whether it holds at the 8B scale.

The headline question SDFT answers: *how do you teach a model a new skill from
demonstrations without it forgetting everything else?*

---

## What SDFT is

Ordinary supervised fine-tuning (SFT) on new demonstrations causes **catastrophic
forgetting** — the model gains the new skill but degrades elsewhere. On-policy RL avoids
this but needs a reward function you usually don't have. **SDFT recovers on-policy-style
learning from demonstrations alone, with no reward model**, by making the model its own
teacher:

- **Student** `π_θ(y | x)` — the model answering from the instruction + scene `x` alone.
- **Teacher** `π(y | x, c)` — the *same* model, additionally shown one in-context
  demonstration `c`. It is better only because it can see an example.

SDFT distills the teacher's in-context-boosted behavior back into the weights so the
student no longer needs the example. Because the student only ever imitates a slightly
better version of *itself*, it stays close to what it already knew — which is what
curbs forgetting.

Per training step:

1. **Rollout** — the student samples its own plan for `x` (on-policy).
2. **Dual scoring** — those response tokens are scored under both the student and the
   teacher (same model + demonstration in the prompt).
3. **Loss** — minimize the reverse KL `D_KL(π_θ(·|x) ‖ π(·|x,c))` over the response
   tokens, averaged per token via an analytic estimator (`qwench/sdft_loss.py`).
4. **EMA teacher** — the teacher tracks an exponential moving average of the student,
   updated after each optimizer step.

A short **go/no-go gate** precedes training: confirm that the demonstration-conditioned
teacher clearly out-plans the bare student. If it doesn't, SDFT has nothing to distill.

---

## Task: LLM-as-planner

- **Input** — a natural-language instruction plus a JSON description of the scene state.
- **Output** — an ordered list of **skill calls** from a fixed API
  ([`schemas/skills.json`](schemas/skills.json)): `navigate_to, pick, place, open,
  close, push, detect, done`.
- **Grader** — a symbolic world model (`qwench/world.py`) executes the plan and checks
  the goal condition, so every plan is auto-graded (parse → schema → execution → goal).
  A ManiSkill-backed executor is the planned drop-in for physical fidelity; the plan
  format and goal checks are unchanged.

This mirrors the paper's strongest result (tool use), retargeted to robot primitives,
and follows the LLM-as-planner pattern of SayCan / Code-as-Policies.

### Where the data comes from

Each example is `(instruction, scene_state) → gold_plan`, generated procedurally with no
hand-labeling and no API dependency — the world model is both the source and the grader:

- **scene_state** — a symbolic scene ([`schemas/scene_state.json`](schemas/scene_state.json)).
- **gold_plan** — constructed from the task's `(initial state → goal)` by a small
  per-family solver, then **executed and kept only if it reaches the goal**.
- **instruction** — templated phrasings of the goal (relation-consistent with the plan).

Four task families: PickAndPlace · Open/Close articulated · Push · Stack. Regenerate with
`python -m qwench.generate` (writes `data/train.jsonl` and `data/heldout.jsonl`).

---

## Layout

```
schemas/        Skill API + plan + scene-state JSON schemas
qwench/         data generation, symbolic world/executor, grader, shared prompts
  world.py        symbolic executor + goal checker (the ground-truth grader)
  tasks.py        scene + goal generators per family
  solver.py       procedural gold-plan construction
  skills.py       plan validation against the schemas
  prompts.py      student / teacher message construction
  grade.py        stage-aware plan grader
  sdft_loss.py    analytic per-token reverse-KL objective
  generate.py     generate → validate → execute/verify → split
data/           generated train/heldout JSONL
eval/gate.py    teacher-beats-student go/no-go gate (Modal)
training/       SFT baseline + SDFT trainer + shared eval/W&B scaffold (Modal)
tests/          world preconditions, validator, grader stages, KL-loss correctness
```

The SDFT trainer (`training/sdft.py`) is a custom subclass of HF `transformers.Trainer`
implementing the rollout → dual-forward → analytic reverse-KL → EMA loop by hand. (TRL's
`GKDTrainer` was the conceptual reference but is not used; `trl` is only the SFT
baseline's dependency.)

**Reproduction, hyperparameters, GPU notes, and run commands live in [SPEC.md](SPEC.md).**

---

## License

Apache-2.0 — see [LICENSE](LICENSE).

## Citation

This project reimplements the method from:

```bibtex
@misc{shenfeld2026selfdistillation,
  title         = {Self-Distillation Enables Continual Learning},
  author        = {Shenfeld, Idan and Damani, Mehul and H\"ubotter, Jonas and Agrawal, Pulkit},
  year          = {2026},
  eprint        = {2601.19897},
  archivePrefix = {arXiv}
}
```

See [CITATION.cff](CITATION.cff) to cite this repository.
