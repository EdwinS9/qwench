# Forgetting experiments — results log

Durable record of the SFT-vs-SDFT forgetting runs (the raw logs lived on ephemeral
cloud pods; this captures the numbers). MMLU = general-capability probe (200 questions
unless noted); plan-success = symbolic-grader held-out.

## 1. LoRA (Modal) — no forgetting
- Setup: LoRA SDFT/SFT, Qwen3-8B, ~1–2 epochs.
- Result: MMLU ≈0.71 for base, SFT, and SDFT — **no measurable forgetting** (drop ≈0).
- Reason: LoRA trains ~1% of params → inherently forgetting-resistant. Can't distinguish
  SFT vs SDFT on the forgetting axis.

## 2. Full-FT premise (4B, lr=1e-5, 3 epochs) — still no forgetting
- base mmlu 0.698 → full-FT SFT mmlu 0.688 (drop +0.01, noise); plan 0.30 → 1.00.
- Conclusion: mild training even at full-FT doesn't forget on our small data.

## 3. Forget sweep (4B, lr=5e-5, full-FT, per-epoch) — forgetting ACHIEVED
- base mmlu 0.73 → SFT mmlu ~0.51 by epoch 2, held ~0.48–0.52 through epoch 14.
- **~20-point MMLU drop** while plan-success = 1.00. This is the regime where forgetting
  appears (hot LR + over-training past convergence). The run was cut by a network blip
  at epoch 14, but the signal is clear.

## 4. H2H (4B, lr=5e-5, full-FT, 30 epochs) — SFT forgets, SDFT COLLAPSED
- base mmlu 0.690, plan 0.30
- **SFT**: mmlu 0.485 (−0.205), plan 0.985 — forgot ~20 pts, kept the task. ✓
- **SDFT**: held mmlu ~0.62 / plan ~0.44 for epochs 1–4, then **diverged at epoch 5**
  → mmlu 0.29 (≈ random), plan 0.00. Training instability, NOT a real finding.
- Diagnosis: self-referential reverse-KL target (teacher = student's own EMA) with no
  ground-truth anchor, run at lr=5e-5 / no-warmup / fast-EMA → feedback loop → runaway.
  Possibly compounded by the 4B in-context teacher being below the ~7B ICL threshold.

## Next: stable SDFT (fixes in training/forget_h2h.py)
lr 1e-5 · cosine+warmup · EMA 0.005 · rollout temp 0.7 · grad-clip 1.0 · collapse guard ·
live W&B logging. Question: does stabilized SDFT hold MMLU above SFT's 0.485 while reaching
similar plan-success? Run: `bash deploy/runpod_deploy.sh "python training/forget_h2h.py --run_sft False"`
