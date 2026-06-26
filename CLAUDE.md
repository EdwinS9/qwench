# qwench — agent instructions

## ⚠️ ALWAYS stop GPU pods after a run finishes

Cloud GPUs bill **per hour while running** — an idle pod left up after training silently
drains the balance. (This already cost ~€50: a pod sat idle for ~1.5 days after a 2.5h run
because it wasn't stopped.)

**Rule:** the moment a training/eval run completes (or fails, or you've pulled the results),
**terminate the pod immediately.**

- RunPod: `runpodctl pod remove <pod-id>` (terminate — also frees disk; results go to W&B so
  nothing is lost). `pod stop` keeps the volume and still costs a little.
- Set `--stop-after <iso-datetime>` (e.g. now + 3h) on launch as a **backstop only** — do not
  rely on it as the primary mechanism (it has failed to fire). Explicit teardown is the rule.
- After any run, verify: `runpodctl pod list` → `[]`, and `runpodctl user` → spend `$0/hr`.
- Modal: detached apps also keep running — confirm with `modal app list` and stop if needed.

## Project context

SDFT (Self-Distillation Fine-Tuning, arXiv:2601.19897) reproduction on Qwen3 for robot
skill planning. Code map is in `README.md`; reproduction details + decisions in `SPEC.md`.

- Training/eval run on cloud GPUs (Modal `training/*.py`, RunPod `deploy/*`).
- RunPod CLI: `/opt/homebrew/bin/runpodctl`, authed via `~/.runpod/config.toml`,
  SSH key `~/.runpod/ssh/runpodctl-ssh-key`.
- Prefer cheaper GPUs sized to the job: full-FT 4B needs ~48GB (A40/A6000 ≈ $0.5/hr),
  not A100-80GB ($1.49/hr). LoRA/eval fit a 24GB card (RTX 4090).
