#!/usr/bin/env bash
# Push the keys in .env to a Modal secret named `wandb-secret`, which
# training/sft.py and training/sdft.py inject into the GPU container.
#
#   ./scripts/push-secret.sh
#
# Re-run any time you change .env (uses --force to overwrite).
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] || { echo "no .env — copy .env.example to .env and fill in WANDB_API_KEY"; exit 1; }
set -a; source .env; set +a
[ -n "${WANDB_API_KEY:-}" ] || { echo "WANDB_API_KEY is empty in .env"; exit 1; }

args=(WANDB_API_KEY="$WANDB_API_KEY")
[ -n "${HF_TOKEN:-}" ] && args+=(HF_TOKEN="$HF_TOKEN")

modal secret create wandb-secret "${args[@]}" --force
echo "✅ pushed wandb-secret to Modal"
