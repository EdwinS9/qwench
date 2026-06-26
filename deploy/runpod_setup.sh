#!/bin/bash
# Paste this into the RunPod Web Terminal (Jupyter → Terminal) after launching a pod.
#
# Pod config:
#   GPU:  A100-SXM4-80GB (or any 80GB A100)
#   Image: runpod/pytorch:2.5.1-py3.11-cuda12.4.1-devel-ubuntu22.04
#   Disk:  60GB
#
# Then paste this entire script. It installs deps, clones the repo, and runs the
# SFT vs SDFT head-to-head forgetting comparison. W&B logs to your account.

set -e

echo "=== 1. install python deps ==="
pip install --quiet transformers>=4.51 trl>=1.0 peft accelerate datasets wandb torch bitsandbytes

echo "=== 2. clone repo ==="
# Replace with a private clone URL + token if the repo is private:
git clone https://github.com/EdwinS9/qwench.git /workspace/qwench || \
  git clone https://YOUR_GITHUB_TOKEN@github.com/EdwinS9/qwench.git /workspace/qwench

echo "=== 3. run training ==="
cd /workspace/qwench
python training/forget_h2h.py

echo "=== DONE ==="
