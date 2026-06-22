"""Multi-GPU full fine-tuning on Modal (milestone 1: SFT under FSDP).

Wraps the torchrun launch: one Modal function gets N GPUs and spawns N processes via
torchrun, which run training/fullft_sft_entry.py with FSDP. This is the standard,
lower-risk half of the forgetting comparison (full-FT SFT). The SDFT-under-FSDP half is
the harder follow-up milestone.

    modal run training/fullft.py --epochs 2                 # full
    modal run training/fullft.py --limit 64 --epochs 1      # smoke
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
N_GPU = 4

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers>=4.51", "trl>=1.0", "accelerate", "datasets", "wandb", "torch")
    .add_local_dir(str(REPO / "qwench"), remote_path="/root/qwench")
    .add_local_dir(str(REPO / "schemas"), remote_path="/root/schemas")
    .add_local_dir(str(REPO / "training"), remote_path="/root/training")
    .add_local_dir(str(REPO / "data"), remote_path="/root/data")
)

app = modal.App("qwench-fullft", image=image)
hf_cache = modal.Volume.from_name("qwench-hf-cache", create_if_missing=True)
ckpts = modal.Volume.from_name("qwench-checkpoints", create_if_missing=True)


@app.function(gpu=f"A100-80GB:{N_GPU}", timeout=6 * 60 * 60,
              volumes={"/root/.cache/huggingface": hf_cache, "/root/checkpoints": ckpts},
              secrets=[modal.Secret.from_name("wandb-secret")])
def train_sft(epochs: int, limit: int, lr: float):
    import subprocess

    cmd = [
        "torchrun", f"--nproc_per_node={N_GPU}",
        "/root/training/fullft_sft_entry.py",
        "--epochs", str(epochs), "--limit", str(limit), "--lr", str(lr),
    ]
    print("launching:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd="/root")
    ckpts.commit()
    print("full-FT SFT done; checkpoint committed to qwench-checkpoints volume")


@app.local_entrypoint()
def main(epochs: int = 2, limit: int = 0, lr: float = 1e-5):
    train_sft.remote(epochs, limit, lr)
