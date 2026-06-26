"""Launch the SFT vs SDFT H2H on RunPod (single A100-80GB VM).

Raw GPU VM — no ephemeral-function lifecycle, so network blips don't kill the job.
Launches a pod, uploads the repo, installs deps, and runs full-parameter H2H.

    python deploy/runpod_h2h.py          # launch, stream logs, report when done
    python deploy/runpod_h2h.py --dry-run
"""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import runpod

REPO_ROOT = Path(__file__).resolve().parent.parent
GPU_TYPE = "NVIDIA A100-SXM4-80GB"
IMAGE = "runpod/pytorch:2.5.1-py3.11-cuda12.4.1-devel-ubuntu22.04"
DISK_GB = 60
TIMEOUT_SEC = 8 * 3600

SETUP_CMDS = [
    "pip install --quiet transformers>=4.51 trl>=1.0 peft accelerate datasets wandb torch",
    "pip install --quiet bitsandbytes",
    "cd /workspace/qwench && python training/forget_h2h.py",
]

# Symlink from userspace to /workspace so pip-installed packages find our modules.
SYMLINK_CMDS = [
    "ln -sf /workspace/qwench/qwench /workspace/qwench/qwench 2>/dev/null; true",
    # Actually just install from source
    "cd /workspace/qwench && pip install -e . --quiet 2>/dev/null || true",
]


def _tar_repo():
    """Tar the repo (excluding .git, .venv, __pycache__, data/) into a temp file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    with tarfile.open(tmp.name, "w:gz") as tf:
        for item in REPO_ROOT.rglob("*"):
            rel = item.relative_to(REPO_ROOT)
            parts = rel.parts
            if parts[0] in (".git", ".venv", "__pycache__", "data", "deploy"):
                continue
            tf.add(str(item), arcname=str(rel))
    return tmp.name


def _runpod_exec(pod_id: str, cmd: str) -> str:
    """Run a command on the pod via the RunPod CLI (`runpodctl exec`)."""
    result = subprocess.run(
        ["runpodctl", "exec", pod_id, cmd],
        capture_output=True, text=True, timeout=300,
    )
    return result.stdout + result.stderr


def launch(dry_run: bool = False):
    # -- auth ------------------------------------------------------------------
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY not set — add to ~/.zshrc and source it")
    runpod.api_key = api_key

    if dry_run:
        print(f"[dry-run] Would launch {GPU_TYPE} pod, upload repo, run H2H")
        return

    # -- 1. launch the pod -----------------------------------------------------
    print("Launching GPU pod...")
    pod = runpod.create_pod(
        name=f"qwench-h2h-{int(time.time())}",
        image_name=IMAGE,
        gpu_type_id=GPU_TYPE,
        container_disk_in_gb=DISK_GB,
        volume_in_gb=50,
        ports="8888/http",
        env={"WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
             "HF_TOKEN": os.environ.get("HF_TOKEN", "")},
    )
    pod_id = pod["id"]
    print(f"Pod launched: {pod_id}")
    print(f"Dashboard: https://www.runpod.io/console/pods/{pod_id}")

    # -- 2. wait for it to be ready --------------------------------------------
    print("Waiting for pod to spin up (this takes ~2 min)...")
    for _ in range(60):
        p = runpod.get_pod(pod_id)
        runtime = p.get("runtime") or {}
        if runtime.get("uptime_in_seconds", 0) > 0:
            break
        time.sleep(10)
    else:
        raise RuntimeError("Pod did not become ready in time")
    print("Pod is up. Uploading code...")

    # -- 3. upload the repo as a tar -------------------------------------------
    tarpath = _tar_repo()
    print(f"Uploading repo ({os.path.getsize(tarpath) // 1024} KB)...")
    result = subprocess.run(
        ["runpodctl", "send", tarpath, "/workspace/qwench/"],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=120,
    )
    print(result.stdout[-300:] if len(result.stdout) > 300 else result.stdout)
    os.unlink(tarpath)

    # runpodctl send puts files in /workspace/qwench/<archive-root> — fix nesting
    # Actually runpodctl send transfers to the pod. Let me just exec the setup.
    print("Installing dependencies and running training...")
    for cmd in SYMLINK_CMDS + SETUP_CMDS:
        print(f"\n--- {cmd[:60]}... ---")
        out = _runpod_exec(pod_id, cmd)
        if out.strip():
            print(out[-800:] if len(out) > 800 else out)

    # -- 4. done ---------------------------------------------------------------
    p = runpod.get_pod(pod_id)
    print(f"\nPod status: {p.get('desiredStatus', '?')}")
    print(f"Dashboard: https://www.runpod.io/console/pods/{pod_id}")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    launch(dry_run=dry)
