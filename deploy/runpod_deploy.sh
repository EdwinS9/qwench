#!/bin/bash
# Launch a qwench training run on RunPod, then ALWAYS terminate the pod.
#
#   bash deploy/runpod_deploy.sh "python training/forget_h2h.py --run_sft False"
#
# Defaults to a cheap A40 (48GB, ~$0.50/hr) — enough for full-FT 4B, ~⅓ the A100 cost.
# Sets a hard --stop-after backstop AND removes the pod when training finishes.
set -euo pipefail
cd "$(dirname "$0")/.."

CMD="${1:-python training/forget_h2h.py}"          # training command to run on the pod
GPU="${GPU:-NVIDIA A40}"                            # cheap 48GB; override e.g. GPU="NVIDIA A100-SXM4-80GB"
CLOUD="${CLOUD:-COMMUNITY}"                         # cheaper than SECURE
IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
KEY="$HOME/.runpod/ssh/runpodctl-ssh-key"
STOP_AT=$(date -u -v+3H +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d "+3 hours" +%Y-%m-%dT%H:%M:%SZ)

RUN_TS=$(date +%Y%m%d-%H%M%S)
SAVE_DIR="runs/$RUN_TS"

cleanup() {  # ALWAYS save results, THEN terminate the pod — even on error / Ctrl-C
    if [ -n "${POD_ID:-}" ]; then
        # 1. SAVE before destroying: pull the run log + any saved adapter back to runs/
        if [ -n "${IP:-}" ] && [ -n "${PORT:-}" ]; then
            mkdir -p "$SAVE_DIR"
            echo ">>> saving results to $SAVE_DIR/ before teardown"
            scp -o StrictHostKeyChecking=no -i "$KEY" -P "$PORT" \
                root@"$IP":/workspace/run.log "$SAVE_DIR/run.log" 2>/dev/null \
                && echo "    saved run.log" || echo "    (no run.log to save)"
            # pull any saved adapter dir (small for LoRA; full-FT models are skipped on the pod)
            scp -r -o StrictHostKeyChecking=no -i "$KEY" -P "$PORT" \
                root@"$IP":/workspace/qwench/out "$SAVE_DIR/out" 2>/dev/null \
                && echo "    saved out/" || true
        fi
        # 2. teardown
        echo ">>> terminating pod $POD_ID"
        runpodctl pod remove "$POD_ID" 2>&1 | head -2 || true
        echo ">>> verify (should be empty):"; runpodctl pod list 2>&1 | tr -d '\r' | head -3
    fi
}
trap cleanup EXIT INT TERM

# -- secrets (W&B from ~/.netrc, HF from .env) --------------------------------
WANDB_KEY=$(python3 -c "import re,os; t=open(os.path.expanduser('~/.netrc')).read() if os.path.exists(os.path.expanduser('~/.netrc')) else ''; m=re.search(r'machine api\.wandb\.ai.*?password\s+(\S+)', t, re.S); print(m.group(1) if m else '')")
HF_KEY=$(python3 -c "import re,os; t=open('.env').read() if os.path.exists('.env') else ''; m=re.search(r'^HF_TOKEN\s*=\s*(\S+)', t, re.M); print(m.group(1) if m else '')")
[ -z "$WANDB_KEY" ] && { echo "ERROR: no W&B key (run: .venv/bin/wandb login)"; exit 1; }

# -- launch -------------------------------------------------------------------
echo ">>> launching $GPU ($CLOUD), stop-after $STOP_AT"
POD_JSON=$(runpodctl pod create --name "qwench-run" --image "$IMAGE" \
    --gpu-id "$GPU" --gpu-count 1 --container-disk-in-gb 60 \
    --cloud-type "$CLOUD" --ports "22/tcp" --stop-after "$STOP_AT" -o json)
POD_ID=$(echo "$POD_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo ">>> pod $POD_ID  (https://www.runpod.io/console/pods/$POD_ID)"

# -- wait for SSH -------------------------------------------------------------
echo -n ">>> waiting for SSH"
for _ in $(seq 1 40); do
    INFO=$(runpodctl pod get "$POD_ID" -o json 2>/dev/null || echo '{}')
    IP=$(echo "$INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); s=d.get('ssh') or {}; print(s.get('ip',''))" 2>/dev/null)
    PORT=$(echo "$INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); s=d.get('ssh') or {}; print(s.get('port',''))" 2>/dev/null)
    if [ -n "$IP" ] && ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 -i "$KEY" -p "$PORT" root@"$IP" true 2>/dev/null; then
        echo " ready ($IP:$PORT)"; break
    fi
    echo -n "."; sleep 10
done
SSH="ssh -o StrictHostKeyChecking=no -i $KEY -p $PORT root@$IP"

# -- upload repo --------------------------------------------------------------
echo ">>> uploading repo"
tar czf /tmp/qwench.tar.gz --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
    --exclude='deploy' --exclude='*.pyc' -C "$(pwd)" .
scp -o StrictHostKeyChecking=no -i "$KEY" -P "$PORT" /tmp/qwench.tar.gz root@"$IP":/workspace/

# -- setup + run (foreground so the EXIT trap terminates the pod when done) ----
echo ">>> setup + train"
$SSH "
set -e
cd /workspace && mkdir -p qwench && tar xzf qwench.tar.gz -C qwench
ln -sf /workspace/qwench/data /root/data
pip install --quiet 'transformers>=4.51' 'trl>=1.0' peft accelerate datasets wandb bitsandbytes modal 2>&1 | tail -1
cd /workspace/qwench
export PYTHONPATH=/workspace/qwench WANDB_API_KEY='$WANDB_KEY' HF_TOKEN='$HF_KEY' TOKENIZERS_PARALLELISM=false
# tee to /workspace/run.log so cleanup() can save it back before teardown
$CMD 2>&1 | tee /workspace/run.log
echo TRAINING_COMPLETE | tee -a /workspace/run.log
"
echo ">>> training finished — results saved to $SAVE_DIR/, pod terminated by the EXIT trap"
