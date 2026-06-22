"""Phase 2 — the teacher-beats-student go/no-go gate, on Modal.

Runs base Qwen3-8B over the heldout set twice:
  - STUDENT: instruction + scene only
  - TEACHER: same + one in-context demonstration (the SDFT teacher)

Grades both with the same symbolic grader and reports plan-success rates. The gate
PASSES if the teacher is meaningfully better than the student — that gap is exactly
what SDFT distills. If the teacher does NOT clearly win, SDFT has nothing to learn
and we stop before spending on training.

Run:
    modal run eval/gate.py                  # full heldout set
    modal run eval/gate.py --limit 32       # quick smoke
    modal run eval/gate.py --model Qwen/Qwen3-14B

Requires Modal auth (run `modal token new` once) and, for gated weights, an `HF_TOKEN`
secret. Qwen3 is open, so a token is usually optional.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers>=4.51", "torch", "accelerate")
    .add_local_dir(str(REPO / "qwench"), remote_path="/root/qwench")
    .add_local_dir(str(REPO / "schemas"), remote_path="/root/schemas")
)

app = modal.App("qwench-gate", image=image)

# Persist the HF cache between runs so we don't re-download 8B each time.
hf_cache = modal.Volume.from_name("qwench-hf-cache", create_if_missing=True)

PASS_MARGIN = 0.15  # teacher must beat student by at least this absolute success-rate gap


@app.function(gpu="A100-80GB", timeout=60 * 60, volumes={"/root/.cache/huggingface": hf_cache})
def run_gate(model: str, heldout: list[dict], train: list[dict], limit: int | None):
    import random
    import sys

    sys.path.insert(0, "/root")
    from collections import Counter

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from qwench.grade import grade
    from qwench.prompts import pick_demo, render_chat, student_messages, teacher_messages

    if limit:
        heldout = heldout[:limit]
    rng = random.Random(0)
    # Plain transformers generation (vLLM dropped — it lagged Qwen3 support). Same stack
    # as training-eval, so the gate's numbers are directly comparable.
    tok = AutoTokenizer.from_pretrained(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required for correct batched decoder generation
    lm = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda").eval()

    @torch.no_grad()
    def generate(prompts: list[str], batch_size: int = 16) -> list[str]:
        out = []
        for i in range(0, len(prompts), batch_size):
            enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            gen = lm.generate(**enc, max_new_tokens=384, do_sample=False,
                              use_cache=True, pad_token_id=tok.pad_token_id)
            prompt_len = enc.input_ids.shape[1]  # left-padded, so identical across the batch
            out.extend(tok.decode(row[prompt_len:], skip_special_tokens=True) for row in gen)
        return out

    student_out = generate([render_chat(tok, student_messages(ex)) for ex in heldout])
    teacher_out = generate([render_chat(tok, teacher_messages(ex, pick_demo(ex, train, rng)))
                            for ex in heldout])

    def score(outputs):
        succ, stages = 0, Counter()
        for ex, text in zip(heldout, outputs, strict=True):
            v = grade(ex, text)
            succ += int(v["success"])
            stages[v["stage"]] += 1
        return succ / len(heldout), stages

    s_rate, s_stages = score(student_out)
    t_rate, t_stages = score(teacher_out)
    gap = t_rate - s_rate
    passed = gap >= PASS_MARGIN

    return {
        "model": model, "n": len(heldout),
        "student_success": round(s_rate, 4), "teacher_success": round(t_rate, 4),
        "gap": round(gap, 4), "pass_margin": PASS_MARGIN, "gate_passed": passed,
        "student_failure_stages": dict(s_stages), "teacher_failure_stages": dict(t_stages),
    }


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-8B", limit: int = 0):
    data = REPO / "data"
    heldout = [json.loads(line) for line in (data / "heldout.jsonl").read_text().splitlines()]
    train = [json.loads(line) for line in (data / "train.jsonl").read_text().splitlines()]
    result = run_gate.remote(model, heldout, train, limit or None)
    print(json.dumps(result, indent=2))
    print()
    if result["gate_passed"]:
        print(f"PASS: teacher beats student by {result['gap']:+.1%}. "
              "SDFT has a real teacher signal to distill; proceed to training.")
    else:
        print(f"FAIL: teacher only {result['gap']:+.1%} over student "
              f"(need >= {PASS_MARGIN:.0%}). The in-context boost is too weak at this scale; "
              "do NOT start SDFT. Options: larger model, richer demos, or simpler tasks.")
