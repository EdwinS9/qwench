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

Requires Modal auth (profile `build-small-hackathon` is already active) and, for
gated weights, an `HF_TOKEN` secret. Qwen3 is open, so a token is usually optional.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.6.3", "transformers>=4.51", "torch")
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

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    from qwench.grade import grade
    from qwench.prompts import pick_demo, student_messages, teacher_messages

    if limit:
        heldout = heldout[:limit]
    rng = random.Random(0)
    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.9, max_model_len=4096)
    sampling = SamplingParams(temperature=0.0, max_tokens=512)

    def render(messages):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    student_prompts = [render(student_messages(ex)) for ex in heldout]
    teacher_prompts = [render(teacher_messages(ex, pick_demo(ex, train, rng))) for ex in heldout]

    student_out = llm.generate(student_prompts, sampling)
    teacher_out = llm.generate(teacher_prompts, sampling)

    def score(outputs):
        succ, stages = 0, Counter()
        for ex, o in zip(heldout, outputs):
            v = grade(ex, o.outputs[0].text)
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
    heldout = [json.loads(l) for l in (data / "heldout.jsonl").read_text().splitlines()]
    train = [json.loads(l) for l in (data / "train.jsonl").read_text().splitlines()]
    result = run_gate.remote(model, heldout, train, limit or None)
    print(json.dumps(result, indent=2))
    print()
    if result["gate_passed"]:
        print(f"✅ GATE PASSED — teacher beats student by {result['gap']:+.1%}. "
              "SDFT has a real teacher signal to distill; proceed to training.")
    else:
        print(f"❌ GATE FAILED — teacher only {result['gap']:+.1%} over student "
              f"(need ≥{PASS_MARGIN:.0%}). The in-context boost is too weak at this scale; "
              "do NOT start SDFT. Options: larger model, richer demos, or simpler tasks.")
