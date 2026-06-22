"""Plumbing test: feeding each example's own gold plan to the grader must succeed.

This validates grade.py + prompts.py end-to-end WITHOUT a model, so the Modal gate
run only has to worry about the model itself.
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwench.grade import grade
from qwench.prompts import pick_demo, student_messages, teacher_messages


def _load(name):
    path = ROOT / "data" / f"{name}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines()]


def test_gold_plans_grade_success():
    rows = _load("heldout")
    assert rows, "no heldout data — run `python -m qwench.generate` first"
    for r in rows:
        gold = json.dumps(r["target"])
        verdict = grade(r, gold)
        assert verdict["success"], f"gold plan failed grading: {verdict} for {r['instruction']}"


def test_prompts_build():
    train, heldout = _load("train"), _load("heldout")
    rng = random.Random(0)
    ex = heldout[0]
    demo = pick_demo(ex, train, rng)
    sm = student_messages(ex)
    tm = teacher_messages(ex, demo)
    assert sm[-1]["content"] in tm[-1]["content"]  # teacher = student + demo prefix
    assert demo["instruction"] in tm[-1]["content"]
    assert sm[0]["content"] == tm[0]["content"]      # same system prompt


if __name__ == "__main__":
    test_gold_plans_grade_success()
    test_prompts_build()
    print("ok")
