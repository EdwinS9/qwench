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
    return [json.loads(line) for line in path.read_text().splitlines()]


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


def _ex(goal):
    """Minimal gradable example: a cube on the counter, robot at the floor."""
    return {
        "goal": goal,
        "scene_state": {
            "objects": [{"id": "cube", "type": "graspable", "at": "counter"},
                        {"id": "counter", "type": "surface"}],
            "robot": {"base_at": "floor", "gripper": {"holding": None}},
        },
    }


def _plan(*steps):
    return json.dumps({"thinking": "", "plan": list(steps) + [{"skill": "done", "args": {}}]})


def test_grade_failure_stages():
    # not JSON -> parse/schema stage
    assert grade(_ex([("at", "cube", "counter")]), "this is not json")["stage"] == "parse_or_schema"
    # schema-valid but precondition violated (pick when base not at cube) -> execution stage
    pick_plan = _plan({"skill": "pick", "args": {"object": "cube"}})
    v = grade(_ex([("at", "cube", "counter")]), pick_plan)
    assert v["stage"] == "execution", v
    # parses + executes cleanly but goal unmet -> goal stage
    v = grade(_ex([("at", "cube", "bin")]), _plan({"skill": "detect", "args": {"object": "cube"}}))
    assert v["stage"] == "goal" and not v["success"], v


if __name__ == "__main__":
    test_gold_plans_grade_success()
    test_prompts_build()
    test_grade_failure_stages()
    print("ok")
