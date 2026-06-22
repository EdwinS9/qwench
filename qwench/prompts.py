"""Prompt construction for student and teacher, shared by the gate eval and training.

- **Student** sees only the instruction + scene state.
- **Teacher** additionally sees ONE worked demonstration of the same task family,
  following the paper's template ("This is an example response ... Now answer with a
  response of your own, including the thinking process").

Keeping this in one module guarantees the gate eval and the SDFT trainer build
identical prompts — the whole method depends on the teacher/student prompts differing
ONLY by the in-context demonstration.
"""

from __future__ import annotations

import json
import random
from typing import Any

from .skills import load_skills


def _skill_catalog() -> str:
    lines = []
    for name, spec in load_skills().items():
        args = ", ".join(spec["required"] + [f"{o}?" for o in spec["optional"]])
        lines.append(f"- {name}({args})")
    return "\n".join(lines)


SYSTEM = (
    "You are a robot task planner. Given an instruction and the current scene state, "
    "output a plan as a single JSON object with two fields: `thinking` (a brief rationale) "
    "and `plan` (an ordered list of steps, each {\"skill\": name, \"args\": {...}}). "
    "Use only these skills:\n"
    f"{_skill_catalog()}\n"
    "Every plan must end with a `done` step. Output ONLY the JSON object, no prose."
)


def _user_block(instruction: str, scene_state: dict[str, Any]) -> str:
    return (
        f"Instruction: {instruction}\n"
        f"Scene state:\n{json.dumps(scene_state, indent=2)}"
    )


def _demo_block(demo: dict[str, Any]) -> str:
    """Format one worked example for the teacher's in-context slot."""
    return (
        f"{_user_block(demo['instruction'], demo['scene_state'])}\n"
        "This is an example of a correct response to a similar instruction:\n"
        f"{json.dumps(demo['target'])}\n"
        "Now answer with a response of your own, including the thinking process."
    )


def student_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": _user_block(example["instruction"], example["scene_state"])},
    ]


def teacher_messages(example: dict[str, Any], demo: dict[str, Any]) -> list[dict[str, str]]:
    user = _demo_block(demo) + "\n\n" + _user_block(example["instruction"], example["scene_state"])
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]


def pick_demo(
    example: dict[str, Any], pool: list[dict[str, Any]], rng: random.Random
) -> dict[str, Any]:
    """Pick an in-context demonstration: same task family, never `example` itself.

    Returning the example would leak the gold target into the teacher prompt (trivially
    correct teacher); a cross-family demo gives no teacher advantage. We therefore prefer
    same-family, fall back to any *other* example, and fail loudly if neither exists —
    rather than silently degrading the teacher signal the method depends on.
    """
    same = [d for d in pool if d["task_family"] == example["task_family"]
            and d["instruction"] != example["instruction"]]
    if same:
        return rng.choice(same)
    others = [d for d in pool if d is not example and d["instruction"] != example["instruction"]]
    if not others:
        raise ValueError(f"no valid demo for an example in family {example['task_family']!r}")
    return rng.choice(others)
