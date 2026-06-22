"""Grade a model-produced plan against an example.

A plan succeeds iff it (a) parses as JSON, (b) passes structural schema validation,
(c) executes without a precondition error, and (d) leaves the world in a state that
satisfies the example's goal. Returns a verdict dict with the failure stage so the
gate eval can report *why* plans fail, not just how often.

Same grader for the gate (Phase 2), SFT/SDFT eval (Phase 3-5). At the GPU pass the
symbolic `execute_plan` is replaced by the ManiSkill executor; this function's
signature is unchanged.
"""

from __future__ import annotations

from typing import Any

from .skills import PlanInvalid, parse_and_validate
from .world import SkillError, execute_plan


def grade(example: dict[str, Any], model_output: str) -> dict[str, Any]:
    goal = [tuple(p) for p in example["goal"]]
    try:
        steps = parse_and_validate(model_output)
    except PlanInvalid as e:
        return {"success": False, "stage": "parse_or_schema", "reason": str(e)}
    try:
        world = execute_plan(example["scene_state"], steps)
    except SkillError as e:
        return {"success": False, "stage": "execution", "reason": str(e)}
    if not world.satisfies(goal):
        return {"success": False, "stage": "goal", "reason": f"goal {goal} not satisfied"}
    return {"success": True, "stage": "ok", "reason": ""}
