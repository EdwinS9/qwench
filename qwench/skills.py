"""Load the skill API and structurally validate plans against the schemas.

Lightweight, dependency-free validation (no jsonschema needed) covering exactly the
constraints in schemas/plan.json + schemas/skills.json:
  - JSON parses into the plan object shape
  - every step.skill is a known skill
  - args contain the required params and no unknown params
  - the final step is `done`
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"


def load_skills() -> dict[str, dict[str, Any]]:
    """Return {skill_name: {required: [...], optional: [...], enums: {...}}}."""
    spec = json.loads((_SCHEMA_DIR / "skills.json").read_text())
    out: dict[str, dict[str, Any]] = {}
    for skill in spec["skills"]:
        params = skill["parameters"]
        props = params.get("properties", {})
        required = list(params.get("required", []))
        optional = [p for p in props if p not in required]
        enums = {p: v["enum"] for p, v in props.items() if "enum" in v}
        out[skill["name"]] = {"required": required, "optional": optional, "enums": enums}
    return out


SKILLS = load_skills()


class PlanInvalid(Exception):
    """Raised when a plan violates the structural schema."""


def validate_plan(plan_obj: Any) -> list[dict[str, Any]]:
    """Validate a parsed plan object; return its `plan` step list or raise PlanInvalid."""
    if not isinstance(plan_obj, dict) or "plan" not in plan_obj:
        raise PlanInvalid("missing top-level 'plan' array")
    steps = plan_obj["plan"]
    if not isinstance(steps, list) or not steps:
        raise PlanInvalid("'plan' must be a non-empty array")

    for i, step in enumerate(steps):
        if not isinstance(step, dict) or "skill" not in step:
            raise PlanInvalid(f"step {i}: missing 'skill'")
        name = step["skill"]
        if name not in SKILLS:
            raise PlanInvalid(f"step {i}: unknown skill '{name}'")
        if "args" not in step:  # plan.json marks 'args' required, even for `done`
            raise PlanInvalid(f"step {i} ({name}): missing 'args'")
        args = step["args"]
        if not isinstance(args, dict):
            raise PlanInvalid(f"step {i}: 'args' must be an object")
        spec = SKILLS[name]
        for req in spec["required"]:
            if req not in args:
                raise PlanInvalid(f"step {i} ({name}): missing required arg '{req}'")
        allowed = set(spec["required"]) | set(spec["optional"])
        for key in args:
            if key not in allowed:
                raise PlanInvalid(f"step {i} ({name}): unknown arg '{key}'")
            if key in spec["enums"] and args[key] not in spec["enums"][key]:
                raise PlanInvalid(
                    f"step {i} ({name}): arg '{key}'='{args[key]}' not in {spec['enums'][key]}"
                )

    if steps[-1]["skill"] != "done":
        raise PlanInvalid("plan must end with a 'done' step")
    return steps


def parse_and_validate(text: str) -> list[dict[str, Any]]:
    """Parse model output text as JSON and validate it. Raises PlanInvalid on failure."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise PlanInvalid(f"not valid JSON: {e}") from e
    return validate_plan(obj)
