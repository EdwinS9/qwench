"""Sanity tests: every constructed plan validates and reaches its goal."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwench.skills import PlanInvalid, validate_plan
from qwench.solver import solve
from qwench.tasks import FAMILIES, generate_instances
from qwench.world import execute_plan


def test_all_families_solve_and_verify():
    for fi, family in enumerate(FAMILIES):
        for inst in generate_instances(family, 100, seed=fi):
            plan = solve(inst)
            validate_plan({"plan": plan})  # raises if structurally bad
            world = execute_plan(inst.scene_state, plan)
            assert world.satisfies(inst.goal), f"{family} plan missed goal {inst.goal}"


def test_validator_rejects_bad_plans():
    for bad in [
        {"plan": []},                                            # empty
        {"plan": [{"skill": "fly", "args": {}}]},                # unknown skill
        {"plan": [{"skill": "pick", "args": {}}]},               # missing required arg
        {"plan": [{"skill": "pick", "args": {"object": "x"}}]},  # no terminal done
        {"plan": [{"skill": "place", "args": {"target": "t", "relation": "under"}},
                  {"skill": "done", "args": {}}]},               # bad enum
    ]:
        try:
            validate_plan(bad)
        except PlanInvalid:
            continue
        raise AssertionError(f"validator accepted bad plan: {bad}")


if __name__ == "__main__":
    test_all_families_solve_and_verify()
    test_validator_rejects_bad_plans()
    print("ok")
