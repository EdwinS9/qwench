"""Phase 1 data generation.

Pipeline per task family:
    1. generate scene instances (tasks.py)
    2. construct the gold plan (solver.py)
    3. structurally validate the plan against the schemas (skills.py)
    4. EXECUTE the plan in the symbolic world and confirm it reaches the goal (world.py)
       -- only verified examples are kept; a failure is logged and dropped.
    5. attach a templated instruction + thinking rationale (instructions.py)

Output: JSONL records, split into train / heldout. The heldout split feeds the
Phase 2 teacher-beats-student gate and the Phase 5 comparison.

    python -m qwench.generate --per-family 200 --out data/

When the ManiSkill executor lands, step 4 calls it instead of the symbolic World;
everything else is unchanged.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .instructions import make_instruction, make_thinking
from .skills import PlanInvalid, validate_plan
from .solver import solve
from .tasks import FAMILIES, generate_instances
from .world import SkillError, execute_plan


def build_record(inst, rng) -> dict | None:
    """Return a verified training record, or None if the plan fails validation/execution."""
    plan = solve(inst)
    try:
        validate_plan({"plan": plan})
    except PlanInvalid as e:
        print(f"  [drop] {inst.family}: invalid plan ({e})")
        return None
    try:
        world = execute_plan(inst.scene_state, plan)
    except SkillError as e:
        print(f"  [drop] {inst.family}: execution failed ({e})")
        return None
    if not world.satisfies(inst.goal):
        print(f"  [drop] {inst.family}: plan did not reach goal {inst.goal}")
        return None
    return {
        "task_family": inst.family,
        "instruction": make_instruction(inst, rng),
        "scene_state": inst.scene_state,
        "target": {"thinking": make_thinking(inst, plan), "plan": plan},
        "goal": [list(p) for p in inst.goal],  # kept for sim grading at eval time
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate SDFT skill-planning data.")
    ap.add_argument("--per-family", type=int, default=200)
    ap.add_argument("--heldout-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    train, heldout = [], []
    kept = dropped = 0

    for fi, family in enumerate(FAMILIES):
        print(f"[{family}] generating {args.per_family} instances...")
        for inst in generate_instances(family, args.per_family, seed=args.seed + fi):
            rec = build_record(inst, rng)
            if rec is None:
                dropped += 1
                continue
            kept += 1
            (heldout if rng.random() < args.heldout_frac else train).append(rec)

    for name, rows in [("train", train), ("heldout", heldout)]:
        path = args.out / f"{name}.jsonl"
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(rows):5d} -> {path}")
    print(f"kept {kept}, dropped {dropped}")
    if dropped:
        raise SystemExit(f"ERROR: {dropped} examples failed verification — fix solver/world.")


if __name__ == "__main__":
    main()
