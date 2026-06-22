"""Per-family procedural solvers: Instance -> gold plan (list of skill steps).

These encode the known-correct skill sequence for each task family. They do NOT
search or guess — the plan follows directly from the goal. Every produced plan is
verified against the symbolic executor in generate.py before being kept, so a solver
bug surfaces as a dropped example rather than a silently-wrong label.
"""

from __future__ import annotations

from typing import Any

from .tasks import Instance


def _step(skill: str, **args: Any) -> dict[str, Any]:
    return {"skill": skill, "args": args}


def _solve_pick_and_place(inst: Instance) -> list[dict[str, Any]]:
    s = inst.slots
    return [
        _step("navigate_to", target=s["object"]),
        _step("pick", object=s["object"]),
        _step("navigate_to", target=s["receptacle"]),
        _step("place", target=s["receptacle"], relation=s["relation"]),
        _step("done"),
    ]


def _solve_open_close(inst: Instance) -> list[dict[str, Any]]:
    s = inst.slots
    verb = "open" if s["verb"] == "open" else "close"
    return [
        _step("navigate_to", target=s["object"]),
        _step(verb, object=s["object"]),
        _step("done"),
    ]


def _solve_push(inst: Instance) -> list[dict[str, Any]]:
    s = inst.slots
    return [
        _step("navigate_to", target=s["object"]),
        _step("push", object=s["object"], target=s["target"]),
        _step("done"),
    ]


def _solve_stack(inst: Instance) -> list[dict[str, Any]]:
    order = inst.slots["order"]  # top -> bottom; build bottom-up
    steps: list[dict[str, Any]] = []
    for i in range(len(order) - 2, -1, -1):
        upper, lower = order[i], order[i + 1]
        steps += [
            _step("navigate_to", target=upper),
            _step("pick", object=upper),
            _step("navigate_to", target=lower),
            _step("place", target=lower, relation="on"),
        ]
    steps.append(_step("done"))
    return steps


_SOLVERS = {
    "pick_and_place": _solve_pick_and_place,
    "open_close": _solve_open_close,
    "push": _solve_push,
    "stack": _solve_stack,
}


def solve(inst: Instance) -> list[dict[str, Any]]:
    return _SOLVERS[inst.family](inst)
