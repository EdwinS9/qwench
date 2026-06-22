"""Task families: scene-instance generators + goal specifications.

Each family mirrors a ManiSkill task type. A generator yields `Instance` objects
holding the initial scene_state (schemas/scene_state.json shape) and a goal
(list of predicates checked by World.satisfies). The solver in solver.py turns
each Instance into a verified gold plan.

When the real ManiSkill executor is wired in (Phase 1 final pass), these generators
are replaced by reading actual ManiSkill task instances; the goal/predicate format
stays the same.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from .world import ARTICULATED, GRASPABLE, LOCATION, RECEPTACLE, SURFACE, is_reachable

# Vocabulary the templated scenes draw from.
GRASPABLES = ["red_cube", "blue_cube", "green_cube", "yellow_block", "apple", "mug", "can"]
RECEPTACLES_IN = ["bin", "box", "basket", "drawer"]   # take relation "in"
RECEPTACLES_ON = ["plate", "tray"]                     # take relation "on"
SURFACES = ["counter", "table", "shelf"]
ARTICULATEDS = ["drawer", "cabinet", "microwave"]
LOCATIONS = ["counter", "table", "sink", "shelf", "floor"]


@dataclass
class Instance:
    family: str
    scene_state: dict[str, Any]
    goal: list[tuple]
    # human-readable slots the instruction templates fill in
    slots: dict[str, str]


def _surface_obj(sid: str) -> dict[str, Any]:
    return {"id": sid, "type": SURFACE}


def _base_robot(base_at: str) -> dict[str, Any]:
    return {"base_at": base_at, "gripper": {"holding": None}}


def _finalize(scene: dict[str, Any]) -> dict[str, Any]:
    """Fill informational `reachable` flags relative to the initial base pose."""
    base_at = scene["robot"]["base_at"]
    for o in scene["objects"]:
        o["reachable"] = is_reachable(base_at, o)
    return scene


# --- PickAndPlace ----------------------------------------------------------
def gen_pick_and_place(rng: random.Random) -> Instance:
    g = rng.choice(GRASPABLES)
    src = rng.choice(SURFACES)
    in_recep = rng.random() < 0.5
    recep = rng.choice(RECEPTACLES_IN if in_recep else RECEPTACLES_ON)
    recep_loc = rng.choice([x for x in LOCATIONS if x != src]) if recep != "drawer" else "counter"
    relation = "in" if in_recep else "on"
    start = rng.choice(LOCATIONS)

    objects = [
        {"id": g, "type": GRASPABLE, "at": src},
        # A drawer may start open or closed; the solver opens it when needed.
        {"id": recep, "type": ARTICULATED if recep == "drawer" else RECEPTACLE,
         "at": recep_loc,
         **({"articulation": rng.choice(["open", "closed"])} if recep == "drawer" else {})},
    ]
    for s in {src, recep_loc} & set(SURFACES):
        objects.append(_surface_obj(s))
    scene = _finalize({"objects": objects, "robot": _base_robot(start)})
    # 4-tuple goal asserts the on/in relation, not just the location.
    return Instance("pick_and_place", scene, [("at", g, recep, relation)],
                    {"object": g, "receptacle": recep, "relation": relation})


# --- Open / Close articulated ---------------------------------------------
def gen_open_close(rng: random.Random) -> Instance:
    a = rng.choice(ARTICULATEDS)
    loc = rng.choice(LOCATIONS)
    start = rng.choice(LOCATIONS)
    open_it = rng.random() < 0.5
    init_state = "closed" if open_it else "open"
    goal_state = "open" if open_it else "closed"
    verb = "open" if open_it else "close"  # the skill name, distinct from the goal state
    objects = [{"id": a, "type": ARTICULATED, "at": loc, "articulation": init_state}]
    if loc in SURFACES:
        objects.append(_surface_obj(loc))
    scene = _finalize({"objects": objects, "robot": _base_robot(start)})
    return Instance("open_close", scene, [("articulation", a, goal_state)],
                    {"object": a, "verb": verb})


# --- Push ------------------------------------------------------------------
def gen_push(rng: random.Random) -> Instance:
    g = rng.choice(GRASPABLES)
    src = rng.choice(SURFACES)
    dst = rng.choice([x for x in LOCATIONS if x != src])
    start = rng.choice(LOCATIONS)
    objects = [{"id": g, "type": GRASPABLE, "at": src}]
    for s in {src, dst} & set(SURFACES):
        objects.append(_surface_obj(s))
    if dst not in SURFACES:
        objects.append({"id": dst, "type": LOCATION})
    scene = _finalize({"objects": objects, "robot": _base_robot(start)})
    return Instance("push", scene, [("at", g, dst)], {"object": g, "target": dst})


# --- Stack / multi-object --------------------------------------------------
def gen_stack(rng: random.Random) -> Instance:
    n = rng.choice([2, 3])
    cubes = rng.sample([c for c in GRASPABLES if c.endswith("cube") or c.endswith("block")], n)
    surface = rng.choice(SURFACES)
    start = rng.choice(LOCATIONS)
    objects = [{"id": c, "type": GRASPABLE, "at": surface} for c in cubes]
    objects.append(_surface_obj(surface))
    # cubes[-1] is the base; stack upward: cubes[i] onto cubes[i+1].
    goal = [("at", cubes[i], cubes[i + 1]) for i in range(n - 1)]
    scene = _finalize({"objects": objects, "robot": _base_robot(start)})
    return Instance("stack", scene, goal, {"cubes": ", ".join(cubes), "order": cubes})


FAMILIES: dict[str, Callable[[random.Random], Instance]] = {
    "pick_and_place": gen_pick_and_place,
    "open_close": gen_open_close,
    "push": gen_push,
    "stack": gen_stack,
}


def generate_instances(family: str, n: int, seed: int) -> Iterator[Instance]:
    rng = random.Random(seed)
    gen = FAMILIES[family]
    for _ in range(n):
        yield gen(rng)
