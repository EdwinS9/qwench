"""Symbolic world model and executor.

This mirrors the semantics of the robot skill API (schemas/skills.json) so we can
*construct and verify* skill plans locally, without a GPU or a full physics sim.

It is deliberately the same interface a ManiSkill-backed executor will expose:

    world = World(scene_state)
    world.apply(skill_name, args)   # raises SkillError if a precondition fails
    world.satisfies(goal)           # True once the goal predicates hold

Phase 1's final pass swaps this symbolic executor for a ManiSkill one (which reads
ground-truth poses and runs the low-level controllers). The plans and goal checks
are identical; only the executor changes.
"""

from __future__ import annotations

import copy
from typing import Any


class SkillError(Exception):
    """Raised when a skill is called with an unsatisfied precondition."""


# Object categories (mirror schemas/scene_state.json `type` enum).
GRASPABLE = "graspable"
RECEPTACLE = "receptacle"
ARTICULATED = "articulated"
SURFACE = "surface"
LOCATION = "location"


def is_reachable(base_at: str, obj: dict[str, Any]) -> bool:
    """An object is reachable iff the base is at it, or at the thing it rests on.

    Single source of truth for the reachability rule, shared by the executor and the
    data generator so generated `reachable` flags can never disagree with the executor.
    """
    return base_at == obj["id"] or base_at == obj.get("at")


class World:
    """A mutable symbolic scene the skill executor operates on."""

    def __init__(self, scene_state: dict[str, Any]):
        self.state = copy.deepcopy(scene_state)
        self._by_id = {o["id"]: o for o in self.state["objects"]}

    # --- queries -----------------------------------------------------------
    def obj(self, oid: str) -> dict[str, Any]:
        if oid not in self._by_id:
            raise SkillError(f"unknown object '{oid}'")
        return self._by_id[oid]

    @property
    def base_at(self) -> str:
        return self.state["robot"]["base_at"]

    @property
    def holding(self) -> str | None:
        return self.state["robot"]["gripper"]["holding"]

    def _reachable(self, oid: str) -> bool:
        return is_reachable(self.base_at, self.obj(oid))

    def _navigable(self) -> set[str]:
        """Valid base targets, derived from the scene: any object id, any location an
        object rests on/in, and the current base pose (so re-navigation is always legal)."""
        targets = {o["id"] for o in self.state["objects"]}
        targets |= {o["at"] for o in self.state["objects"] if o.get("at")}
        targets.add(self.base_at)
        return targets

    def refresh_reachability(self) -> None:
        """Recompute the informational `reachable` flags from the current base pose."""
        for o in self.state["objects"]:
            o["reachable"] = self._reachable(o["id"])

    # --- skills (one method per entry in schemas/skills.json) --------------
    def navigate_to(self, target: str) -> None:
        # Target may be an object id or a location some object rests on; reject the rest
        # so plans that navigate to phantom places cannot reach a goal.
        if target not in self._navigable():
            raise SkillError(f"cannot navigate to unknown target '{target}'")
        self.state["robot"]["base_at"] = target
        self.refresh_reachability()

    def pick(self, object: str) -> None:
        o = self.obj(object)
        if self.holding is not None:
            raise SkillError(f"cannot pick '{object}': gripper holds '{self.holding}'")
        if o["type"] != GRASPABLE:
            raise SkillError(f"cannot pick non-graspable '{object}' ({o['type']})")
        if not self._reachable(object):
            raise SkillError(f"cannot pick '{object}': not reachable from '{self.base_at}'")
        o["at"] = None
        o["relation"] = None  # drop any stale resting relation
        self.state["robot"]["gripper"]["holding"] = object

    def place(self, target: str, relation: str = "on") -> None:
        if self.holding is None:
            raise SkillError("cannot place: gripper is empty")
        tgt = self.obj(target)  # raises on unknown target
        if tgt["type"] == ARTICULATED and tgt.get("articulation") != "open":
            raise SkillError(
                f"cannot place into '{target}': it is "
                f"{tgt.get('articulation', 'closed')}, not open"
            )
        if self.base_at != target:
            raise SkillError(f"cannot place on '{target}': base is at '{self.base_at}'")
        held = self.obj(self.holding)
        held["at"] = target
        held["relation"] = relation  # 'on' | 'in' — checked by satisfies() 4-tuple goals
        self.state["robot"]["gripper"]["holding"] = None

    def open(self, object: str) -> None:
        o = self.obj(object)
        if o["type"] != ARTICULATED:
            raise SkillError(f"cannot open non-articulated '{object}'")
        if not self._reachable(object):
            raise SkillError(f"cannot open '{object}': not reachable")
        o["articulation"] = "open"

    def close(self, object: str) -> None:
        o = self.obj(object)
        if o["type"] != ARTICULATED:
            raise SkillError(f"cannot close non-articulated '{object}'")
        if not self._reachable(object):
            raise SkillError(f"cannot close '{object}': not reachable")
        o["articulation"] = "closed"

    def push(self, object: str, target: str) -> None:
        o = self.obj(object)
        if self.holding is not None:
            raise SkillError("cannot push while holding an object")
        if not self._reachable(object):
            raise SkillError(f"cannot push '{object}': not reachable")
        self.obj(target)  # raises on unknown destination
        o["at"] = target

    def detect(self, object: str) -> None:
        self.obj(object)  # perception only — validates existence, no effect

    def done(self) -> None:
        pass  # terminal marker, no effect

    # The skill names are exactly the executor method names above; argument validation
    # lives in skills.validate_plan, so apply() only needs to dispatch.
    _SKILLS = frozenset(
        {"navigate_to", "pick", "place", "open", "close", "push", "detect", "done"}
    )

    def apply(self, skill: str, args: dict[str, Any]) -> None:
        if skill not in self._SKILLS:
            raise SkillError(f"unknown skill '{skill}'")
        getattr(self, skill)(**args)

    # --- goals -------------------------------------------------------------
    def satisfies(self, goal: list[tuple]) -> bool:
        """Goal is a list of predicates; all must hold.

        Predicate forms:
            ("at", object_id, target_id)              -> object rests on/in target
            ("at", object_id, target_id, relation)    -> also require the 'on'|'in' relation
            ("articulation", object_id, "open"|"closed")
        """
        for pred in goal:
            kind = pred[0]
            if kind == "at":
                oid, target = pred[1], pred[2]
                o = self.obj(oid)
                if o.get("at") != target:
                    return False
                if len(pred) == 4 and o.get("relation") != pred[3]:
                    return False
            elif kind == "articulation":
                _, oid, want = pred
                if self.obj(oid).get("articulation") != want:
                    return False
            else:
                raise SkillError(f"unknown goal predicate '{kind}'")
        return True


def execute_plan(scene_state: dict[str, Any], plan: list[dict[str, Any]]) -> World:
    """Run a full plan from an initial scene; returns the resulting World.

    Raises SkillError on the first failing step. Used both during data generation
    (to verify a constructed plan reaches its goal) and during evaluation (to grade
    a model-produced plan).
    """
    world = World(scene_state)
    for step in plan:
        world.apply(step["skill"], step.get("args", {}))
    return world
