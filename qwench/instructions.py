"""Templated natural-language instructions + a teacher-style `thinking` rationale.

Pure templates (no API). The paper's teacher prompt asks for "the thinking process",
so each gold target carries a short rationale derived from the plan. If Phase 2 shows
the model over-fits template phrasing, add a one-shot LLM paraphrase pass over the
`instruction` field only (the plan/thinking stay machine-generated).
"""

from __future__ import annotations

import random

from .tasks import Instance


def _readable(oid: str) -> str:
    return oid.replace("_", " ")


_TEMPLATES = {
    "pick_and_place": [
        "put the {object} {relation} the {receptacle}",
        "place the {object} {relation} the {receptacle}",
        "move the {object} {prep} the {receptacle}",
        "the {object} should go {relation} the {receptacle}",
    ],
    "open_close": [
        "{verb} the {object}",
        "please {verb} the {object}",
        "I need the {object} {verb_past}",
    ],
    "push": [
        "push the {object} to the {target}",
        "shove the {object} over to the {target}",
        "slide the {object} to the {target}",
    ],
    "stack": [
        "stack the cubes: {cubes}",
        "build a stack with {cubes}",
        "pile up {cubes} on top of each other",
    ],
}

_VERB_PAST = {"open": "opened", "close": "closed"}


def make_instruction(inst: Instance, rng: random.Random) -> str:
    tmpl = rng.choice(_TEMPLATES[inst.family])
    slots = {k: _readable(v) if isinstance(v, str) else v for k, v in inst.slots.items()}
    if inst.family == "pick_and_place":
        # Keep the natural "into/onto" phrasing consistent with the gold relation.
        slots["prep"] = "into" if inst.slots["relation"] == "in" else "onto"
    if inst.family == "open_close":
        slots["verb_past"] = _VERB_PAST[inst.slots["verb"]]
    if inst.family == "stack":
        slots["cubes"] = ", then ".join(_readable(c) for c in reversed(inst.slots["order"]))
    return tmpl.format(**slots)


def make_thinking(inst: Instance, plan: list[dict]) -> str:
    """A short rationale mirroring the plan (the teacher-style 'thinking process')."""
    phrases = []
    for step in plan:
        s, a = step["skill"], step.get("args", {})
        if s == "navigate_to":
            phrases.append(f"go to the {_readable(a['target'])}")
        elif s == "pick":
            phrases.append(f"grasp the {_readable(a['object'])}")
        elif s == "place":
            phrases.append(f"set it {a.get('relation', 'on')} the {_readable(a['target'])}")
        elif s == "open":
            phrases.append(f"open the {_readable(a['object'])}")
        elif s == "close":
            phrases.append(f"close the {_readable(a['object'])}")
        elif s == "push":
            phrases.append(f"push the {_readable(a['object'])} to the {_readable(a['target'])}")
        elif s == "done":
            phrases.append("then the goal is met")
    return "I need to " + ", ".join(phrases) + "."
