"""Direct unit tests for the symbolic World executor.

The World is the ground truth the whole evaluation rests on, so its precondition
guards are tested explicitly here (not just indirectly via gold plans reaching goals).
Plain-assert style, runnable with `python tests/test_world.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwench.world import SkillError, World


def _scene(objects, base_at, holding=None):
    return {"objects": objects, "robot": {"base_at": base_at, "gripper": {"holding": holding}}}


def assert_raises(fn, *, msg):
    try:
        fn()
    except SkillError:
        return
    raise AssertionError(f"expected SkillError: {msg}")


CUBE = {"id": "cube", "type": "graspable", "at": "counter"}
TABLE = {"id": "table", "type": "surface"}
DRAWER_CLOSED = {"id": "drawer", "type": "articulated", "at": "counter", "articulation": "closed"}


def test_pick_guards():
    # holding already
    w = World(_scene([dict(CUBE), TABLE], base_at="cube", holding="something"))
    assert_raises(lambda: w.pick("cube"), msg="pick while holding")
    # non-graspable
    w = World(_scene([dict(CUBE), TABLE], base_at="table"))
    assert_raises(lambda: w.pick("table"), msg="pick non-graspable")
    # unreachable
    w = World(_scene([dict(CUBE)], base_at="floor"))
    assert_raises(lambda: w.pick("cube"), msg="pick unreachable")
    # unknown object
    w = World(_scene([dict(CUBE)], base_at="cube"))
    assert_raises(lambda: w.pick("ghost"), msg="pick unknown object")


def test_place_guards():
    # empty gripper
    w = World(_scene([dict(CUBE), TABLE], base_at="table"))
    assert_raises(lambda: w.place("table"), msg="place with empty gripper")
    # base not at target
    w = World(_scene([dict(CUBE), TABLE], base_at="floor", holding="cube"))
    assert_raises(lambda: w.place("table"), msg="place when base not at target")
    # into a closed drawer
    w = World(_scene([dict(CUBE), dict(DRAWER_CLOSED)], base_at="drawer", holding="cube"))
    assert_raises(lambda: w.place("drawer", "in"), msg="place into closed drawer")
    # nonexistent target
    w = World(_scene([dict(CUBE)], base_at="cube", holding="cube"))
    assert_raises(lambda: w.place("nowhere"), msg="place onto nonexistent target")


def test_open_close_guards():
    w = World(_scene([dict(CUBE)], base_at="cube"))
    assert_raises(lambda: w.open("cube"), msg="open non-articulated")
    w = World(_scene([dict(DRAWER_CLOSED)], base_at="floor"))
    assert_raises(lambda: w.open("drawer"), msg="open unreachable")
    w = World(_scene([dict(CUBE)], base_at="cube"))
    assert_raises(lambda: w.close("cube"), msg="close non-articulated")


def test_push_guards():
    w = World(_scene([dict(CUBE), TABLE], base_at="cube", holding="cube"))
    assert_raises(lambda: w.push("cube", "table"), msg="push while holding")
    w = World(_scene([dict(CUBE), TABLE], base_at="floor"))
    assert_raises(lambda: w.push("cube", "table"), msg="push unreachable")
    w = World(_scene([dict(CUBE)], base_at="cube"))
    assert_raises(lambda: w.push("cube", "nowhere"), msg="push to nonexistent destination")


def test_navigate_and_dispatch_guards():
    w = World(_scene([dict(CUBE)], base_at="counter"))
    assert_raises(lambda: w.navigate_to("atlantis"), msg="navigate to unknown target")
    # navigating to a location an object rests on is allowed
    w.navigate_to("counter")  # cube.at == counter
    assert w.base_at == "counter"
    assert_raises(lambda: w.apply("fly", {}), msg="unknown skill via apply")


def test_satisfies():
    # build an end-state by executing a small plan
    w = World(_scene([dict(CUBE), TABLE], base_at="cube"))
    w.pick("cube")
    w.navigate_to("table")
    w.place("table", "on")
    assert w.satisfies([("at", "cube", "table")]) is True
    assert w.satisfies([("at", "cube", "table", "on")]) is True
    assert w.satisfies([("at", "cube", "table", "in")]) is False   # relation matters
    assert w.satisfies([("at", "cube", "shelf")]) is False
    # articulation predicate
    w2 = World(_scene([dict(DRAWER_CLOSED)], base_at="drawer"))
    assert w2.satisfies([("articulation", "drawer", "closed")]) is True
    w2.open("drawer")
    assert w2.satisfies([("articulation", "drawer", "open")]) is True
    # unknown predicate raises
    assert_raises(lambda: w2.satisfies([("color", "drawer", "red")]), msg="unknown predicate")


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ok")


if __name__ == "__main__":
    _run_all()
