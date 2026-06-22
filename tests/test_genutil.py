"""Unit tests for the torch-free generation helpers used by batched generation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.genutil import trim_to_first_eos

EOS = 7


def test_trim_to_first_eos():
    # keeps through the first EOS, drops trailing padding (pad == eos)
    assert trim_to_first_eos([1, 2, 3, EOS, EOS, EOS], EOS) == [1, 2, 3, EOS]
    # no EOS (hit max_new_tokens) -> whole row unchanged
    assert trim_to_first_eos([1, 2, 3], EOS) == [1, 2, 3]
    # immediate EOS -> empty-ish continuation of just the EOS
    assert trim_to_first_eos([EOS, EOS], EOS) == [EOS]
    # empty input
    assert trim_to_first_eos([], EOS) == []
    # only the first EOS terminates, even with later real tokens (shouldn't happen, but defined)
    assert trim_to_first_eos([1, EOS, 9, 9], EOS) == [1, EOS]


if __name__ == "__main__":
    test_trim_to_first_eos()
    print("ok")
