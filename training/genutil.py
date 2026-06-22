"""Torch-free generation helpers (kept dependency-light so they're unit-testable)."""

from __future__ import annotations


def trim_to_first_eos(token_ids: list[int], eos_id: int) -> list[int]:
    """Continuation up to and including the first EOS; drops trailing padding.

    Batched generation pads finished sequences with pad_token (== eos_token for Qwen), so a
    row looks like ``[...real..., EOS, pad, pad]``. We keep through the first EOS (the real end
    of the plan) and discard the rest. No EOS (hit max_new_tokens) -> the whole row unchanged.
    """
    out: list[int] = []
    for t in token_ids:
        out.append(t)
        if t == eos_id:
            break
    return out
