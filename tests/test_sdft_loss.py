"""Verify analytic_token_kl matches a reference KL computation. Skips without torch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch
    from torch.distributions import Categorical
except ModuleNotFoundError:
    print("skip (no torch) — run on the Modal image")
    sys.exit(0)

from qwench.sdft_loss import analytic_token_kl


def test_matches_reference_and_masks():
    torch.manual_seed(0)
    B, T, V = 2, 5, 17
    s = torch.randn(B, T, V, requires_grad=True)
    t = torch.randn(B, T, V)
    mask = torch.ones(B, T)
    mask[0, 3:] = 0  # mask out some tokens

    got = analytic_token_kl(s, t, mask)

    ref_per_token = Categorical(logits=s).probs * (
        torch.log_softmax(s, -1) - torch.log_softmax(t, -1)
    )
    ref_per_token = ref_per_token.sum(-1)
    ref = (ref_per_token * mask).sum() / mask.sum()

    assert torch.allclose(got, ref, atol=1e-5), (got.item(), ref.item())
    got.backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    print("ok")


if __name__ == "__main__":
    test_matches_reference_and_masks()
