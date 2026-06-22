"""The SDFT objective: analytic per-token reverse KL.

The student π_θ(·|x) is trained to match the teacher π(·|x,c) on the student's own
(on-policy) response tokens, minimizing

    D_KL( π_θ(·|x) ‖ π(·|x,c) )

decomposed per token. The paper uses the *analytic* estimator: at each position,
marginalize over the full vocabulary rather than using a sampled scalar KL — biased
in theory but the most stable in their ablations.

This module is pure torch and CPU-testable; the Modal trainer imports it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def analytic_token_kl(
    student_logits: torch.Tensor,   # [B, T, V] — requires grad
    teacher_logits: torch.Tensor,   # [B, T, V] — detached (EMA teacher)
    mask: torch.Tensor,             # [B, T] — 1 on response tokens, 0 elsewhere
) -> torch.Tensor:
    """Mean over masked tokens of KL(student_t ‖ teacher_t), differentiable in student.

    KL(p‖q) = Σ_v p_v (log p_v − log q_v), with p=softmax(student), q=softmax(teacher).
    """
    log_p = F.log_softmax(student_logits, dim=-1)
    log_q = F.log_softmax(teacher_logits.detach(), dim=-1)
    p = log_p.exp()
    kl_per_token = (p * (log_p - log_q)).sum(dim=-1)  # [B, T]
    denom = mask.sum().clamp_min(1.0)
    return (kl_per_token * mask).sum() / denom
