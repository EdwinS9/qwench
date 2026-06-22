"""General-capability eval (MMLU) — the forgetting probe.

Scores a model on multiple-choice MMLU by comparing the next-token log-probabilities of
the option letters (A/B/C/D) after an "Answer:" prompt — the standard log-likelihood MMLU
protocol, no generation needed. Used to measure how much fine-tuning on planning degrades
the base model's general knowledge (the axis where SDFT should beat SFT).

Runs inside the Modal image (needs torch); imported by training/forgetting.py.
"""

from __future__ import annotations

import torch

LETTERS = ["A", "B", "C", "D"]


def format_question(item: dict) -> str:
    lines = [
        "Answer the following multiple-choice question with the letter of the correct option.",
        item["question"],
    ]
    lines += [f"{L}. {c}" for L, c in zip(LETTERS, item["choices"], strict=True)]
    lines.append("Answer:")
    return "\n".join(lines)


@torch.no_grad()
def mmlu_accuracy(model, tok, items, batch_size: int = 16) -> float:
    """Fraction of MMLU items where the highest-logprob option letter is correct."""
    device = next(model.parameters()).device
    # token id for each option letter as it appears after "Answer:" (leading space)
    letter_ids = torch.tensor(
        [tok(f" {L}", add_special_tokens=False).input_ids[-1] for L in LETTERS],
        device=device,
    )
    prev_side = tok.padding_side
    tok.padding_side = "left"  # so the answer slot is at position -1 for every row
    correct = 0
    try:
        was_training = model.training
        model.eval()
        for k in range(0, len(items), batch_size):
            chunk = items[k:k + batch_size]
            prompts = [format_question(it) for it in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(device)
            last = model(**enc).logits[:, -1, :]            # [B, vocab]
            pred = last[:, letter_ids].argmax(dim=-1)        # [B] -> index into LETTERS
            answers = torch.tensor([it["answer"] for it in chunk], device=device)
            correct += int((pred == answers).sum().item())
        if was_training:
            model.train()
    finally:
        tok.padding_side = prev_side
    return correct / max(len(items), 1)
