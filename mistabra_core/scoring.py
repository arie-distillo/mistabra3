"""Score (spec v3 §5.1) and standing (§5.2).

`score` is the PoC's documented strength formula, unchanged, so it stays comparable with earlier measurements.
Standing has two strategies behind one function: `judged` (the reasoning model weighs the attached trajectories
together — the frame under test) and `noisy_or` (a deterministic baseline that can only accumulate support).
"""
from __future__ import annotations

from .types import StandingUse

STATUS_WEIGHT = {"confirmed": 1.0, "rejected": 0.0}


def score_from(reach: float, fit: float, gap: float, specificity: float, credibility: float) -> float:
    """Deterministic and replayable. gap < 0 means an ordinary explanation fits better than the hypothesis."""
    s = 0.4 * fit + 0.4 * max(gap, 0.0) + 0.2 * reach
    if gap < 0:
        s *= max(0.0, 1.0 + gap)
    s *= 0.85 + 0.15 * specificity
    s *= 0.7 + 0.3 * credibility
    return round(min(1.0, max(0.0, s)), 3)


def score_margin(reach: float, gap: float, specificity: float, credibility: float) -> float:
    """Option B for a hypothesis articulated from this very episode: such a claim is tailored to the episode, so its fit
    is high by construction; score by how much better it explains the episode than the ordinary explanations (gap) and
    by reach only. Same penalty and adjustments as score_from; weights rescaled so the maximum stays 1."""
    s = (2 / 3) * max(gap, 0.0) + (1 / 3) * reach
    if gap < 0:
        s *= max(0.0, 1.0 + gap)
    s *= 0.85 + 0.15 * specificity
    s *= 0.7 + 0.3 * credibility
    return round(min(1.0, max(0.0, s)), 3)


def weight_of(status: str, proposed_weight: float) -> float:
    return STATUS_WEIGHT.get(status, proposed_weight)


def noisy_or(items: list[dict]) -> float:
    """Supporting trajectories accumulate; each weakening one removes up to half of what was accumulated."""
    p, damp = 1.0, 1.0
    for t in items:
        if t.get("stance") == "neutral":
            continue
        if t.get("stance") == "weakens":
            damp *= 1.0 - 0.5 * t["weight"]
        else:
            p *= 1.0 - max(0.0, min(1.0, t["weight"] * t["score"]))
    return round((1.0 - p) * damp, 3)


def uses(items: list[dict]) -> tuple[StandingUse, ...]:
    return tuple(StandingUse(trajectory_id=t["id"], status=t["status"], weight=t["weight"]) for t in items)
