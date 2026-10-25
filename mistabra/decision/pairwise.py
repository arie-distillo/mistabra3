"""`pairwise`: the PoC's original method behind the DecisionModel interface. No LLM call of its own.

Strength is the documented formula over the abducer's fit, the gap to the best baseline, reach, specificity
and source credibility — unchanged, so stored strengths replay identically. The posterior is derived from the
same numbers (weights = fits, none = 1 - best fit) so that both methods can be compared on one output shape;
it is a translation of pairwise judgments, not an additional judgment.
"""
from __future__ import annotations
from .base import NONE, Abduction, HypothesisSet, Posterior, Scored, State, kl_bits, normalise, not_implemented_action

NAME = "pairwise"


def strength_from(reach: float, fit: float, gap: float, specificity: float, credibility: float) -> float:
    """Deterministic, documented, replayable. gap < 0 means an ordinary explanation fits better."""
    s = 0.4 * fit + 0.4 * max(gap, 0.0) + 0.2 * reach
    if gap < 0:
        s *= max(0.0, 1.0 + gap)        # ordinary explanation wins: scale down
    s *= 0.85 + 0.15 * specificity
    s *= 0.7 + 0.3 * credibility
    return round(min(1.0, max(0.0, s)), 3)


class PairwiseDecision:
    name = NAME

    def __init__(self, gateway, settings):
        self.gw, self.settings = gateway, settings

    def score(self, context, state: State, episode: dict, hypotheses: HypothesisSet, abduction: Abduction, *,
              model=None, prompt_version=None, subject=None) -> Scored:
        best_baseline = max((b["fit"] for b in abduction.baselines), default=0.0)
        metrics = {"reach": round(abduction.reach, 3), "fit": round(abduction.fit, 3), "gap": round(abduction.fit - best_baseline, 3),
                   "specificity": abduction.specificity, "credibility": abduction.credibility}
        strength = strength_from(**metrics)
        weights = {abduction.hypothesis_id: abduction.fit}
        for h in hypotheses.hypotheses:
            weights.setdefault(h.id, 0.05)
        fits = {b["claim"].strip().lower(): b["fit"] for b in abduction.baselines}
        for b in hypotheses.baselines:
            weights[b.id] = fits.get(b.claim.strip().lower(), 0.05)
        weights[NONE] = max(0.0, 1.0 - max(weights.values()))
        dist = normalise(weights)
        prior = state.last.distribution if state.last else None
        post = Posterior(distribution=dist, info_gain=kl_bits(dist, prior), rationale="pairwise: posterior derived from abducer fits; strength from the documented formula",
                         decision=NAME, model=model or self.settings.model, prompt_version=prompt_version or "1", state_hash=state.hash())
        return Scored(posterior=post, strength=strength, metrics=metrics, unexplained=False, llm_call_ids=())

    next_action = staticmethod(not_implemented_action)
