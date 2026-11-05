"""`setwise`: the decision-model frame under test, with the reasoning model as the scorer.

One call scores the whole hypothesis set for the risk — risk hypotheses, their baselines, and "none of these" —
given the context, what is already attached to each hypothesis (the state) and the abducer's proposal for this
signal. The output is a distribution; the signal's strength is the posterior mass on the risk hypotheses, and
information gain is KL to the previous posterior. "none" above the threshold marks the signal unexplained, and
the engine then creates a Lambda hypothesis and re-scores once.

This is the reference implementation: later, a distilled model is trained on its outputs and plugs in here.
"""
from __future__ import annotations
from .base import NONE, Abduction, HypothesisSet, Posterior, Scored, State, kl_bits, normalise, not_implemented_action
from .pairwise import strength_from

NAME = "setwise"
PROMPT = "decision.setwise"


class SetwiseDecision:
    name = NAME

    def __init__(self, gateway, settings):
        self.gw, self.settings = gateway, settings

    def score(self, context, state: State, episode: dict, hypotheses: HypothesisSet, abduction: Abduction, *,
              model=None, prompt_version=None, subject=None) -> Scored:
        prior = state.last.distribution if state.last else None
        variables = {
            "episode": episode, "context": context,
            "hypotheses": [{"id": h.id, "kind": h.kind, "claim": h.claim, "predicts": list(h.predicts),
                            "attached": [{"summary": a.summary, "occurred": a.occurred, "fit": a.fit, "reach": a.reach, "status": a.status}
                                         for a in state.attached.get(h.id, ())]}
                           for h in (*hypotheses.hypotheses, *hypotheses.baselines)],
            "abduction": {"hypothesis_id": abduction.hypothesis_id, "claim": abduction.claim, "rationale": abduction.rationale,
                          "reach": abduction.reach, "fit": abduction.fit,
                          "baselines": [{"claim": b["claim"], "ordinary_outcome": b["ordinary_outcome"], "rationale": b["rationale"], "fit": b["fit"]} for b in abduction.baselines]},
            "prior": prior, "none_id": NONE,
        }
        out, call_id = self.gw.call(PROMPT, variables, subject=subject, model=model or self.settings.decision_model or None, prompt_version=prompt_version)
        dist = normalise({k: float(out.distribution.get(k, 0.0)) for k in hypotheses.ids})
        post = Posterior(distribution=dist, info_gain=kl_bits(dist, prior), rationale=out.rationale, decision=NAME,
                         model=model or self.gw.model_for("reasoning"), prompt_version=prompt_version or self.gw.version_of(PROMPT), state_hash=state.hash())
        best_baseline = max((b["fit"] for b in abduction.baselines), default=0.0)
        metrics = {"reach": round(abduction.reach, 3), "fit": round(abduction.fit, 3), "gap": round(abduction.fit - best_baseline, 3),
                   "specificity": abduction.specificity, "credibility": abduction.credibility,
                   "pairwise_strength": strength_from(abduction.reach, abduction.fit, abduction.fit - best_baseline, abduction.specificity, abduction.credibility)}
        strength = round(post.risk_mass(hypotheses), 3)
        return Scored(posterior=post, strength=strength, metrics=metrics, unexplained=post.none >= self.settings.none_threshold, llm_call_ids=(call_id,))

    next_action = staticmethod(not_implemented_action)
