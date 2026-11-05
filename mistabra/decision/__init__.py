from .base import (NONE, Abduction, Action, AttachedSignal, DecisionModel, HypothesisRef, HypothesisSet, Posterior, Scored, State,
                   kl_bits, normalise)
from .pairwise import PairwiseDecision, strength_from
from .setwise import SetwiseDecision

DECISIONS = {"pairwise": PairwiseDecision, "setwise": SetwiseDecision}


def make_decision(name: str, gateway, settings) -> DecisionModel:
    try:
        return DECISIONS[name](gateway, settings)
    except KeyError:
        raise ValueError(f"unknown decision model {name!r} (expected one of {sorted(DECISIONS)})") from None


__all__ = ["NONE", "Abduction", "Action", "AttachedSignal", "DecisionModel", "HypothesisRef", "HypothesisSet", "Posterior", "Scored", "State",
           "kl_bits", "normalise", "PairwiseDecision", "SetwiseDecision", "strength_from", "DECISIONS", "make_decision"]
