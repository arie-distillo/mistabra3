"""The decision model: scores a risk's hypothesis set when a new episode (signal) arrives.

Roles are fixed across implementations:
  abducer  — the trajectory prompts (risk trajectory, baselines) propose the explanations; the engine owns it.
  decision — this interface: given context, the state of the hypothesis set and the abduction for this signal,
             return a posterior over {hypotheses, baselines, none} plus the signal's strength.
  engine   — persists trajectories, hypotheses, posterior and strength; creates a Lambda hypothesis when the
             set cannot explain the signal and re-scores once.

Implementations: `pairwise` (today's PoC method behind the interface) and `setwise` (the frame under test).
"""
from __future__ import annotations
import hashlib, json, math
from dataclasses import dataclass, field
from typing import Protocol

NONE = "none"


@dataclass(frozen=True)
class HypothesisRef:
    id: str                      # stored id, or "new" for the claim the abducer just proposed
    claim: str
    kind: str                    # "risk" | "baseline"
    is_lambda: bool = False
    predicts: tuple[str, ...] = ()
    status: str = "open"


@dataclass(frozen=True)
class HypothesisSet:
    risk_id: str
    hypotheses: tuple[HypothesisRef, ...]    # risk hypotheses, including "new" when the abducer proposed one
    baselines: tuple[HypothesisRef, ...]     # ordinary explanations (this signal's and the risk's earlier ones)

    @property
    def ids(self) -> list[str]:
        return [h.id for h in self.hypotheses] + [b.id for b in self.baselines] + [NONE]

    def by_id(self, hid: str) -> HypothesisRef | None:
        return next((h for h in (*self.hypotheses, *self.baselines) if h.id == hid), None)


@dataclass(frozen=True)
class AttachedSignal:
    signal_id: str
    summary: str
    occurred: str | None
    reach: float
    fit: float
    status: str


@dataclass(frozen=True)
class State:
    """What the set already knows: signals attached to each hypothesis, and the last posterior for the risk."""
    attached: dict[str, tuple[AttachedSignal, ...]]
    last: "Posterior | None"

    def hash(self) -> str:
        raw = json.dumps({k: [a.signal_id for a in v] for k, v in sorted(self.attached.items())}, sort_keys=True)
        raw += "|" + (json.dumps(self.last.distribution, sort_keys=True) if self.last else "")
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Abduction:
    """The abducer's proposal for this signal: one risk trajectory and the baseline trajectories."""
    hypothesis_id: str           # existing id, or "new"
    claim: str
    rationale: str
    reach: float
    fit: float
    entities: tuple[str, ...]
    missing: tuple[str, ...]
    predicts: tuple[str, ...]
    baselines: tuple[dict, ...]  # {claim, ordinary_outcome, rationale, fit}
    specificity: float
    credibility: float
    llm_call_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Posterior:
    distribution: dict[str, float]   # hypothesis id | "none" -> probability; sums to 1
    info_gain: float                 # bits: KL(posterior || prior), prior = last posterior or uniform
    rationale: str
    decision: str
    model: str
    prompt_version: str
    state_hash: str = ""

    @property
    def none(self) -> float:
        return self.distribution.get(NONE, 0.0)

    def risk_mass(self, hypotheses: HypothesisSet) -> float:
        return sum(self.distribution.get(h.id, 0.0) for h in hypotheses.hypotheses)


@dataclass(frozen=True)
class Scored:
    posterior: Posterior
    strength: float                  # the signal's strength under this decision model, in [0, 1]
    metrics: dict                    # fit/gap/reach/specificity/credibility — kept under both methods for replay
    unexplained: bool = False        # the set (including the abducer's proposal) does not explain the signal
    llm_call_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Action:
    kind: str                        # "run_radar" | "ask_analyst" | "alert" | "wait"
    target_id: str | None
    expected_info_gain: float
    why: str = ""


class DecisionModel(Protocol):
    name: str

    def score(self, context: list[str], state: State, episode: dict, hypotheses: HypothesisSet, abduction: Abduction, *,
              model: str | None = None, prompt_version: str | None = None, subject: tuple[str, str] | None = None) -> Scored: ...

    def next_action(self, context: list[str], state: State, options: tuple[Action, ...]) -> Action: ...   # reserved


# ---- shared arithmetic ----
def normalise(weights: dict[str, float], floor: float = 1e-4) -> dict[str, float]:
    w = {k: max(float(v), floor) for k, v in weights.items()}
    z = sum(w.values())
    return {k: round(v / z, 4) for k, v in w.items()}


def kl_bits(posterior: dict[str, float], prior: dict[str, float] | None) -> float:
    keys = list(posterior)
    if not prior:
        prior = {k: 1.0 / len(keys) for k in keys}
    p = normalise({k: posterior.get(k, 0.0) for k in keys})
    q = normalise({k: prior.get(k, 0.0) for k in keys})
    return round(sum(p[k] * math.log2(p[k] / q[k]) for k in keys), 4)


def not_implemented_action(context, state, options):
    raise NotImplementedError("next_action is reserved for the action side of the decision model (VOI probe)")
