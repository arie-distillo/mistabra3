"""Public types of the Mistabra Core API (spec v3, sections 3–4).

Inputs: MissionContext, OperationalContext, TrajectoryInput. Output: Assessment.
All models are frozen: `assess` reads its inputs and never mutates them.
"""
from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)


# ---------------------------------------------------------------- mission context
class Document(Frozen):
    id: str
    title: str = ""
    text: str


class Risk(Frozen):
    id: str
    text: str
    polarity: Literal["negative", "positive"] = "negative"
    status: Literal["active", "retired"] = "active"


class MissionContext(Frozen):
    """What the analyst or organization gives the system to define the mission. Versioned."""
    version: str
    documents: tuple[Document, ...] = ()
    risks: tuple[Risk, ...] = ()

    def risk(self, risk_id: str) -> Risk | None:
        return next((r for r in self.risks if r.id == risk_id), None)

    @property
    def active_risks(self) -> tuple[Risk, ...]:
        return tuple(r for r in self.risks if r.status == "active")


# ---------------------------------------------------------------- trajectory elements
class SignalContent(Frozen):
    """What in an episode is relevant to a hypothesis. All fields optional."""
    actor: str | None = None
    behaviour: str | None = None
    target: str | None = None
    means: str | None = None
    time: str | None = None
    modifiers: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    summary: str = ""

    @property
    def specificity(self) -> float:
        return 1.0 if self.target else (0.5 if self.entities else 0.2)


class Episode(Frozen):
    id: str
    text: str
    source: str = "unknown"
    credibility: float = Field(0.5, ge=0, le=1)
    occurred: str | tuple[str, str] | None = None


# ---------------------------------------------------------------- operational context
HypothesisStatus = Literal["proposed", "open", "supported", "rejected"]
TrajectoryStatus = Literal["proposed", "confirmed", "rejected"]


class Hypothesis(Frozen):
    id: str
    version: int = 1
    risk_id: str
    claim: str
    entities: tuple[str, ...] = ()
    origin: Literal["analyst", "system"] = "analyst"
    role: Literal["primary", "baseline"] = "primary"
    is_lambda: bool = Field(False, alias="lambda")
    status: HypothesisStatus = "open"
    baseline_of: tuple[str, ...] = ()          # for baselines: the primaries this one competes with
    refines: str | None = None                 # the held hypothesis whose mechanism this one corrects (v3.2)
    standing: float | None = None              # the last standing the layer reported for it, kept by the other layer, which
                                               # clears it when the status of attached evidence changes (v3.2); anchors `without`


class StoredTrajectory(Frozen):
    """A trajectory the other layer kept, in the shape of AssessedTrajectory, with its status."""
    id: str
    version: int = 1
    episode_id: str
    occurred: str | None = None
    hypothesis_id: str
    risk_id: str
    role: Literal["primary", "baseline"] = "primary"
    signal: SignalContent = SignalContent()
    score: float = 0.0
    fit: float = 0.0
    reach: float = 0.0
    rationale: str = ""
    ordinary_outcome: str | None = None
    stance: Literal["supports", "weakens", "neutral"] = "supports"
    status: TrajectoryStatus = "proposed"


class OperationalContext(Frozen):
    """What the system and analyst currently believe: hypotheses with relations, and every trajectory kept."""
    hypotheses: tuple[Hypothesis, ...] = ()
    trajectories: tuple[StoredTrajectory, ...] = ()

    def hypothesis(self, hid: str) -> Hypothesis | None:
        return next((h for h in self.hypotheses if h.id == hid), None)

    def primaries(self, risk_id: str, *, include_rejected: bool = False) -> list[Hypothesis]:
        return [h for h in self.hypotheses if h.risk_id == risk_id and h.role == "primary"
                and (include_rejected or h.status != "rejected")]

    def baselines_of(self, hid: str) -> list[Hypothesis]:
        return [h for h in self.hypotheses if h.role == "baseline" and hid in h.baseline_of and h.status != "rejected"]

    def attached(self, hid: str) -> list[StoredTrajectory]:
        return [t for t in self.trajectories if t.hypothesis_id == hid and t.role == "primary"]

    def precedents(self, risk_id: str, k: int = 5) -> list[StoredTrajectory]:
        out = [t for t in self.trajectories if t.risk_id == risk_id and t.status == "confirmed" and t.role == "primary"]
        return out[-k:]


# ---------------------------------------------------------------- input
class SignalDraft(Frozen):
    """A signal the analyst wrote: content fields and/or a free-text note of what matters."""
    content: SignalContent | None = None
    note: str = ""

    @model_validator(mode="after")
    def _non_empty(self):
        if self.content is None and not self.note.strip():
            raise ValueError("a signal draft needs content or a note")
        return self


class HypothesisDraft(Frozen):
    claim: str


class TrajectoryInput(Frozen):
    """Exactly one episode; any of signal, hypothesis (stored id or draft) and risk (stored id)."""
    episode: Episode
    signal: SignalDraft | None = None
    hypothesis: Union[str, HypothesisDraft, None] = None
    risk: str | None = None

    @property
    def pattern(self) -> int:
        """The input pattern number of spec v3 section 4.2 (1–8)."""
        key = (self.signal is not None, self.hypothesis is not None, self.risk is not None)
        return {(False, False, False): 1, (True, False, False): 2, (False, True, False): 3, (False, False, True): 4,
                (True, True, False): 5, (True, False, True): 6, (False, True, True): 7, (True, True, True): 8}[key]


# ---------------------------------------------------------------- output
Source = Literal["given", "predefined", "articulated"]


class SignalOut(Frozen):
    content: SignalContent
    source: Source
    radar: str | None = None                   # the radar query that matched, when predefined


class HypothesisRef(Frozen):
    source: Source
    id: str | None = None                      # given (stored) or predefined
    version: int | None = None
    claim: str
    entities: tuple[str, ...] = ()
    match_confidence: float | None = None      # predefined only
    status: HypothesisStatus | None = None     # of the stored hypothesis, if any
    refines: str | None = None                 # articulated only: the held hypothesis whose mechanism this one corrects


class RiskRef(Frozen):
    risk_id: str
    source: Literal["given", "predefined"]


class AssessedTrajectory(Frozen):
    episode_id: str
    role: Literal["primary", "baseline"]
    signal: SignalOut
    hypothesis: HypothesisRef
    risk: RiskRef
    score: float
    reach: float
    fit: float
    missing: tuple[str, ...] = ()
    rationale: str | None = None
    ordinary_outcome: str | None = None
    stance: Literal["supports", "weakens", "neutral"] = "supports"   # weakens: counter-evidence; neutral: a given hypothesis the episode does not bear on

    @property
    def lambda_signal(self) -> bool:
        return self.signal.source == "articulated"

    @property
    def lambda_hypothesis(self) -> bool:
        return self.hypothesis.source == "articulated" and self.role == "primary"


class StandingUse(Frozen):
    trajectory_id: str
    status: TrajectoryStatus | Literal["this"]
    weight: float


class Standing(Frozen):
    value: float
    without: float
    delta: float
    used: tuple[StandingUse, ...] = ()
    rationale: str | None = None


class AssessmentGroup(Frozen):
    target: AssessedTrajectory
    baselines: tuple[AssessedTrajectory, ...] = ()
    standing: Standing
    inconsistency: str | None = None


class LLMCallRecord(Frozen):
    task: str
    prompt_version: str
    model: str
    cached: bool
    ok: bool
    latency_ms: int
    tokens_in: int | None = None
    tokens_out: int | None = None
    retries: int = 0                           # validation failures sent back to the model before it answered


class Provenance(Frozen):
    implementation: str
    implementation_version: str
    models: tuple[str, ...]
    prompts: dict[str, str]
    mission_version: str
    operational_items: dict[str, int]          # id -> version of every hypothesis and trajectory read
    input_hash: str
    deterministic: bool
    calls: tuple[LLMCallRecord, ...] = ()


class Assessment(Frozen):
    groups: tuple[AssessmentGroup, ...]
    pattern: int
    provenance: Provenance

    def group_for(self, hypothesis_id: str) -> AssessmentGroup | None:
        return next((g for g in self.groups if g.target.hypothesis.id == hypothesis_id), None)
