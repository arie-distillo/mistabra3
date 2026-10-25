"""Public, frozen models returned by reads."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class ContextItem(Frozen):
    id: str; kind: str; title: str | None; text: str; tags: list[str]; origin: str; created_at: str


class Risk(Frozen):
    id: str; text: str; polarity: Literal["negative", "positive"]; open: bool; status: str
    entities: list[str]; origin: str; created_at: str


class Radar(Frozen):
    id: str; risk_id: str; query: str; enabled: bool; origin: str; created_at: str


class Source(Frozen):
    id: str; name: str; type: str; credibility: float


class Episode(Frozen):
    id: str; source_id: str; text: str; occurred_kind: str; occurred_start: str | None; occurred_end: str | None
    metadata: dict; created_at: str


class SignalContent(BaseModel):
    """Structured extraction: what in the episode is relevant to the risk. All fields optional."""
    actor: str | None = None
    behaviour: str | None = None
    target: str | None = None
    means: str | None = None
    time: str | None = None
    modifiers: list[str] = Field(default_factory=list)      # e.g. concealment, repetition, night, tooling
    entities: list[str] = Field(default_factory=list)
    summary: str = ""


class Metrics(Frozen):
    reach: float; fit: float; gap: float; specificity: float; credibility: float


class Trajectory(Frozen):
    id: str; signal_id: str; episode_id: str; risk_id: str; radar_id: str | None; hypothesis_id: str
    outcome: Literal["risk", "ordinary"]; ordinary_outcome: str | None
    reach: float; fit: float; rationale: str; status: str; origin: str; created_at: str


class Signal(Frozen):
    id: str; episode_id: str; risk_id: str; origin: str
    via: Literal["radar", "lambda", "analyst"]; radar_id: str | None
    content: SignalContent; note: str | None
    strength_system: float | None; strength_analyst: float | None
    metrics: Metrics | None; missing: list[str]; status: str
    trajectory: Trajectory | None; baselines: list[Trajectory]
    versions: dict; created_at: str


class Hypothesis(Frozen):
    id: str; risk_id: str; claim: str; entities: list[str]; status: str; origin: str; created_at: str


class CoverageItem(Frozen):
    hypothesis: Hypothesis; support: int; max_reach: float; first_seen: str | None; last_seen: str | None


class Coverage(Frozen):
    risk_id: str; items: list[CoverageItem]; path_reach: float      # best reach among supported hypotheses
