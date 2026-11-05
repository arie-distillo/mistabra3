"""Versioned prompts. Each renders to (system, user) with the input as a JSON block and declares the pydantic
schema its output must satisfy. Adapted from the PoC: radar.derive / radar.read / signal.lambda_check carried
over; the abducer now completes any missing part of a trajectory and reports match confidence; baselines are
assessed against the stored set first; standing.judge is new."""
from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------- output schemas
class ContentOut(BaseModel):
    @field_validator("modifiers", "entities", mode="before", check_fields=False)
    @classmethod
    def _listify(cls, v):                 # a single string where a list is expected
        return [v] if isinstance(v, str) else (v or [])

    actor: str | None = None
    behaviour: str | None = None
    target: str | None = None
    means: str | None = None
    time: str | None = None
    modifiers: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    summary: str = ""


class RadarDraft(BaseModel):
    query: str


class RadarsOut(BaseModel):
    radars: list[RadarDraft] = Field(default_factory=list)

    @field_validator("radars", mode="before")
    @classmethod
    def _strings_are_queries(cls, v):   # some models return ["query", ...] instead of [{"query": ...}]
        return [{"query": x} if isinstance(x, str) else x for x in v] if isinstance(v, list) else v


class RadarReadOut(BaseModel):
    relevant: bool
    confidence: float = 0.5
    content: ContentOut | None = None


class LambdaCheckOut(BaseModel):
    relevant: bool
    confidence: float = 0.5
    why: str = ""
    content: ContentOut | None = None


class TrajectoryDraft(BaseModel):
    hypothesis_id: str | None = None       # a stored id, "given", or null for a new claim
    claim: str | None = None
    match_confidence: float = 0.0
    rationale: str = ""
    reach: float = 0.0
    fit: float = 0.0
    entities: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class AbduceOut(BaseModel):
    relevant: bool = True
    signal: ContentOut | None = None
    trajectories: list[TrajectoryDraft] = Field(default_factory=list)


class BaselineAssessed(BaseModel):
    baseline_id: str
    fit: float
    ordinary_outcome: str = ""
    rationale: str = ""


class BaselineNew(BaseModel):
    claim: str
    ordinary_outcome: str = ""
    rationale: str = ""
    fit: float


class BaselinesOut(BaseModel):
    assessed: list[BaselineAssessed] = Field(default_factory=list)
    new: list[BaselineNew] = Field(default_factory=list)


class StandingOut(BaseModel):
    value: float
    rationale: str = ""


@dataclass(frozen=True)
class Prompt:
    id: str
    version: str
    system: str
    instructions: str
    schema: type[BaseModel]
    tier: str                                   # reasoning | fast

    def render(self, variables: dict) -> tuple[str, str]:
        user = (f"TASK: {self.id}\n\n{self.instructions}\n\nINPUT_JSON:\n{json.dumps(variables, ensure_ascii=False, indent=1, sort_keys=True)}\n\n"
                "Respond with a single JSON object matching the schema and nothing else.")
        return self.system, user


ANALYST = ("You are a careful intelligence analyst's co-pilot. You reason about whether an observed episode could be a step "
           "toward a stated risk through a hypothesis about the world, and you always weigh the ordinary explanations of the "
           "same observation. You never infer intent the episode does not show. Output only valid JSON.")

CONTENT = ("Signal content fields: actor, behaviour, target, means, time, modifiers (e.g. concealment, repetition, night, tooling, "
           "anonymity, explicit intent), entities (named people, places, organisations), summary (one sentence).")

PROMPTS: dict[str, Prompt] = {
    "radar.derive": Prompt("radar.derive", "1", ANALYST,
        "Given a risk, domain context and the hypotheses currently held about it, write 3-6 radars. A radar is a query that decides "
        "whether an episode (a short text report) is relevant to the risk and asks for something specific to extract. Each query is one "
        "or two sentences, concrete, and covers a distinct kind of precursor or observable. "
        "Schema: {\"radars\":[{\"query\":str}]}", RadarsOut, "reasoning"),
    "radar.read": Prompt("radar.read", "1", ANALYST,
        "Decide whether the episode is what the radar's query asks for, for the given risk. Be literal: the radar must actually match. "
        "If it matches, extract the signal content. " + CONTENT + " "
        "Schema: {\"relevant\":bool,\"confidence\":0-1,\"content\":{...}|null}", RadarReadOut, "fast"),
    "signal.lambda_check": Prompt("signal.lambda_check", "1", ANALYST,
        "No radar matched this episode. Using the risk statement and context only, judge whether the episode could nevertheless be a step "
        "toward the risk — something a thoughtful analyst would want to look at even though no query anticipated it. Be conservative: most "
        "episodes are not relevant to most risks. If relevant, extract the signal content and say why in one sentence. " + CONTENT + " "
        "Schema: {\"relevant\":bool,\"confidence\":0-1,\"why\":str,\"content\":{...}|null}", LambdaCheckOut, "fast"),
    "trajectory.abduce": Prompt("trajectory.abduce", "1", ANALYST,
        "Complete and assess the trajectory episode -> signal -> hypothesis -> risk.\n"
        "- signal: if INPUT signal is null, articulate what in the episode connects it to the risk (or to the given hypothesis) as signal "
        "content; if a signal is given (content or analyst note), keep it as is and return signal null. " + CONTENT + "\n"
        "- If given_hypothesis is set, return exactly one trajectory with hypothesis_id \"given\" assessing that hypothesis.\n"
        "- Otherwise return up to `max` trajectories, best first, each to a distinct hypothesis that could connect the episode to the risk. "
        "Reuse a stored hypothesis whenever it expresses the same claim: set hypothesis_id to its id and match_confidence (0-1) to how "
        "sure you are it is the same claim. Only when none fits, set hypothesis_id null and state a new claim (a claim about the world, "
        "e.g. 'the house on Elm Street is a target of hostile preparation'). Never propose a claim that a rejected hypothesis already "
        "expresses; if the episode supports a rejected one, reference its id.\n"
        "- relevant: false (and no trajectories) if the episode plausibly connects to the risk through no hypothesis at all.\n"
        "- rationale: 2-4 sentences telling the story from the episode via the hypothesis to the risk. reach: how far along the path to "
        "the risk the episode sits (0 = remote precursor, 1 = the risk itself). fit: how well the episode supports the hypothesis on its "
        "own (raise for concealment, repetition, odd hours, tooling, explicit intent, named target; lower for a named benign role or "
        "stated purpose). missing: information that would change the assessment. Use precedents (confirmed past trajectories) for consistency.\n"
        "Schema: {\"relevant\":bool,\"signal\":{...}|null,\"trajectories\":[{\"hypothesis_id\":str|null,\"claim\":str|null,"
        "\"match_confidence\":0-1,\"rationale\":str,\"reach\":0-1,\"fit\":0-1,\"entities\":[str],\"missing\":[str]}]}", AbduceOut, "reasoning"),
    "trajectory.baselines": Prompt("trajectory.baselines", "1", ANALYST,
        "Weigh the ordinary explanations of the same episode against the hypothesis. First assess every stored baseline: for each, the "
        "ordinary outcome it ends in, a one-to-two sentence rationale, and fit — how well the episode supports THIS ordinary explanation "
        "given everything stated (0 = implausible, 1 = fully explained; a named benign role or stated purpose raises it; concealment, "
        "repetition, anonymity, tooling lower it). Then add up to `max_new` new baselines only for ordinary explanations none of the stored "
        "ones covers (none if they cover it). "
        "Schema: {\"assessed\":[{\"baseline_id\":str,\"fit\":0-1,\"ordinary_outcome\":str,\"rationale\":str}],"
        "\"new\":[{\"claim\":str,\"ordinary_outcome\":str,\"rationale\":str,\"fit\":0-1}]}", BaselinesOut, "reasoning"),
    "standing.judge": Prompt("standing.judge", "1", ANALYST,
        "Judge how well the hypothesis is supported, against its ordinary explanations (baselines), by ALL the trajectories attached to "
        "it, taken together. Each trajectory has a status and a weight: confirmed by the analyst (weight 1), proposed by the system and "
        "not yet confirmed (lower weight; treat as less certain), rejected by the analyst (weight 0: ignore it as support). Combine them as "
        "an analyst would: several individually weak trajectories that point the same way can make the hypothesis likely; two that report "
        "the same fact count once; a trajectory whose best explanation is ordinary adds little; evidence that rules out the hypothesis's "
        "mechanism lowers it even if each item alone looks neutral. value: 0 = unsupported or refuted, 1 = established. rationale: 2-4 "
        "sentences. Schema: {\"value\":0-1,\"rationale\":str}", StandingOut, "reasoning"),
}
