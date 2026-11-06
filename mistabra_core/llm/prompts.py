"""Versioned prompts. Each renders to (system, user) with the input as a JSON block and declares the pydantic
schema its output must satisfy. Adapted from the PoC: radar.derive / radar.read / signal.lambda_check carried
over; the abducer now completes any missing part of a trajectory and reports match confidence; baselines are
assessed against the stored set first; standing.judge is new."""
from __future__ import annotations

import json
from dataclasses import dataclass

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------- output schemas
class Out(BaseModel):
    """Base of every output schema. Some models (e.g. Qwen 3.5 via OpenRouter) send nested objects and lists inside tool
    arguments as JSON-encoded strings — "signal": "{\"actor\": ...}" — or send "" for "none". Such strings are decoded,
    and empty strings in non-text fields are treated as not given, before validation."""

    @model_validator(mode="before")
    @classmethod
    def _decode_embedded_json(cls, data):
        if isinstance(data, str) and data.strip()[:1] in ("{", "["):
            data = _loads(data)
        if isinstance(data, dict):
            out = {}
            for k, v in data.items():
                if isinstance(v, str) and k in cls.model_fields and not _is_str_field(cls, k):
                    if not v.strip():
                        continue                      # "" for an object / list / number: not given, use the default
                    if v.strip()[:1] in ("{", "["):
                        v = _loads(v)
                out[k] = v
            data = out
        return data


def _loads(v: str):
    try:
        return json.loads(v.strip())
    except ValueError:
        return v


def _is_str_field(cls, name: str) -> bool:
    ann = cls.model_fields[name].annotation
    return ann is str or str(ann) in ("str | None", "typing.Optional[str]")


class ContentOut(Out):
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


class RadarDraft(Out):
    query: str


class RadarsOut(Out):
    radars: list[RadarDraft] = Field(default_factory=list)

    @field_validator("radars", mode="before")
    @classmethod
    def _strings_are_queries(cls, v):   # some models return ["query", ...] instead of [{"query": ...}]
        return [{"query": x} if isinstance(x, str) else x for x in v] if isinstance(v, list) else v


class RadarReadOut(Out):
    relevant: bool
    confidence: float = 0.5
    content: ContentOut | None = None


class LambdaCheckOut(Out):
    relevant: bool
    confidence: float = 0.5
    why: str = ""
    content: ContentOut | None = None


class TrajectoryDraft(Out):
    hypothesis_id: str | None = None       # a stored id, "given", or null for a new claim
    claim: str | None = None
    match_confidence: float = 0.0
    rationale: str = ""
    reach: float = 0.0
    fit: float = 0.0
    entities: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    stance: Literal["supports", "weakens"] = "supports"


class StoredStance(Out):
    hypothesis_id: str
    stance: Literal["supports", "weakens", "unrelated"] = "unrelated"
    fit: float = 0.0
    rationale: str = ""


class AbduceOut(Out):
    relevant: bool = True
    signal: ContentOut | None = None
    trajectories: list[TrajectoryDraft] = Field(default_factory=list)
    stored: list[StoredStance] = Field(default_factory=list)   # one entry per stored hypothesis


class BaselineAssessed(Out):
    baseline_id: str
    fit: float
    ordinary_outcome: str = ""
    rationale: str = ""


class BaselineNew(Out):
    claim: str
    ordinary_outcome: str = ""
    rationale: str = ""
    fit: float


class BaselinesOut(Out):
    assessed: list[BaselineAssessed] = Field(default_factory=list)
    new: list[BaselineNew] = Field(default_factory=list)


class StandingOut(Out):
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
                "Answer by calling the output tool. Pass nested objects and lists as JSON objects and arrays, "
                "never as strings containing JSON.")
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
    "trajectory.abduce": Prompt("trajectory.abduce", "3", ANALYST,
        "Complete and assess the trajectory episode -> signal -> hypothesis -> risk.\n"
        "- signal: if INPUT signal is null, articulate what in the episode connects it to the risk (or to the given hypothesis) as signal "
        "content; if a signal is given (content or analyst note), keep it as is and return signal null. " + CONTENT + "\n"
        "- If given_hypothesis is set, return exactly one trajectory with hypothesis_id \"given\" assessing that hypothesis.\n"
        "- Otherwise return up to `max` trajectories, best first, each to a distinct hypothesis that could connect the episode to the risk. "
        "Reuse a stored hypothesis whenever it expresses the same claim: set hypothesis_id to its id and match_confidence (0-1) to how "
        "sure you are it is the same claim. Only when none fits, set hypothesis_id null and state a new claim (a claim about the world, "
        "e.g. 'the house on Elm Street is a target of hostile preparation'). Never propose a claim that a rejected hypothesis already "
        "expresses; if the episode supports a rejected one, reference its id.\n"
        "- Every hypothesis is a claim under which the risk becomes MORE likely. Ordinary explanations (things are normal, routine, "
        "benign, under control) are NOT hypotheses here: they are baselines, assessed separately; never return them as trajectories.\n"
        "- stored: one entry for EVERY stored hypothesis, with its stance toward this episode: \"supports\" (the episode makes it more "
        "likely; also return it in trajectories), \"weakens\" (the episode makes it less likely or removes its mechanism: "
        "counter-evidence), or \"unrelated\"; fit 0-1 (low for weakens); rationale one sentence.\n"
        "- If given_hypothesis is set and the episode makes it less likely, return its trajectory with stance \"weakens\".\n"
        "- relevant: false (and no trajectories) only if the episode neither supports nor weakens any way to the risk.\n"
        "- rationale: 2-4 sentences telling the story from the episode via the hypothesis to the risk. reach: how far along the path to "
        "the risk the episode sits (0 = remote precursor, 1 = the risk itself). fit: how well the episode supports the hypothesis on its "
        "own (raise for concealment, repetition, odd hours, tooling, explicit intent, named target; lower for a named benign role or "
        "stated purpose). missing: information that would change the assessment. Use precedents (confirmed past trajectories) for consistency.\n"
        "Schema: {\"relevant\":bool,\"signal\":{...}|null,\"trajectories\":[{\"hypothesis_id\":str|null,\"claim\":str|null,"
        "\"match_confidence\":0-1,\"rationale\":str,\"reach\":0-1,\"fit\":0-1,\"entities\":[str],\"missing\":[str],\"stance\":\"supports\"|\"weakens\"}],"
        "\"stored\":[{\"hypothesis_id\":str,\"stance\":\"supports\"|\"weakens\"|\"unrelated\",\"fit\":0-1,\"rationale\":str}]}", AbduceOut, "reasoning"),
    "trajectory.baselines": Prompt("trajectory.baselines", "1", ANALYST,
        "Weigh the ordinary explanations of the same episode against the hypothesis. First assess every stored baseline: for each, the "
        "ordinary outcome it ends in, a one-to-two sentence rationale, and fit — how well the episode supports THIS ordinary explanation "
        "given everything stated (0 = implausible, 1 = fully explained; a named benign role or stated purpose raises it; concealment, "
        "repetition, anonymity, tooling lower it). Then add up to `max_new` new baselines only for ordinary explanations none of the stored "
        "ones covers (none if they cover it). "
        "Schema: {\"assessed\":[{\"baseline_id\":str,\"fit\":0-1,\"ordinary_outcome\":str,\"rationale\":str}],"
        "\"new\":[{\"claim\":str,\"ordinary_outcome\":str,\"rationale\":str,\"fit\":0-1}]}", BaselinesOut, "reasoning"),
    "standing.judge": Prompt("standing.judge", "4", ANALYST,
        "Judge how well the hypothesis is supported, against its ordinary explanations (baselines), by ALL the trajectories attached to "
        "it, taken together. Each trajectory has a status and a weight: confirmed by the analyst (weight 1), proposed by the system and "
        "not yet confirmed (lower weight; treat as less certain), rejected by the analyst (weight 0: ignore it as support). Combine them as "
        "an analyst would: several individually weak trajectories that point the same way can make the hypothesis likely; two that report "
        "the same fact count once; a trajectory whose best explanation is ordinary adds little; a trajectory with stance \"weakens\" is "
        "counter-evidence and lowers the hypothesis in proportion to its weight; evidence that rules out the hypothesis's "
        "mechanism lowers it even if each item alone looks neutral. Use the whole scale: 0 = refuted; 0.05-0.2 = a single weak or unconfirmed indication; 0.2-0.4 = some support, ordinary "
        "explanations still as likely; 0.4-0.6 = several indications pointing the same way, ordinary explanations strained; 0.6-0.8 = "
        "strong, specific support from independent sources; 0.8-1 = confirmed by the analyst or near-certain. Proposed (unconfirmed) "
        "trajectories alone should rarely put a hypothesis above 0.7. When `previous_standing` is given, it is the standing before "
        "the trajectory with status \"this\" (the new episode): judge the update it causes. A supporting \"this\" keeps or raises the "
        "value; a \"this\" with stance \"weakens\" must lower it below previous_standing, the more so the more directly it removes the "
        "hypothesis's mechanism (an episode that removes the very condition the hypothesis depends on can halve it or more). value: 0-1. rationale: 2-4 "
        "sentences. Schema: {\"value\":0-1,\"rationale\":str}", StandingOut, "reasoning"),
}
