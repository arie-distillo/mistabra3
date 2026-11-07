"""Versioned prompts. Each renders to (system, user) with the input as a JSON block and declares the pydantic
schema its output must satisfy.

Three calls per episode and risk (plus radar.derive, cached while the stored hypotheses do not change):
  episode.triage  (fast)       is the episode relevant, directly or through a held hypothesis; which radar it answers
  episode.assess  (reasoning)  a verdict on every held hypothesis, new hypotheses, the ordinary explanations of each
  standing.judge  (reasoning)  the standing of every hypothesis the episode touched, before and after it
"""
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


class TriageOut(Out):
    relevant: bool = False
    confidence: float = 0.5
    radar: int | None = None                 # index of the radar the episode answers, if any
    touches: list[str] = Field(default_factory=list)   # stored hypotheses the episode bears on, for or against
    why: str = ""
    content: ContentOut | None = None


class BaselineAssessed(Out):
    baseline_id: str
    fit: float = 0.0
    ordinary_outcome: str = ""
    rationale: str = ""


class BaselineNew(Out):
    claim: str
    ordinary_outcome: str = ""
    rationale: str = ""
    fit: float = 0.0


class BaselinesOut(Out):
    assessed: list[BaselineAssessed] = Field(default_factory=list)
    new: list[BaselineNew] = Field(default_factory=list)


class _Assessment(Out):
    fit: float = 0.0
    reach: float = 0.0
    rationale: str = ""
    missing: list[str] = Field(default_factory=list)
    baselines: BaselinesOut = Field(default_factory=BaselinesOut)


class StoredVerdict(_Assessment):
    hypothesis_id: str
    stance: Literal["supports", "weakens", "unrelated"] = "unrelated"
    confidence: float = 0.8                  # that the stance is right
    revised_claim: str | None = None         # the corrected mechanism, when the evidence shows it works differently


class NewHypothesis(_Assessment):
    claim: str
    entities: list[str] = Field(default_factory=list)
    refines: str | None = None               # id of the held hypothesis whose mechanism this one corrects


class GivenVerdict(_Assessment):
    stance: Literal["supports", "weakens", "unrelated"] = "supports"
    entities: list[str] = Field(default_factory=list)


class AssessOut(Out):
    relevant: bool = True
    signal: ContentOut | None = None
    given: GivenVerdict | None = None
    stored: list[StoredVerdict] = Field(default_factory=list)
    new: list[NewHypothesis] = Field(default_factory=list)


class SharpenOut(Out):
    sharper_claim: str | None = None
    why: str = ""


class StandingItem(Out):
    key: str
    before: float = 0.0
    value: float = 0.0
    rationale: str = ""


class StandingsOut(Out):
    standings: list[StandingItem] = Field(default_factory=list)


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

_SCALE = ("Use the whole scale: 0 = refuted; 0.05-0.2 = a single weak or unconfirmed indication; 0.2-0.4 = some support, ordinary "
          "explanations still as likely; 0.4-0.6 = several indications pointing the same way, ordinary explanations strained; 0.6-0.8 = "
          "strong, specific support from independent sources; 0.8-1 = confirmed by the analyst or near-certain. Proposed (unconfirmed) "
          "trajectories alone never put a hypothesis above 0.8: that range needs confirmation by the analyst.")

PROMPTS: dict[str, Prompt] = {
    "radar.derive": Prompt("radar.derive", "1", ANALYST,
        "Given a risk, domain context and the hypotheses currently held about it, write 3-6 radars. A radar is a query that decides "
        "whether an episode (a short text report) is relevant to the risk and asks for something specific to extract. Each query is one "
        "or two sentences, concrete, and covers a distinct kind of precursor or observable. "
        "Schema: {\"radars\":[{\"query\":str}]}", RadarsOut, "reasoning"),
    "episode.triage": Prompt("episode.triage", "1", ANALYST,
        "Decide whether the episode bears on the risk, directly or through one of the hypotheses currently held about it.\n"
        "- Judge the episode in the light of the held hypotheses and their evidence: a fact that is unremarkable on its own (a device "
        "update, a printer running at night, a budget line) is relevant when it confirms, extends or contradicts the mechanism a held "
        "hypothesis proposes.\n"
        "- touches: the ids of held hypotheses the episode bears on, for or against.\n"
        "- radar: the index of the radar whose query the episode answers (be literal: it must actually match), or null.\n"
        "- relevant: true if the episode touches a held hypothesis, answers a radar, or could be a step toward the risk that a thoughtful "
        "analyst would want to look at. Be conservative otherwise: most episodes are not relevant to most risks. confidence 0-1.\n"
        "- If relevant, content: what in the episode connects it to the risk. " + CONTENT + " why: one sentence.\n"
        "Schema: {\"relevant\":bool,\"confidence\":0-1,\"radar\":int|null,\"touches\":[str],\"why\":str,\"content\":{...}|null}",
        TriageOut, "fast"),
    "episode.assess": Prompt("episode.assess", "5", ANALYST,
        "Assess the episode against the risk and the hypotheses held about it (episode -> signal -> hypothesis -> risk).\n"
        "- signal: if both INPUT signal and detected_signal are null, articulate what in the episode connects it to the risk (or to the "
        "given hypothesis) as signal content; otherwise return null. " + CONTENT + "\n"
        "- If given_hypothesis is set: assess only that hypothesis, in `given`: stance supports, weakens, or unrelated; leave stored and "
        "new empty.\n"
        "- Stances, for held and given hypotheses alike: \"unrelated\" is the usual verdict for an episode that neither makes the "
        "hypothesis more nor less likely, including routine details merely consistent with normal operation (a manifest signed, the "
        "kind of cargo, a log-in). \"weakens\" needs a reason the hypothesis becomes less likely: the episode contradicts its "
        "mechanism, removes a condition it depends on, or shows a competing cause actually at work. A competing cause that is "
        "merely possible is not a reason. Decide the stance with the hypothesis's `evidence` in view: facts already on file, each "
        "marked with the stance it got alone, (neutral) ones included. A fact that is neutral alone can complete a conjunction with "
        "this episode (a recorded payload plus a stated limit shows an overload; a vehicle's range plus a distance rules a route "
        "out): then the stance follows from the conjunction.\n"
        "- Otherwise:\n"
        "  - stored: one verdict for EVERY held hypothesis (rejected ones included): stance \"supports\" (the episode makes it more "
        "likely), \"weakens\" (less likely, or it removes the condition the hypothesis depends on), or \"unrelated\"; confidence 0-1 "
        "that the stance is right. Each held hypothesis lists recent evidence already attached to it: judge the episode in that light; "
        "a fact that is unremarkable alone can confirm the mechanism the evidence points to. \"supports\" means the episode fits that "
        "hypothesis's own mechanism. Do not stretch a held hypothesis to cover evidence that another mechanism explains more "
        "directly (e.g. registration and ownership changes are not 'layering of funds'): give it \"unrelated\" and state that "
        "mechanism as a new hypothesis. \"weakens\" means the episode cuts against the hypothesis's core mechanism, not against a "
        "side detail or an example in its wording.\n"
        "  - revised_claim (on each verdict): read this episode together with the hypothesis's evidence and ask whether the mechanism "
        "works differently from how the claim states it, in a way that matters (not merely more detail; e.g. the data left the "
        "building on paper, not over the network). If so, write the corrected claim; otherwise null. Most of the time it is null.\n"
        "  - new: at most `max_new` hypotheses that no held one states, best first, only if the episode points to a mechanism the held "
        "ones miss. A hypothesis is a specific mechanism by which the risk comes about: name the concrete actors, systems, places and "
        "means the episode shows (e.g. 'an insider used the print server to print out the customer database', not 'IT processes are "
        "flawed'). It must make the risk MORE likely. Not hypotheses: generalisations of a held one (weak processes, poor monitoring, "
        "a broader trend), restatements of a held one (give its verdict instead), rejected claims (never restate them), and ordinary "
        "explanations (normal, routine, benign: those are baselines).\n"
        "  - Every claim (new or revised) states ONE mechanism. No hedged lists ('e.g. temperature, humidity or contamination'), no "
        "'X or Y': pick the mechanism the evidence shows; a different one is a separate hypothesis.\n"
        "  - If `pointed` is true (the analyst pointed the episode at this risk) and the episode bears on the risk at all, return at "
        "least one supporting held or new hypothesis: a weak one is fine, standing will say how weak.\n"
        "- For every hypothesis the episode supports (given, held or new): rationale 2-4 sentences from the episode via the hypothesis "
        "to the risk (here you may use the evidence); reach and fit rate THIS EPISODE ALONE, as if the listed evidence did not exist "
        "(the evidence decides whether the episode bears on the hypothesis, never how strongly it scores: combining evidence is the "
        "standing judgment's job): reach: how far along the path to the risk the episode sits (0 = remote precursor, 1 = the risk "
        "itself); fit: how well the episode supports the hypothesis on its own (raise for concealment, repetition, odd hours, tooling, explicit intent, "
        "a named target; lower for a named benign role or stated purpose); missing: information that would change the assessment; "
        "baselines: the ordinary explanations of the same episode, weighed against this hypothesis: assess every baseline listed for "
        "it (assessed: baseline_id, fit, ordinary_outcome, rationale) and add new ones (new: claim, ordinary_outcome, rationale, fit) "
        "only for ordinary explanations none of them covers, at most `max_new_baselines` (1 if it has listed baselines). Baseline fit: "
        "how well the episode supports that ordinary explanation (a named benign role or stated purpose raises it; concealment, "
        "repetition, anonymity, tooling lower it). Use precedents (confirmed past trajectories) for consistency.\n"
        "- For a hypothesis the episode weakens: fit = how consistent the episode still is with it (low), rationale 1-2 sentences, "
        "no baselines.\n"
        "- relevant: false only if the episode bears on nothing here.\n"
        "Schema: {\"relevant\":bool,\"signal\":{...}|null,"
        "\"given\":{\"stance\":\"supports\"|\"weakens\"|\"unrelated\",\"fit\":0-1,\"reach\":0-1,\"rationale\":str,\"missing\":[str],\"entities\":[str],\"baselines\":B}|null,"
        "\"stored\":[{\"hypothesis_id\":str,\"stance\":\"supports\"|\"weakens\"|\"unrelated\",\"confidence\":0-1,\"revised_claim\":str|null,\"fit\":0-1,\"reach\":0-1,"
        "\"rationale\":str,\"missing\":[str],\"baselines\":B}],"
        "\"new\":[{\"claim\":str,\"entities\":[str],\"fit\":0-1,\"reach\":0-1,\"rationale\":str,\"missing\":[str],\"baselines\":B}]} where "
        "B = {\"assessed\":[{\"baseline_id\":str,\"fit\":0-1,\"ordinary_outcome\":str,\"rationale\":str}],"
        "\"new\":[{\"claim\":str,\"ordinary_outcome\":str,\"rationale\":str,\"fit\":0-1}]}", AssessOut, "reasoning"),
    "hypothesis.sharpen": Prompt("hypothesis.sharpen", "1", ANALYST,
        "A hypothesis has gathered several supporting facts; the last one is new. Read the claim and the facts together and ask: do "
        "they show the mechanism more specifically than the claim states (who or what is involved, through which route, by which "
        "means), or show that it works differently? A vague claim ('a chemical leak near Mill Road', 'a vulnerability in the "
        "software') usually can be sharpened once several facts are in. If so, write the sharper claim: ONE mechanism, naming the "
        "actors, systems, routes and means the facts show, and nothing the facts do not support. If the claim already says what the "
        "facts show, return null. why: one sentence naming the facts that make it sharper. "
        "Schema: {\"sharper_claim\":str|null,\"why\":str}", SharpenOut, "reasoning"),
    "standing.judge": Prompt("standing.judge", "9", ANALYST,
        "For each hypothesis listed, judge its standing: how well it is supported, against its ordinary explanations (baselines), by "
        "ALL the trajectories attached to it, taken together. Give two values: `before`, from its `trajectories` alone, and `value`, "
        "with `this` (the new episode's trajectory) added. Each trajectory has a status and a weight: confirmed by the analyst "
        "(weight 1), proposed by the system and not yet confirmed (lower weight; less certain), rejected by the analyst (weight 0: "
        "ignore it as support). Combine them as an analyst would: several individually weak trajectories that point the same way can "
        "make the hypothesis likely; two that report the same fact count once; a trajectory whose best explanation is ordinary adds "
        "little; a trajectory with stance \"weakens\" is counter-evidence and lowers the hypothesis in proportion to its weight; "
        "evidence that rules out the hypothesis's mechanism lowers it even if each item alone looks neutral. A trajectory with stance "
        "\"neutral\" bore on nothing on its own; it adds nothing by itself, but read it together with the others: facts that are "
        "neutral alone can jointly support or rule out the hypothesis (a recorded weight and a stated limit). " + _SCALE + " "
        "With no trajectories, before is 0. When a hypothesis comes with `before` already set (the standing last reported for it), "
        "return that value as before and judge value as the change from it.  A supporting `this` keeps or raises the value; a `this` with stance \"weakens\" must lower "
        "it below before, in proportion to what it shows: evidence that rules out the mechanism, or removes the very condition the "
        "hypothesis depends on, can halve it or more; evidence of a competing cause, or a detail that merely fits it less well, "
        "lowers it modestly (a few hundredths to a tenth). The hypotheses are alternatives for the same risk: judge each on its own "
        "evidence, consistently with the others. rationale: 2-4 sentences on the change. "
        "Schema: {\"standings\":[{\"key\":str,\"before\":0-1,\"value\":0-1,\"rationale\":str}]}", StandingsOut, "reasoning"),
}
