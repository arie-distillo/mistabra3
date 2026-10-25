"""Versioned prompts. Each renders to (system, user) with the input as a JSON block,
and declares the pydantic schema the output must satisfy."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pydantic import BaseModel, Field


# ---- output schemas ----
class RadarDraft(BaseModel):
    query: str
class RadarsOut(BaseModel):
    radars: list[RadarDraft] = Field(default_factory=list)

class ContentOut(BaseModel):
    actor: str | None = None; behaviour: str | None = None; target: str | None = None; means: str | None = None
    time: str | None = None; modifiers: list[str] = Field(default_factory=list); entities: list[str] = Field(default_factory=list); summary: str = ""
class RadarReadOut(BaseModel):
    relevant: bool; confidence: float = 0.5; content: ContentOut | None = None; missing: list[str] = Field(default_factory=list)

class LambdaCheckOut(BaseModel):
    relevant: bool; confidence: float = 0.5; why: str = ""; content: ContentOut | None = None

class RiskTrajectoryOut(BaseModel):
    hypothesis_id: str | None = None          # reuse an existing hypothesis
    claim: str | None = None                  # or state a new one
    rationale: str; reach: float; fit: float
    entities: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    predicts: list[str] = Field(default_factory=list)   # observations expected next if the claim is true

class SetwiseOut(BaseModel):
    distribution: dict[str, float]
    rationale: str = ""

class BaselineOut(BaseModel):
    ordinary_outcome: str; claim: str; rationale: str; fit: float
class BaselinesOut(BaseModel):
    baselines: list[BaselineOut] = Field(default_factory=list)


@dataclass(frozen=True)
class Prompt:
    id: str; version: str; system: str; instructions: str; schema: type[BaseModel]; tier: str   # tier: reasoning | fast

    def render(self, variables: dict) -> tuple[str, str]:
        user = (f"TASK: {self.id}\n\n{self.instructions}\n\nINPUT_JSON:\n{json.dumps(variables, ensure_ascii=False, indent=1)}\n\n"
                "Respond with a single JSON object matching the schema and nothing else.")
        return self.system, user


ANALYST = ("You are a careful intelligence analyst's co-pilot. You reason about whether an observed event could be a step "
           "toward a stated risk, and you always consider the ordinary explanations too. Output only valid JSON.")

PROMPTS: dict[str, Prompt] = {
    "radar.derive": Prompt("radar.derive", "1", ANALYST,
        "Given a risk and domain context, write 3-6 radars. A radar is a query that decides whether an episode (a short text report) "
        "could be relevant to the risk, and what to extract from it. Each query is one or two sentences, concrete, covering a distinct "
        "kind of precursor (attention to targets, expressed intent, acquisition of means, coordination, rehearsal...). "
        "Schema: {\"radars\":[{\"query\":str}]}", RadarsOut, "reasoning"),
    "radar.read": Prompt("radar.read", "1", ANALYST,
        "Decide whether the episode is relevant to the radar's query for the given risk. If relevant, extract the content in a structured way: "
        "actor, behaviour, target, means, time, modifiers (e.g. concealment, repetition, night, tooling, anonymity), entities (names of people, "
        "places, organisations mentioned), summary (one sentence). List what information is missing that would matter. Be literal: do not infer "
        "intent. Schema: {\"relevant\":bool,\"confidence\":0-1,\"content\":{...}|null,\"missing\":[str]}", RadarReadOut, "fast"),
    "signal.lambda_check": Prompt("signal.lambda_check", "1", ANALYST,
        "No radar matched this episode. Using the risk statement and context only, judge whether the episode could nevertheless be a step toward "
        "the risk — something a thoughtful analyst would want to look at even though no query anticipated it (a Lambda signal). Be conservative: "
        "most episodes are not relevant to most risks. If relevant, extract content as in radar reading and say why in one sentence. "
        "Schema: {\"relevant\":bool,\"confidence\":0-1,\"why\":str,\"content\":{...}|null}", LambdaCheckOut, "fast"),
    "trajectory.risk": Prompt("trajectory.risk", "1", ANALYST,
        "Write the trajectory from this signal to the risk: the single hypothesis (a claim about the world, e.g. 'the house on Elm Street is a "
        "target') that would connect the observed content to the risk, and a rationale in 2-4 sentences telling the story from the episode to the "
        "risk. If one of the existing hypotheses already expresses the claim, reuse its id instead of writing a new claim; if force_new is true, the "
        "existing set has been judged unable to explain this signal, so write a new claim that does. "
        "reach: how far along the path to the risk this signal sits (0 = remote precursor, 1 = the risk itself). fit: how well the episode "
        "supports the hypothesis (0 = barely, 1 = strongly; raise for concealment, repetition, odd hours, tooling, explicit intent, named target). "
        "Use precedents (confirmed past trajectories) for consistency. missing: information that would change the assessment. "
        "predicts: 2-4 short observations you would expect to appear next if the claim is true (used to notice when expected evidence fails to arrive). "
        "Schema: {\"hypothesis_id\":str|null,\"claim\":str|null,\"rationale\":str,\"reach\":0-1,\"fit\":0-1,\"entities\":[str],\"missing\":[str],\"predicts\":[str]}",
        RiskTrajectoryOut, "reasoning"),
    "decision.setwise": Prompt("decision.setwise", "1", ANALYST,
        "Score the whole hypothesis set for this risk at once. Input: the new episode; context; the set — risk hypotheses (kind 'risk') and ordinary "
        "explanations (kind 'baseline'), each with what it predicts and the signals already attached to it (the state); the abducer's proposal for this "
        "signal (a trajectory to one hypothesis, and baseline trajectories); the prior distribution if any. Return a probability distribution over "
        "every id in the set plus none_id ('none of these': the episode is explained by none of them). Judge each hypothesis given EVERYTHING attached "
        "to it, not the episode alone: two individually weak signals on the same hypothesis can make it likely. Put mass on 'none' when neither the "
        "risk hypotheses nor the ordinary explanations account for the episode. Probabilities must sum to 1. rationale: 2-4 sentences. "
        "Schema: {\"distribution\":{id:0-1,...},\"rationale\":str}", SetwiseOut, "reasoning"),
    "trajectory.baseline": Prompt("trajectory.baseline", "1", ANALYST,
        "Write the ordinary explanations of the same episode: up to N baseline trajectories, each ending in an ordinary outcome (e.g. 'property "
        "listing', 'tourist album', 'lawful hunting'). For each: the ordinary outcome, the claim that would make it true, a one-to-two sentence "
        "rationale, and fit: how well the episode supports THIS ordinary explanation given everything stated (0 = implausible, 1 = fully "
        "explained; a named benign role or stated purpose raises it; concealment, repetition, anonymity, tooling lower it). "
        "Schema: {\"baselines\":[{\"ordinary_outcome\":str,\"claim\":str,\"rationale\":str,\"fit\":0-1}]}", BaselinesOut, "reasoning"),
}
