"""Versioned prompt templates. Every rendered prompt starts with a TASK line and
carries its input as a JSON block, so outputs are reproducible and the fake
provider can parse the same input the real model sees."""
from __future__ import annotations
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    id: str
    version: str
    system: str
    instructions: str

    def render(self, variables: dict) -> tuple[str, str]:
        user = f"TASK: {self.id}\n\n{self.instructions}\n\nINPUT_JSON:\n{json.dumps(variables, ensure_ascii=False, indent=1)}\n\nRespond with a single JSON object and nothing else."
        return self.system, user


JSON_ONLY = "You are a precise information-extraction component inside an analyst tool. Output only valid JSON matching the requested schema."

PROMPTS: dict[str, Prompt] = {
    "ontology.extract": Prompt(
        "ontology.extract", "1",
        JSON_ONLY,
        "From the document chunk, extract domain-ontology elements. objectives: risks (polarity 'negative') or goals "
        "(polarity 'positive') the organization states. concepts: domain vocabulary terms with aliases and an optional parent term. "
        "entity_types: classes of resolvable individuals (person, organization, location, vehicle, ...). sources: named information "
        "sources with a credibility estimate 0-1 if stated. Every element must quote the supporting span in 'evidence'. "
        "Schema: {\"objectives\":[{\"label\",\"polarity\",\"description\",\"concerns_entity_types\":[],\"evidence\"}],"
        "\"concepts\":[{\"label\",\"aliases\":[],\"parent_label\",\"evidence\"}],\"entity_types\":[{\"label\",\"attributes\":[],\"evidence\"}],"
        "\"sources\":[{\"label\",\"credibility\",\"evidence\"}]}"),
    "ontology.infer_signals": Prompt(
        "ontology.infer_signals", "1",
        "You are a senior analyst designing early-warning indicators. Output only valid JSON.",
        "For the objective, infer the signal taxonomy: the precursor signal types that would typically be observed before the "
        "objective materializes, ordered toward it. For each signal type give proximity_to_harm in [0,1] (1 = the objective itself), "
        "precursor_to (labels of later signal types), and observables: concrete forms of behaviour visible in text that can serve this "
        "signal type, each with a base-rate prior in [0,1] (how often that observable, when seen, actually serves this signal type). "
        "Schema: {\"signal_types\":[{\"label\",\"description\",\"proximity_to_harm\",\"precursor_to\":[],"
        "\"observables\":[{\"label\",\"description\",\"prior\"}]}]}"),
    "extract.episode": Prompt(
        "extract.episode", "1",
        JSON_ONLY,
        "From the episode text extract (a) observations: one or more short factual summaries of what was observed, each with the "
        "ontology observable ids it plausibly instantiates (from the provided list; empty if none) and modifiers (time_of_day, "
        "repetition, concealment, tooling, proximity, target_specificity) when present; (b) entity mentions: verbatim surface strings "
        "of persons, organizations, locations, vehicles, institutions, digital identifiers, with a type from the provided entity types "
        "and any attributes. Schema: {\"observations\":[{\"text\",\"observable_ids\":[],\"modifiers\":{},\"mention_surfaces\":[]}],"
        "\"mentions\":[{\"surface\",\"entity_type\",\"attributes\":{}}]}"),
    "entities.adjudicate": Prompt(
        "entities.adjudicate", "1",
        JSON_ONLY,
        "Decide whether the mention refers to one of the candidate entities. Be conservative: choose 'link' only when the evidence "
        "supports identity, 'new' when it is clearly a different individual, 'unresolved' when it cannot be determined. "
        "Schema: {\"decision\":\"link|new|unresolved\",\"entity_id\":null,\"confidence\":0.0,\"rationale\":\"\"}"),
}
