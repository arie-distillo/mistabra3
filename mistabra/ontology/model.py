"""Custom JSON ontology schema (Pydantic). Two behavioural layers — observables and
signal types — plus concepts, entity types, sources. Baseline explanations are
deliberately absent: they are world knowledge generated at runtime, not domain model."""
from __future__ import annotations
import re
from typing import Literal
from pydantic import BaseModel, Field, model_validator


def slug(kind: str, label: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:60]
    return f"{kind}.{s}"


class Provenance(BaseModel):
    doc: str
    chunk: int
    evidence: str | None = None


class Element(BaseModel):
    id: str
    label: str
    description: str | None = None
    status: Literal["inferred", "confirmed", "imported"] = "confirmed"
    provenance: list[Provenance] = Field(default_factory=list)


class Objective(Element):
    polarity: Literal["negative", "positive"] = "negative"
    concerns_entity_types: list[str] = Field(default_factory=list)


class CanServe(BaseModel):
    signal_type_id: str
    prior: float = Field(ge=0.0, le=1.0)


class Observable(Element):
    can_serve: list[CanServe] = Field(default_factory=list)


class SignalType(Element):
    objective_ids: list[str] = Field(default_factory=list)
    proximity_to_harm: float = Field(default=0.5, ge=0.0, le=1.0)
    precursor_to: list[str] = Field(default_factory=list)


class Concept(Element):
    parent_id: str | None = None
    aliases: list[str] = Field(default_factory=list)


class EntityType(Element):
    attributes: list[str] = Field(default_factory=list)


class SourceDef(Element):
    credibility: float = Field(default=0.5, ge=0.0, le=1.0)


class Ontology(BaseModel):
    schema_version: str = "1.0"
    ontology_id: str
    version: int = 0
    published_at: str | None = None
    published_by: str | None = None
    objectives: list[Objective] = Field(default_factory=list)
    signal_types: list[SignalType] = Field(default_factory=list)
    observables: list[Observable] = Field(default_factory=list)
    concepts: list[Concept] = Field(default_factory=list)
    entity_types: list[EntityType] = Field(default_factory=list)
    sources: list[SourceDef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_refs(self):
        ids = {e.id for group in (self.objectives, self.signal_types, self.observables, self.concepts, self.entity_types, self.sources) for e in group}
        if len(ids) != sum(len(g) for g in (self.objectives, self.signal_types, self.observables, self.concepts, self.entity_types, self.sources)):
            raise ValueError("duplicate element ids")
        st = {s.id for s in self.signal_types}
        et = {e.id for e in self.entity_types}
        for s in self.signal_types:
            for o in s.objective_ids:
                if o not in {x.id for x in self.objectives}: raise ValueError(f"{s.id}: unknown objective {o}")
            for p in s.precursor_to:
                if p not in st: raise ValueError(f"{s.id}: unknown precursor_to {p}")
        for o in self.observables:
            for c in o.can_serve:
                if c.signal_type_id not in st: raise ValueError(f"{o.id}: unknown signal type {c.signal_type_id}")
        for c in self.concepts:
            if c.parent_id and c.parent_id not in {x.id for x in self.concepts}: raise ValueError(f"{c.id}: unknown parent {c.parent_id}")
        for o in self.objectives:
            for t in o.concerns_entity_types:
                if t not in et: raise ValueError(f"{o.id}: unknown entity type {t}")
        return self

    # --- views ---
    def element_count(self) -> int:
        return sum(len(g) for g in (self.objectives, self.signal_types, self.observables, self.concepts, self.entity_types, self.sources))

    def induced_progression(self, objective_id: str) -> dict:
        """Hypothesis-induced subgraph: signal types precursor_to the objective (with proximity), and observables serving them."""
        sts = [s for s in self.signal_types if objective_id in s.objective_ids]
        st_ids = {s.id for s in sts}
        obs = [o for o in self.observables if any(c.signal_type_id in st_ids for c in o.can_serve)]
        return {"objective_id": objective_id,
                "signal_types": sorted(({"id": s.id, "label": s.label, "proximity_to_harm": s.proximity_to_harm, "precursor_to": s.precursor_to} for s in sts), key=lambda x: x["proximity_to_harm"]),
                "observables": [{"id": o.id, "label": o.label, "can_serve": [c.model_dump() for c in o.can_serve if c.signal_type_id in st_ids]} for o in obs]}

    def concept_index(self) -> dict[str, str]:
        """normalized label/alias -> concept id"""
        idx = {}
        for c in self.concepts:
            for name in [c.label, *c.aliases]:
                idx[name.strip().lower()] = c.id
        return idx
