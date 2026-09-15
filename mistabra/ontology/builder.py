"""Ontology builder: documents -> chunks -> extraction (operational model) -> merge ->
signal-taxonomy inference per objective (reasoning model) -> draft with provenance."""
from __future__ import annotations
from typing import Callable
from pydantic import BaseModel, Field
from ..llm import Gateway
from .model import slug, Provenance


class _XObjective(BaseModel):
    label: str; polarity: str = "negative"; description: str | None = None
    concerns_entity_types: list[str] = Field(default_factory=list); evidence: str | None = None
class _XConcept(BaseModel):
    label: str; aliases: list[str] = Field(default_factory=list); parent_label: str | None = None; evidence: str | None = None
class _XEntityType(BaseModel):
    label: str; attributes: list[str] = Field(default_factory=list); evidence: str | None = None
class _XSource(BaseModel):
    label: str; credibility: float | None = None; evidence: str | None = None
class Extracted(BaseModel):
    objectives: list[_XObjective] = Field(default_factory=list); concepts: list[_XConcept] = Field(default_factory=list)
    entity_types: list[_XEntityType] = Field(default_factory=list); sources: list[_XSource] = Field(default_factory=list)

class _IObservable(BaseModel):
    label: str; description: str | None = None; prior: float = 0.1
class _ISignal(BaseModel):
    label: str; description: str | None = None; proximity_to_harm: float = 0.5
    precursor_to: list[str] = Field(default_factory=list); observables: list[_IObservable] = Field(default_factory=list)
class Inferred(BaseModel):
    signal_types: list[_ISignal] = Field(default_factory=list)


def chunk_text(text: str, max_chars: int) -> list[str]:
    paras, chunks, cur = [p.strip() for p in text.split("\n\n") if p.strip()], [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > max_chars:
            chunks.append(cur); cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks or [text]


def _norm(s: str) -> str:
    return " ".join(s.lower().replace("-", " ").split())


class OntologyBuilder:
    def __init__(self, gateway: Gateway, chunk_chars: int = 1400, log: Callable[[str], None] | None = None):
        self.gw, self.chunk_chars, self.log = gateway, chunk_chars, (log or (lambda m: None))

    def build(self, ontology_id: str, docs: list[tuple[str, str]], existing: dict | None = None) -> dict:
        """Returns a draft dict {kind: {element_id: body}} with provenance/status on every element."""
        draft = {k: {} for k in ("objectives", "concepts", "entity_types", "sources", "signal_types", "observables")}
        prov_of = {}
        chunks = [(name, i, c) for name, text in docs for i, c in enumerate(chunk_text(text, self.chunk_chars))]
        self.log(f"ontology build: {len(docs)} docs, {len(chunks)} chunks")
        for name, i, c in chunks:
            x, call_id = self.gw.call("ontology.extract", "operational", {"doc": name, "chunk_index": i, "chunk_text": c}, Extracted)
            for o in x.objectives:
                self._merge(draft["objectives"], slug("obj", o.label), {"label": o.label, "polarity": o.polarity if o.polarity in ("negative", "positive") else "negative",
                            "description": o.description, "concerns_entity_types": []}, Provenance(doc=name, chunk=i, evidence=o.evidence), o.concerns_entity_types)
            for cpt in x.concepts:
                self._merge(draft["concepts"], slug("con", cpt.label), {"label": cpt.label, "aliases": cpt.aliases, "parent_label": cpt.parent_label}, Provenance(doc=name, chunk=i, evidence=cpt.evidence))
            for et in x.entity_types:
                self._merge(draft["entity_types"], slug("et", et.label), {"label": et.label.lower(), "attributes": et.attributes}, Provenance(doc=name, chunk=i, evidence=et.evidence))
            for s in x.sources:
                self._merge(draft["sources"], slug("src", s.label), {"label": s.label, "credibility": s.credibility if s.credibility is not None else 0.5}, Provenance(doc=name, chunk=i, evidence=s.evidence))
        # resolve concept parents and objective entity-type refs by label
        by_label = {_norm(c["label"]): cid for cid, c in draft["concepts"].items()}
        for c in draft["concepts"].values():
            pl = c.pop("parent_label", None)
            c["parent_id"] = by_label.get(_norm(pl)) if pl else None
        et_by_label = {_norm(e["label"]): eid for eid, e in draft["entity_types"].items()}
        for o in draft["objectives"].values():
            o["concerns_entity_types"] = [et_by_label[_norm(t)] for t in o.pop("_et_labels", []) if _norm(t) in et_by_label]
        # infer signal taxonomy per objective (reasoning model)
        for oid, o in draft["objectives"].items():
            self.log(f"infer signals for {oid}")
            inf, call_id = self.gw.call("ontology.infer_signals", "reasoning",
                                        {"objective_id": oid, "objective_label": o["label"], "polarity": o["polarity"], "description": o.get("description")}, Inferred)
            st_ids = {_norm(s.label): slug("sig", s.label) for s in inf.signal_types}
            for s in inf.signal_types:
                sid = st_ids[_norm(s.label)]
                body = draft["signal_types"].setdefault(sid, {"label": s.label, "description": s.description, "objective_ids": [], "proximity_to_harm": s.proximity_to_harm,
                                                              "precursor_to": [], "status": "inferred", "provenance": []})
                if oid not in body["objective_ids"]: body["objective_ids"].append(oid)
                for p in s.precursor_to:
                    pid = st_ids.get(_norm(p))
                    if pid and pid not in body["precursor_to"] and pid != sid: body["precursor_to"].append(pid)
                for ob in s.observables:
                    obid = slug("obs", ob.label)
                    ob_body = draft["observables"].setdefault(obid, {"label": ob.label, "description": ob.description, "can_serve": [], "status": "inferred", "provenance": []})
                    if not any(c["signal_type_id"] == sid for c in ob_body["can_serve"]):
                        ob_body["can_serve"].append({"signal_type_id": sid, "prior": max(0.0, min(1.0, ob.prior))})
        return draft

    @staticmethod
    def _merge(bucket: dict, eid: str, body: dict, prov: Provenance, et_labels: list[str] | None = None):
        cur = bucket.setdefault(eid, {**body, "status": "confirmed", "provenance": []})
        cur["provenance"].append(prov.model_dump())
        if "aliases" in body:
            cur["aliases"] = sorted(set(cur.get("aliases", [])) | set(body["aliases"]))
        if body.get("description") and not cur.get("description"):
            cur["description"] = body["description"]
        if et_labels:
            cur.setdefault("_et_labels", []).extend(et_labels)
