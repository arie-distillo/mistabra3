"""Episode -> observations + entity mentions (operational model), stored with pipeline version."""
from __future__ import annotations
import json
from pydantic import BaseModel, Field
from ..llm import Gateway
from ..store import Store, new_id, now

PIPELINE_VERSION = "extract-1"


class _Obs(BaseModel):
    text: str; observable_ids: list[str] = Field(default_factory=list)
    modifiers: dict = Field(default_factory=dict); mention_surfaces: list[str] = Field(default_factory=list)
class _Mention(BaseModel):
    surface: str; entity_type: str; attributes: dict = Field(default_factory=dict)
class Extraction(BaseModel):
    observations: list[_Obs] = Field(default_factory=list); mentions: list[_Mention] = Field(default_factory=list)


class Extractor:
    def __init__(self, store: Store, gateway: Gateway, ontology_provider, vectors=None, jobs=None):
        self.store, self.gw, self.onto, self.vectors, self.jobs = store, gateway, ontology_provider, vectors, jobs

    def run(self, episode_id: str, actor: str = "system") -> dict:
        ep = self.store.one("SELECT * FROM episode WHERE id=?", (episode_id,))
        if ep is None: raise KeyError(episode_id)
        onto = self.onto()
        observables = [{"id": o.id, "label": o.label, "description": o.description} for o in onto.observables] if onto else []
        entity_types = [{"id": e.id.split(".", 1)[-1], "label": e.label} for e in onto.entity_types] if onto else []
        if not entity_types:
            entity_types = [{"id": t, "label": t} for t in ("person", "organization", "location", "institution", "vehicle")]
        try:
            x, call_id = self.gw.call("extract.episode", "operational",
                                      {"episode_id": episode_id, "episode_text": ep["text"], "observables": observables, "entity_types": entity_types}, Extraction)
        except Exception as e:
            with self.store.tx(actor) as tx:
                tx.execute("UPDATE episode SET extraction_status='failed' WHERE id=?", (episode_id,))
                tx.audit("episode.extraction_failed", "episode", episode_id, after={"error": str(e)[:300]})
            raise
        valid_obs = {o["id"] for o in observables}
        valid_types = {e["id"] for e in entity_types}
        obs_ids, mention_ids = [], []
        with self.store.tx(actor) as tx:
            tx.execute("DELETE FROM observation WHERE episode_id=?", (episode_id,))
            tx.execute("DELETE FROM entity_mention WHERE episode_id=?", (episode_id,))
            surface_to_obs = {}
            for i, o in enumerate(x.observations):
                oid = new_id(); obs_ids.append(oid)
                tx.execute("INSERT INTO observation(id,episode_id,ordinal,text,observable_ids_json,modifiers_json,created_at,pipeline_version,llm_call_id)"
                           " VALUES(?,?,?,?,?,?,?,?,?)", (oid, episode_id, i, o.text, json.dumps([v for v in o.observable_ids if v in valid_obs]),
                                                       json.dumps(o.modifiers), now(), PIPELINE_VERSION, call_id))
                for s in o.mention_surfaces: surface_to_obs.setdefault(s, oid)
            for m in x.mentions:
                mid = new_id(); mention_ids.append(mid)
                et = m.entity_type if m.entity_type in valid_types else (next(iter(valid_types)) if valid_types else m.entity_type)
                tx.execute("INSERT INTO entity_mention(id,episode_id,observation_id,surface,entity_type,attributes_json,created_at) VALUES(?,?,?,?,?,?,?)",
                           (mid, episode_id, surface_to_obs.get(m.surface), m.surface, et, json.dumps(m.attributes), now()))
            tx.execute("UPDATE episode SET extraction_status='done' WHERE id=?", (episode_id,))
            tx.audit("episode.extracted", "episode", episode_id, after={"observations": len(obs_ids), "mentions": len(mention_ids)}, llm_call_id=call_id)
        if self.vectors is not None:
            self.vectors.upsert("observation", [(oid, o.text, {"episode_id": episode_id}) for oid, o in zip(obs_ids, x.observations)])
            with self.store.tx(actor) as tx:
                for oid in obs_ids:
                    tx.execute("INSERT OR REPLACE INTO vector_record(object_type,object_id,embedder,indexed_at) VALUES(?,?,?,?)", ("observation", oid, self.vectors.embedder.name, now()))
        if self.jobs is not None and mention_ids:
            self.jobs.enqueue("resolve", {"episode_id": episode_id, "mention_ids": mention_ids})
        return {"observations": obs_ids, "mentions": mention_ids}
