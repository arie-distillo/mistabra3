from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from ..store import Store, new_id, now


@dataclass
class OccurrenceTime:
    kind: str = "unknown"            # exact | range | unknown
    start: str | None = None         # ISO-8601
    end: str | None = None

    @classmethod
    def parse(cls, v) -> "OccurrenceTime":
        if v is None: return cls()
        if isinstance(v, cls): return v
        if isinstance(v, str): return cls("exact", v, v)
        if isinstance(v, dict):
            if "start" in v and "end" in v: return cls("range", v["start"], v["end"])
            if "exact" in v: return cls("exact", v["exact"], v["exact"])
        raise ValueError(f"bad occurrence_time {v!r}")


class SourcesAPI:
    def __init__(self, store: Store): self.store = store

    def register(self, name: str, type: str, credibility: float, actor: str, config: dict | None = None) -> str:
        sid = new_id()
        with self.store.tx(actor) as tx:
            tx.execute("INSERT INTO source(id,name,type,credibility,config_json,created_at) VALUES(?,?,?,?,?,?)",
                       (sid, name, type, credibility, json.dumps(config or {}), now()))
            tx.audit("source.registered", "source", sid, after={"name": name, "type": type, "credibility": credibility})
        return sid

    def set_credibility(self, source_id: str, credibility: float, actor: str) -> None:
        with self.store.tx(actor) as tx:
            r = tx.one("SELECT credibility FROM source WHERE id=?", (source_id,))
            if r is None: raise KeyError(source_id)
            tx.execute("UPDATE source SET credibility=? WHERE id=?", (credibility, source_id))
            tx.audit("source.credibility_changed", "source", source_id, before={"credibility": r["credibility"]}, after={"credibility": credibility})

    def get(self, source_id: str) -> dict | None:
        r = self.store.one("SELECT * FROM source WHERE id=?", (source_id,)); return dict(r) if r else None

    def by_name(self, name: str) -> dict | None:
        r = self.store.one("SELECT * FROM source WHERE name=?", (name,)); return dict(r) if r else None

    def list(self) -> list[dict]:
        return [dict(r) for r in self.store.all("SELECT * FROM source ORDER BY name")]


class EpisodesAPI:
    def __init__(self, store: Store, jobs, callbacks=None):
        self.store, self.jobs, self.cb = store, jobs, callbacks

    def ingest(self, source_id: str, text: str, actor: str, occurrence_time=None, metadata: dict | None = None,
               enqueue_extraction: bool = True) -> dict:
        """Idempotent on (source, content hash). Returns {episode_id, created: bool}."""
        h = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        occ = OccurrenceTime.parse(occurrence_time)
        existing = self.store.one("SELECT id FROM episode WHERE source_id=? AND content_hash=?", (source_id, h))
        if existing: return {"episode_id": existing["id"], "created": False}
        eid = new_id()
        with self.store.tx(actor) as tx:
            tx.execute("INSERT INTO episode(id,source_id,content_hash,text,occurrence_kind,occurrence_start,occurrence_end,ingested_at,metadata_json)"
                       " VALUES(?,?,?,?,?,?,?,?,?)", (eid, source_id, h, text, occ.kind, occ.start, occ.end, now(), json.dumps(metadata or {})))
            tx.audit("episode.ingested", "episode", eid, after={"source_id": source_id, "occurrence": occ.__dict__, "chars": len(text)})
        if enqueue_extraction: self.jobs.enqueue("extract", {"episode_id": eid})
        if self.cb: self.cb.emit("episode.ingested", {"episode_id": eid})
        return {"episode_id": eid, "created": True}

    def ingest_many(self, source_id: str, records, actor: str) -> dict:
        res = {"created": 0, "duplicates": 0, "ids": []}
        for r in records:
            out = self.ingest(source_id, r["text"], actor, r.get("occurrence_time"), r.get("metadata"))
            res["created" if out["created"] else "duplicates"] += 1; res["ids"].append(out["episode_id"])
        return res

    def get(self, episode_id: str) -> dict | None:
        r = self.store.one("SELECT * FROM episode WHERE id=?", (episode_id,)); return dict(r) if r else None

    def observations(self, episode_id: str) -> list[dict]:
        return [{**dict(r), "observable_ids": json.loads(r["observable_ids_json"]), "modifiers": json.loads(r["modifiers_json"])}
                for r in self.store.all("SELECT * FROM observation WHERE episode_id=? ORDER BY ordinal", (episode_id,))]

    def mentions(self, episode_id: str) -> list[dict]:
        return [{**dict(r), "attributes": json.loads(r["attributes_json"])} for r in self.store.all("SELECT * FROM entity_mention WHERE episode_id=? ORDER BY created_at, surface", (episode_id,))]

    def list(self, limit: int = 100) -> list[dict]:
        return [dict(r) for r in self.store.all("SELECT * FROM episode ORDER BY ingested_at DESC LIMIT ?", (limit,))]
