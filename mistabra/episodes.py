from __future__ import annotations
import hashlib, json
from .store import Store, new_id, now, dumps
from .models import Source, Episode


def parse_occurred(v):
    if v is None: return ("unknown", None, None)
    if isinstance(v, str): return ("exact", v, v)
    if isinstance(v, (tuple, list)) and len(v) == 2: return ("range", v[0], v[1])
    if isinstance(v, dict):
        if "start" in v and "end" in v: return ("range", v["start"], v["end"])
        if "exact" in v: return ("exact", v["exact"], v["exact"])
    raise ValueError(f"bad occurred value {v!r}")


def _episode(r) -> Episode:
    return Episode(id=r["id"], source_id=r["source_id"], text=r["text"], occurred_kind=r["occurred_kind"], occurred_start=r["occurred_start"],
                   occurred_end=r["occurred_end"], metadata=json.loads(r["metadata_json"]), created_at=r["created_at"])


class SourcesAPI:
    def __init__(self, store: Store): self.store = store

    def create(self, name: str, actor: str, *, credibility: float = 0.5, type: str = "manual") -> Source:
        existing = self.store.one("SELECT * FROM source WHERE name=?", (name,))
        if existing: return Source(id=existing["id"], name=existing["name"], type=existing["type"], credibility=existing["credibility"])
        sid = new_id()
        with self.store.tx(actor) as tx:
            tx.execute("INSERT INTO source(id,name,type,credibility,created_at) VALUES(?,?,?,?,?)", (sid, name, type, credibility, now()))
            tx.audit("source.created", "source", sid, after={"name": name, "credibility": credibility})
        return Source(id=sid, name=name, type=type, credibility=credibility)

    def list(self) -> list[Source]:
        return [Source(id=r["id"], name=r["name"], type=r["type"], credibility=r["credibility"]) for r in self.store.all("SELECT * FROM source ORDER BY name")]

    def update(self, source_id: str, actor: str, *, credibility: float) -> Source:
        with self.store.tx(actor) as tx:
            r = tx.one("SELECT * FROM source WHERE id=?", (source_id,))
            if r is None: raise KeyError(source_id)
            tx.execute("UPDATE source SET credibility=? WHERE id=?", (credibility, source_id))
            tx.audit("source.updated", "source", source_id, before={"credibility": r["credibility"]}, after={"credibility": credibility})
        r = self.store.one("SELECT * FROM source WHERE id=?", (source_id,))
        return Source(id=r["id"], name=r["name"], type=r["type"], credibility=r["credibility"])


class EpisodesAPI:
    def __init__(self, store: Store, engine): self.store, self.engine = store, engine

    def create(self, text: str, actor: str, *, source: str, occurred=None, metadata: dict | None = None) -> Episode:
        """Idempotent on (source, content hash). Consequences run inline: radar reads, Lambda-signal check, signals, trajectories."""
        src = self.store.one("SELECT * FROM source WHERE id=? OR name=?", (source, source))
        if src is None: raise KeyError(f"source {source!r} not registered")
        h = hashlib.sha256(text.strip().encode()).hexdigest()
        existing = self.store.one("SELECT * FROM episode WHERE source_id=? AND content_hash=?", (src["id"], h))
        if existing: return _episode(existing)
        kind, start, end = parse_occurred(occurred)
        eid = new_id()
        with self.store.tx(actor) as tx:
            tx.execute("INSERT INTO episode(id,source_id,content_hash,text,occurred_kind,occurred_start,occurred_end,metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                       (eid, src["id"], h, text, kind, start, end, dumps(metadata or {}), now()))
            tx.audit("episode.created", "episode", eid, after={"source": src["name"], "occurred": [kind, start, end], "chars": len(text)})
        self.engine.process_episode(eid, actor="system")
        return self.get(eid)

    def create_many(self, records, actor: str) -> list[Episode]:
        return [self.create(r["text"], actor, source=r["source"], occurred=r.get("occurred"), metadata=r.get("metadata")) for r in records]

    def get(self, episode_id: str) -> Episode:
        r = self.store.one("SELECT * FROM episode WHERE id=?", (episode_id,))
        if r is None: raise KeyError(episode_id)
        return _episode(r)

    def list(self, *, source: str | None = None, since: str | None = None, limit: int = 100) -> list[Episode]:
        q, p = "SELECT e.* FROM episode e JOIN source s ON s.id=e.source_id WHERE 1=1", []
        if source: q += " AND (s.id=? OR s.name=?)"; p += [source, source]
        if since: q += " AND e.created_at>=?"; p.append(since)
        return [_episode(r) for r in self.store.all(q + " ORDER BY e.created_at LIMIT ?", [*p, limit])]
