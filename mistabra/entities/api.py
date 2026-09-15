"""Entity store: resolve mentions (job handler), merge/split/re-link (audited, reversible), links."""
from __future__ import annotations
import json
from ..store import Store, new_id, now
from .resolver import Decision, normalize_name, canonical_name


class EntitiesAPI:
    def __init__(self, store: Store, resolver, callbacks=None, vectors=None):
        self.store, self.resolver, self.cb, self.vectors = store, resolver, callbacks, vectors

    # ---- resolution ----
    def resolve_mentions(self, mention_ids: list[str], actor: str = "system") -> dict:
        counts = {"link": 0, "new": 0, "unresolved": 0}
        for mid in mention_ids:
            m = self.store.one("SELECT * FROM entity_mention WHERE id=?", (mid,))
            if m is None: continue
            ep = self.store.one("SELECT text FROM episode WHERE id=?", (m["episode_id"],))
            cands = self._candidates(m["entity_type"], m["surface"])
            d: Decision = self.resolver.resolve({"surface": m["surface"], "entity_type": m["entity_type"], "attributes": json.loads(m["attributes_json"]),
                                                 "context": ep["text"][:600] if ep else ""}, cands)
            with self.store.tx(actor) as tx:
                if d.outcome == "link":
                    eid = d.entity_id
                    self._add_alias(tx, eid, m["surface"])
                elif d.outcome == "new":
                    eid = new_id()
                    cname = canonical_name(m["surface"])
                    tx.execute("INSERT INTO entity(id,entity_type,canonical_name,aliases_json,attributes_json,created_at) VALUES(?,?,?,?,?,?)",
                               (eid, m["entity_type"], cname, json.dumps([m["surface"]] if m["surface"] != cname else []), m["attributes_json"], now()))
                    tx.audit("entity.created", "entity", eid, after={"type": m["entity_type"], "name": cname, "from_mention": mid})
                else:
                    eid = None
                tx.execute("UPDATE entity_mention SET entity_id=?, resolution=?, resolution_score=?, resolution_rationale=? WHERE id=?",
                           (eid, d.outcome, d.score, d.rationale, mid))
                if eid:
                    tx.execute("INSERT OR IGNORE INTO entity_link(entity_id,object_type,object_id,role) VALUES(?,?,?,?)", (eid, "episode", m["episode_id"], "mentioned"))
                    if m["observation_id"]:
                        tx.execute("INSERT OR IGNORE INTO entity_link(entity_id,object_type,object_id,role) VALUES(?,?,?,?)", (eid, "observation", m["observation_id"], "mentioned"))
                tx.audit("mention.resolved", "entity_mention", mid, after={"outcome": d.outcome, "entity_id": eid, "score": d.score, "rationale": d.rationale})
            counts[d.outcome] += 1
        return counts

    def _candidates(self, entity_type: str, surface: str, limit: int = 50) -> list[dict]:
        rows = self.store.all("SELECT * FROM entity WHERE entity_type=? AND status='active'", (entity_type,))
        cands = [self._row(r) for r in rows]
        # cheap prefilter: share a token with the surface, else keep a bounded sample
        toks = set(normalize_name(surface).split())
        hit = [c for c in cands if toks & set(normalize_name(" ".join([c["canonical_name"], *c["aliases"]])).split())]
        return (hit or cands)[:limit]

    @staticmethod
    def _add_alias(tx, eid, surface):
        r = tx.one("SELECT canonical_name, aliases_json FROM entity WHERE id=?", (eid,))
        if r is None: return
        aliases = json.loads(r["aliases_json"])
        if surface != r["canonical_name"] and surface not in aliases:
            aliases.append(surface)
            tx.execute("UPDATE entity SET aliases_json=? WHERE id=?", (json.dumps(aliases), eid))

    # ---- corrections ----
    def merge(self, keep_id: str, drop_id: str, actor: str) -> None:
        """Merge drop into keep. Mentions are re-linked, never rewritten; reversible via audit."""
        with self.store.tx(actor) as tx:
            k, d = tx.one("SELECT * FROM entity WHERE id=?", (keep_id,)), tx.one("SELECT * FROM entity WHERE id=?", (drop_id,))
            if k is None or d is None: raise KeyError("entity not found")
            before = {"keep": self._row(k), "drop": self._row(d)}
            aliases = sorted(set(json.loads(k["aliases_json"])) | set(json.loads(d["aliases_json"])) | {d["canonical_name"]} - {k["canonical_name"]})
            tx.execute("UPDATE entity SET aliases_json=? WHERE id=?", (json.dumps(aliases), keep_id))
            tx.execute("UPDATE entity SET status='merged', merged_into=? WHERE id=?", (keep_id, drop_id))
            tx.execute("UPDATE entity_mention SET entity_id=? WHERE entity_id=?", (keep_id, drop_id))
            for l in tx.all("SELECT * FROM entity_link WHERE entity_id=?", (drop_id,)):
                tx.execute("INSERT OR IGNORE INTO entity_link(entity_id,object_type,object_id,role) VALUES(?,?,?,?)", (keep_id, l["object_type"], l["object_id"], l["role"]))
            tx.execute("DELETE FROM entity_link WHERE entity_id=?", (drop_id,))
            tx.audit("entity.merged", "entity", keep_id, before=before, after={"merged_from": drop_id, "aliases": aliases})
        if self.cb: self.cb.emit("entity.merged", {"keep": keep_id, "drop": drop_id})

    def split(self, entity_id: str, mention_ids: list[str], new_name: str, actor: str) -> str:
        new_eid = new_id()
        with self.store.tx(actor) as tx:
            e = tx.one("SELECT * FROM entity WHERE id=?", (entity_id,))
            if e is None: raise KeyError(entity_id)
            tx.execute("INSERT INTO entity(id,entity_type,canonical_name,aliases_json,attributes_json,created_at) VALUES(?,?,?,?,?,?)",
                       (new_eid, e["entity_type"], new_name, json.dumps([]), "{}", now()))
            for mid in mention_ids:
                m = tx.one("SELECT * FROM entity_mention WHERE id=? AND entity_id=?", (mid, entity_id))
                if m is None: raise KeyError(f"mention {mid} not on entity {entity_id}")
                tx.execute("UPDATE entity_mention SET entity_id=?, resolution='link', resolution_rationale=? WHERE id=?", (new_eid, f"split by {actor}", mid))
                tx.execute("INSERT OR IGNORE INTO entity_link(entity_id,object_type,object_id,role) VALUES(?,?,?,?)", (new_eid, "episode", m["episode_id"], "mentioned"))
            tx.audit("entity.split", "entity", entity_id, after={"new_entity": new_eid, "mentions": mention_ids, "name": new_name})
        return new_eid

    def relink(self, mention_id: str, entity_id: str | None, actor: str) -> None:
        with self.store.tx(actor) as tx:
            m = tx.one("SELECT * FROM entity_mention WHERE id=?", (mention_id,))
            if m is None: raise KeyError(mention_id)
            tx.execute("UPDATE entity_mention SET entity_id=?, resolution=?, resolution_rationale=? WHERE id=?",
                       (entity_id, "link" if entity_id else "unresolved", f"relinked by {actor}", mention_id))
            if entity_id:
                tx.execute("INSERT OR IGNORE INTO entity_link(entity_id,object_type,object_id,role) VALUES(?,?,?,?)", (entity_id, "episode", m["episode_id"], "mentioned"))
            tx.audit("mention.relinked", "entity_mention", mention_id, before={"entity_id": m["entity_id"]}, after={"entity_id": entity_id})

    def link_object(self, entity_id: str, object_type: str, object_id: str, role: str, actor: str) -> None:
        with self.store.tx(actor) as tx:
            tx.execute("INSERT OR IGNORE INTO entity_link(entity_id,object_type,object_id,role) VALUES(?,?,?,?)", (entity_id, object_type, object_id, role))
            tx.audit("entity.linked", "entity", entity_id, after={"object_type": object_type, "object_id": object_id, "role": role})

    # ---- reads ----
    @staticmethod
    def _row(r) -> dict:
        return {**dict(r), "aliases": json.loads(r["aliases_json"]), "attributes": json.loads(r["attributes_json"])}

    def get(self, entity_id: str) -> dict | None:
        r = self.store.one("SELECT * FROM entity WHERE id=?", (entity_id,)); return self._row(r) if r else None

    def list(self, entity_type: str | None = None, status: str = "active") -> list[dict]:
        q, p = "SELECT * FROM entity WHERE status=?", [status]
        if entity_type: q += " AND entity_type=?"; p.append(entity_type)
        return [self._row(r) for r in self.store.all(q + " ORDER BY entity_type, canonical_name", p)]

    def find(self, name: str, entity_type: str | None = None) -> list[dict]:
        n = normalize_name(name)
        return [e for e in self.list(entity_type) if n in (normalize_name(e["canonical_name"]), *(normalize_name(a) for a in e["aliases"]))]

    def mentions_of(self, entity_id: str) -> list[dict]:
        return [dict(r) for r in self.store.all("SELECT * FROM entity_mention WHERE entity_id=? ORDER BY created_at", (entity_id,))]

    def unresolved(self) -> list[dict]:
        return [dict(r) for r in self.store.all("SELECT * FROM entity_mention WHERE resolution='unresolved' ORDER BY created_at")]

    def links(self, entity_id: str) -> list[dict]:
        return [dict(r) for r in self.store.all("SELECT * FROM entity_link WHERE entity_id=?", (entity_id,))]

    def entities_for(self, object_type: str, object_id: str) -> list[dict]:
        return [self._row(r) for r in self.store.all("SELECT e.* FROM entity e JOIN entity_link l ON l.entity_id=e.id WHERE l.object_type=? AND l.object_id=? AND e.status='active'", (object_type, object_id))]

    def graph(self):
        """Entity-object link graph as networkx (built on demand; small at Phase-1 scale)."""
        import networkx as nx
        g = nx.Graph()
        for l in self.store.all("SELECT * FROM entity_link"):
            g.add_node(l["entity_id"], kind="entity"); g.add_node(l["object_id"], kind=l["object_type"])
            g.add_edge(l["entity_id"], l["object_id"], role=l["role"])
        return g
