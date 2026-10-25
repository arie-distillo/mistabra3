from __future__ import annotations
import json, re
from .store import Store, new_id, now, dumps
from .models import Risk, Radar


def _risk(r) -> Risk:
    return Risk(id=r["id"], text=r["text"], polarity=r["polarity"], open=bool(r["open"]), status=r["status"],
                entities=json.loads(r["entities_json"]), origin=r["origin"], created_at=r["created_at"])


def _radar(r) -> Radar:
    return Radar(id=r["id"], risk_id=r["risk_id"], query=r["query"], enabled=bool(r["enabled"]), origin=r["origin"], created_at=r["created_at"])


class RisksAPI:
    def __init__(self, store: Store): self.store = store

    def create(self, text: str, actor: str, *, polarity: str = "negative", open: bool = False, entities=()) -> Risk:
        rid = new_id()
        ents = list(entities) or re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b", text)
        with self.store.tx(actor) as tx:
            tx.execute("INSERT INTO risk(id,text,polarity,open,status,entities_json,origin,created_at) VALUES(?,?,?,?,?,?,?,?)",
                       (rid, text, polarity, int(open), "active", dumps(ents), "analyst", now()))
            tx.audit("risk.created", "risk", rid, after={"text": text, "polarity": polarity, "open": open})
        return self.get(rid)

    def get(self, risk_id: str) -> Risk:
        r = self.store.one("SELECT * FROM risk WHERE id=?", (risk_id,))
        if r is None: raise KeyError(risk_id)
        return _risk(r)

    def list(self, status: str | None = "active") -> list[Risk]:
        q, p = "SELECT * FROM risk", ()
        if status: q += " WHERE status=?"; p = (status,)
        return [_risk(r) for r in self.store.all(q + " ORDER BY created_at", p)]

    def update(self, risk_id: str, actor: str, **fields) -> Risk:
        allowed = {"text", "open", "status", "entities"}
        bad = set(fields) - allowed
        if bad: raise ValueError(f"cannot update {bad}")
        with self.store.tx(actor) as tx:
            before = tx.one("SELECT * FROM risk WHERE id=?", (risk_id,))
            if before is None: raise KeyError(risk_id)
            if "text" in fields: tx.execute("UPDATE risk SET text=? WHERE id=?", (fields["text"], risk_id))
            if "open" in fields: tx.execute("UPDATE risk SET open=? WHERE id=?", (int(fields["open"]), risk_id))
            if "status" in fields: tx.execute("UPDATE risk SET status=? WHERE id=?", (fields["status"], risk_id))
            if "entities" in fields: tx.execute("UPDATE risk SET entities_json=? WHERE id=?", (dumps(list(fields["entities"])), risk_id))
            tx.audit("risk.updated", "risk", risk_id, before={k: before[k if k != "entities" else "entities_json"] for k in fields}, after=fields)
        return self.get(risk_id)


class RadarsAPI:
    def __init__(self, store: Store, gw, engine, context):
        self.store, self.gw, self.engine, self.context = store, gw, engine, context

    def create(self, risk_id: str, query: str, actor: str, *, origin: str = "analyst", enabled: bool = True) -> Radar:
        rid = new_id()
        with self.store.tx(actor) as tx:
            if tx.one("SELECT 1 FROM risk WHERE id=?", (risk_id,)) is None: raise KeyError(risk_id)
            tx.execute("INSERT INTO radar(id,risk_id,query,enabled,origin,created_at) VALUES(?,?,?,?,?,?)", (rid, risk_id, query, int(enabled), origin, now()))
            tx.audit("radar.created", "radar", rid, after={"risk_id": risk_id, "query": query, "origin": origin})
        return self.get(rid)

    def derive(self, risk_id: str, actor: str, *, enabled: bool = True) -> list[Radar]:
        risk = self.store.one("SELECT * FROM risk WHERE id=?", (risk_id,))
        if risk is None: raise KeyError(risk_id)
        out, call_id = self.gw.call("radar.derive", {"risk_text": risk["text"], "context": self.engine.context_snippets(risk["text"], 6)}, subject=("risk", risk_id))
        made = []
        for d in out.radars:
            q = d.query.strip()
            if q and not self.store.one("SELECT 1 FROM radar WHERE risk_id=? AND query=?", (risk_id, q)):
                made.append(self.create(risk_id, q, actor, origin="system", enabled=enabled))
        return made

    def get(self, radar_id: str) -> Radar:
        r = self.store.one("SELECT * FROM radar WHERE id=?", (radar_id,))
        if r is None: raise KeyError(radar_id)
        return _radar(r)

    def list(self, risk_id: str | None = None) -> list[Radar]:
        q, p = "SELECT * FROM radar", ()
        if risk_id: q += " WHERE risk_id=?"; p = (risk_id,)
        return [_radar(r) for r in self.store.all(q + " ORDER BY created_at", p)]

    def update(self, radar_id: str, actor: str, *, enabled: bool | None = None, query: str | None = None) -> Radar:
        with self.store.tx(actor) as tx:
            before = tx.one("SELECT * FROM radar WHERE id=?", (radar_id,))
            if before is None: raise KeyError(radar_id)
            if enabled is not None: tx.execute("UPDATE radar SET enabled=? WHERE id=?", (int(enabled), radar_id))
            if query is not None: tx.execute("UPDATE radar SET query=? WHERE id=?", (query, radar_id))
            tx.audit("radar.updated", "radar", radar_id, before={"enabled": bool(before["enabled"]), "query": before["query"]}, after={"enabled": enabled, "query": query})
        return self.get(radar_id)
