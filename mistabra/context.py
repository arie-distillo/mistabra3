"""Context: memory, not schema. Lexical search is enough for the PoC."""
from __future__ import annotations
import json, re
from pathlib import Path
from .store import Store, new_id, now, dumps
from .models import ContextItem

_TOKEN = re.compile(r"\w+")


def _row(r) -> ContextItem:
    return ContextItem(id=r["id"], kind=r["kind"], title=r["title"], text=r["text"], tags=json.loads(r["tags_json"]), origin=r["origin"], created_at=r["created_at"])


def search_items(store: Store, query: str, k: int = 10, kinds=None) -> list[dict]:
    q = {t.lower() for t in _TOKEN.findall(query) if len(t) > 2}
    rows = store.all("SELECT * FROM context_item" + (" WHERE kind IN (%s)" % ",".join("?" * len(kinds)) if kinds else ""), tuple(kinds or ()))
    scored = []
    for r in rows:
        toks = _TOKEN.findall((r["title"] or "") + " " + r["text"])
        tl = [t.lower() for t in toks]
        hits = sum(1 for t in tl if t in q)
        if hits:
            scored.append((hits / (1 + len(tl) ** 0.5), r))
    scored.sort(key=lambda x: -x[0])
    return [dict(r) for _, r in scored[:k]]


class ContextAPI:
    def __init__(self, store: Store): self.store = store

    def create(self, text: str, actor: str, *, title: str | None = None, kind: str = "document", tags=()) -> ContextItem:
        cid = new_id()
        with self.store.tx(actor) as tx:
            tx.execute("INSERT INTO context_item(id,kind,title,text,tags_json,origin,created_at) VALUES(?,?,?,?,?,?,?)",
                       (cid, kind, title, text, dumps(list(tags)), "analyst", now()))
            tx.audit("context.created", "context_item", cid, after={"kind": kind, "title": title, "chars": len(text)})
        return self.get(cid)

    def create_file(self, path, actor: str, **kw) -> ContextItem:
        p = Path(path)
        return self.create(p.read_text(encoding="utf-8"), actor, title=kw.pop("title", p.name), **kw)

    def get(self, item_id: str) -> ContextItem:
        r = self.store.one("SELECT * FROM context_item WHERE id=?", (item_id,))
        if r is None: raise KeyError(item_id)
        return _row(r)

    def list(self, kind: str | None = None) -> list[ContextItem]:
        q, p = "SELECT * FROM context_item", ()
        if kind: q += " WHERE kind=?"; p = (kind,)
        return [_row(r) for r in self.store.all(q + " ORDER BY created_at", p)]

    def search(self, query: str, k: int = 10, kinds=None) -> list[ContextItem]:
        return [_row(r) for r in [self.store.one("SELECT * FROM context_item WHERE id=?", (d["id"],)) for d in search_items(self.store, query, k, kinds)]]
