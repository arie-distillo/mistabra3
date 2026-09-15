"""Draft review (accept/edit/reject/merge), publish (validate + immutable version), import, load."""
from __future__ import annotations
import json
from pathlib import Path
from ..store import Store, new_id, now
from .model import Ontology, slug
from .builder import OntologyBuilder

KINDS = ("objectives", "signal_types", "observables", "concepts", "entity_types", "sources")


class OntologyAPI:
    def __init__(self, store: Store, builder: OntologyBuilder, data_dir: Path, callbacks=None):
        self.store, self.builder, self.dir, self.cb = store, builder, Path(data_dir) / "ontology", callbacks
        self.dir.mkdir(parents=True, exist_ok=True)

    # ---- drafts ----
    def build_draft(self, ontology_id: str, docs: list[tuple[str, str]], actor: str) -> str:
        base = self.latest_version(ontology_id)
        draft = self.builder.build(ontology_id, docs)
        did = new_id()
        with self.store.tx(actor) as tx:
            tx.execute("INSERT INTO ontology_draft(id,ontology_id,created_at,created_by,status,base_version) VALUES(?,?,?,?,?,?)",
                       (did, ontology_id, now(), actor, "in_review", base.version if base else None))
            if base:  # carry existing elements into the draft as accepted
                for kind in KINDS:
                    for e in getattr(base, kind):
                        body = e.model_dump(); body.pop("id")
                        self._put_element(tx, did, kind, e.id, body, "accepted", "existing")
            for kind in KINDS:
                for eid, body in draft[kind].items():
                    if tx.one("SELECT 1 FROM ontology_draft_element WHERE draft_id=? AND kind=? AND element_id=?", (did, kind, eid)):
                        continue
                    self._put_element(tx, did, kind, eid, body, "pending", "extracted" if body.get("status") == "confirmed" else "inferred")
            tx.audit("ontology.draft_built", "ontology_draft", did, after={"ontology_id": ontology_id, "elements": sum(len(v) for v in draft.values())})
        return did

    def import_taxonomy(self, draft_id: str, taxonomy: dict, actor: str, origin: str = "imported") -> int:
        """Add elements from an external taxonomy file (same JSON shape, ids optional) to a draft as pending."""
        n = 0
        with self.store.tx(actor) as tx:
            for kind in KINDS:
                for body in taxonomy.get(kind, []):
                    body = dict(body); eid = body.pop("id", None) or slug(kind[:3], body["label"])
                    body["status"] = "imported"; body.setdefault("provenance", [])
                    if not tx.one("SELECT 1 FROM ontology_draft_element WHERE draft_id=? AND kind=? AND element_id=?", (draft_id, kind, eid)):
                        self._put_element(tx, draft_id, kind, eid, body, "pending", origin); n += 1
            tx.audit("ontology.imported", "ontology_draft", draft_id, after={"elements": n})
        return n

    @staticmethod
    def _put_element(tx, draft_id, kind, eid, body, status, origin):
        tx.execute("INSERT INTO ontology_draft_element(id,draft_id,kind,element_id,body_json,status,origin) VALUES(?,?,?,?,?,?,?)",
                   (new_id(), draft_id, kind, eid, json.dumps(body, ensure_ascii=False), status, origin))

    def draft_elements(self, draft_id: str, status: str | None = None) -> list[dict]:
        q, p = "SELECT * FROM ontology_draft_element WHERE draft_id=?", [draft_id]
        if status: q += " AND status=?"; p.append(status)
        return [{**dict(r), "body": json.loads(r["body_json"])} for r in self.store.all(q + " ORDER BY kind, element_id", p)]

    def review(self, draft_id: str, kind: str, element_id: str, action: str, actor: str, patch: dict | None = None, into: str | None = None) -> None:
        with self.store.tx(actor) as tx:
            row = tx.one("SELECT * FROM ontology_draft_element WHERE draft_id=? AND kind=? AND element_id=?", (draft_id, kind, element_id))
            if row is None: raise KeyError(f"{kind}/{element_id} not in draft")
            before = {"status": row["status"], "body": json.loads(row["body_json"])}
            body = dict(before["body"])
            if action == "accept": status = "accepted"; body["status"] = "confirmed" if body.get("status") != "imported" else "imported"
            elif action == "reject": status = "rejected"
            elif action == "edit":
                body.update(patch or {}); body["status"] = "confirmed"; status = "accepted"
            elif action == "merge":
                if not into: raise ValueError("merge requires 'into'")
                tgt = tx.one("SELECT * FROM ontology_draft_element WHERE draft_id=? AND kind=? AND element_id=?", (draft_id, kind, into))
                if tgt is None: raise KeyError(f"merge target {into} not in draft")
                tb = json.loads(tgt["body_json"])
                tb["provenance"] = tb.get("provenance", []) + body.get("provenance", [])
                if "aliases" in tb: tb["aliases"] = sorted(set(tb.get("aliases", [])) | set(body.get("aliases", [])) | {body["label"]})
                tx.execute("UPDATE ontology_draft_element SET body_json=? WHERE id=?", (json.dumps(tb, ensure_ascii=False), tgt["id"]))
                status = "merged"
            else: raise ValueError(action)
            tx.execute("UPDATE ontology_draft_element SET status=?, body_json=?, merged_into=? WHERE id=?",
                       (status, json.dumps(body, ensure_ascii=False), into if action == "merge" else None, row["id"]))
            tx.audit(f"ontology.element_{action}", "ontology_draft_element", row["id"], before=before, after={"status": status, "body": body})

    def accept_all(self, draft_id: str, actor: str, kinds: tuple[str, ...] = KINDS) -> int:
        n = 0
        for e in self.draft_elements(draft_id, "pending"):
            if e["kind"] in kinds:
                self.review(draft_id, e["kind"], e["element_id"], "accept", actor); n += 1
        return n

    def publish(self, draft_id: str, actor: str) -> str:
        d = self.store.one("SELECT * FROM ontology_draft WHERE id=?", (draft_id,))
        if d is None: raise KeyError(draft_id)
        accepted = self.draft_elements(draft_id, "accepted")
        accepted_ids = {(e["kind"], e["element_id"]) for e in accepted}
        groups = {k: [] for k in KINDS}
        for e in accepted:
            b = dict(e["body"]); b["id"] = e["element_id"]
            # drop dangling references to elements that were rejected/merged
            if e["kind"] == "signal_types":
                b["precursor_to"] = [p for p in b.get("precursor_to", []) if ("signal_types", p) in accepted_ids]
                b["objective_ids"] = [o for o in b.get("objective_ids", []) if ("objectives", o) in accepted_ids]
            if e["kind"] == "observables":
                b["can_serve"] = [c for c in b.get("can_serve", []) if ("signal_types", c["signal_type_id"]) in accepted_ids]
            if e["kind"] == "concepts" and b.get("parent_id") and ("concepts", b["parent_id"]) not in accepted_ids:
                b["parent_id"] = None
            if e["kind"] == "objectives":
                b["concerns_entity_types"] = [t for t in b.get("concerns_entity_types", []) if ("entity_types", t) in accepted_ids]
            groups[e["kind"]].append(b)
        latest = self.latest_version(d["ontology_id"])
        version = (latest.version + 1) if latest else 1
        onto = Ontology(ontology_id=d["ontology_id"], version=version, published_at=now(), published_by=actor, **groups)  # validates refs
        path = self.dir / d["ontology_id"] / f"v{version}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        vid = new_id()
        with self.store.tx(actor) as tx:
            path.write_text(onto.model_dump_json(indent=1))
            tx.execute("INSERT INTO ontology_version(id,ontology_id,version,path,published_at,published_by,element_count) VALUES(?,?,?,?,?,?,?)",
                       (vid, d["ontology_id"], version, str(path), onto.published_at, actor, onto.element_count()))
            tx.execute("UPDATE ontology_draft SET status='published', published_version_id=? WHERE id=?", (vid, draft_id))
            tx.audit("ontology.published", "ontology_version", vid, after={"ontology_id": d["ontology_id"], "version": version, "elements": onto.element_count()})
        if self.cb: self.cb.emit("ontology.published", {"ontology_id": d["ontology_id"], "version": version, "version_id": vid})
        return vid

    # ---- versions ----
    def latest_version(self, ontology_id: str) -> Ontology | None:
        r = self.store.one("SELECT * FROM ontology_version WHERE ontology_id=? ORDER BY version DESC LIMIT 1", (ontology_id,))
        return self.load(r["id"]) if r else None

    def latest_version_id(self, ontology_id: str) -> str | None:
        r = self.store.one("SELECT id FROM ontology_version WHERE ontology_id=? ORDER BY version DESC LIMIT 1", (ontology_id,))
        return r["id"] if r else None

    def load(self, version_id: str) -> Ontology:
        r = self.store.one("SELECT * FROM ontology_version WHERE id=?", (version_id,))
        if r is None: raise KeyError(version_id)
        return Ontology.model_validate_json(Path(r["path"]).read_text())

    def versions(self, ontology_id: str) -> list[dict]:
        return [dict(r) for r in self.store.all("SELECT * FROM ontology_version WHERE ontology_id=? ORDER BY version", (ontology_id,))]

    def export(self, version_id: str) -> str:
        return self.load(version_id).model_dump_json(indent=1)
