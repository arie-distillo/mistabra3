"""Embedded vector index, one table per object_type, persisted under <data_dir>/vectors.

Backends: "numpy" (default) — persistent brute-force cosine search, adequate for Phase-1
volumes (tens of thousands of vectors) and deterministic; "lancedb" — optional, for
hybrid search and larger stores. LanceDB showed intermittent interpreter-shutdown
crashes in some environments, which is why it is opt-in rather than default."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

try:
    import lancedb  # type: ignore
except Exception:  # pragma: no cover
    lancedb = None


class VectorIndex:
    def __init__(self, data_dir: Path | str, embedder, backend: str = "numpy"):
        self.dir = Path(data_dir) / "vectors"; self.dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        if backend == "lancedb" and lancedb is None:
            raise RuntimeError("vector_backend=lancedb requires the 'lancedb' package (pip install mistabra[vectors])")
        self.backend = backend
        self._db = lancedb.connect(str(self.dir / "lancedb")) if backend == "lancedb" else None
        self._np: dict[str, dict] = {}

    def upsert(self, object_type: str, items: list[tuple[str, str, dict]]) -> None:
        if not items: return
        vecs = self.embedder.embed([t for _, t, _ in items])
        rows = [{"id": i, "text": t, "vector": v, "meta": json.dumps(m)} for (i, t, m), v in zip(items, vecs)]
        if self.backend == "lancedb":
            names = self._tables()
            if object_type in names:
                tbl = self._db.open_table(object_type)
                ids = [r["id"] for r in rows]
                tbl.delete(f"id IN ({', '.join(repr(i) for i in ids)})")
                tbl.add(rows)
            else:
                self._db.create_table(object_type, data=rows)
        else:
            store = self._load_np(object_type)
            for r in rows: store[r["id"]] = r
            self._save_np(object_type)

    def search(self, object_type: str, query: str, k: int = 10) -> list[dict]:
        qv = self.embedder.embed([query])[0]
        if self.backend == "lancedb":
            if object_type not in self._tables(): return []
            res = self._db.open_table(object_type).search(qv).limit(k).to_list()
            return [{"id": r["id"], "text": r["text"], "meta": json.loads(r["meta"]), "score": float(1 - r.get("_distance", 0.0) / 2)} for r in res]
        store = self._load_np(object_type)
        if not store: return []
        ids = list(store); M = np.array([store[i]["vector"] for i in ids], dtype=np.float32)
        sims = M @ np.array(qv, dtype=np.float32)
        order = np.argsort(-sims)[:k]
        return [{"id": ids[j], "text": store[ids[j]]["text"], "meta": json.loads(store[ids[j]]["meta"]), "score": float(sims[j])} for j in order]

    def _tables(self) -> list[str]:
        if self._db is None: return []
        return list(self._db.list_tables().tables) if hasattr(self._db, "list_tables") and hasattr(self._db.list_tables(), "tables") else list(self._db.list_tables())

    def close(self) -> None:
        """Release the embedded DB handle explicitly (teardown order matters for native libs)."""
        self._db = None

    def count(self, object_type: str) -> int:
        if self.backend == "lancedb":
            return self._db.open_table(object_type).count_rows() if object_type in self._tables() else 0
        return len(self._load_np(object_type))

    def _load_np(self, t):
        if t not in self._np:
            p = self.dir / f"{t}.json"
            self._np[t] = json.loads(p.read_text()) if p.exists() else {}
        return self._np[t]

    def _save_np(self, t):
        (self.dir / f"{t}.json").write_text(json.dumps(self._np[t]))
