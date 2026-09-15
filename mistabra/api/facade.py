"""Public entry point. A host imports `Mistabra`, points it at a data directory, and
uses the facade objects; identity is passed as `actor` on every write."""
from __future__ import annotations
import atexit, json, weakref
from collections import defaultdict
from pathlib import Path
from typing import Callable
from ..config import Settings
from ..store import Store
from ..jobs import Jobs
from ..llm import Gateway, make_provider, HashingEmbedder
from ..vectors import VectorIndex
from ..ontology import OntologyBuilder, OntologyAPI
from ..ingest import SourcesAPI, EpisodesAPI
from ..extract import Extractor
from ..entities import EntitiesAPI, SimpleResolver


class Callbacks:
    def __init__(self): self._subs = defaultdict(list)
    def on(self, event: str, fn: Callable[[dict], None]) -> None: self._subs[event].append(fn)
    def emit(self, event: str, payload: dict) -> None:
        for fn in self._subs[event] + self._subs["*"]:
            fn({"event": event, **payload})


class AuditAPI:
    def __init__(self, store: Store): self.store = store
    def events(self, object_type: str | None = None, object_id: str | None = None, actor: str | None = None, limit: int = 200) -> list[dict]:
        q, p = "SELECT * FROM audit_event WHERE 1=1", []
        if object_type: q += " AND object_type=?"; p.append(object_type)
        if object_id: q += " AND object_id=?"; p.append(object_id)
        if actor: q += " AND actor=?"; p.append(actor)
        rows = self.store.all(q + " ORDER BY rowid DESC LIMIT ?", [*p, limit])
        return [{**dict(r), "before": json.loads(r["before_json"]) if r["before_json"] else None, "after": json.loads(r["after_json"]) if r["after_json"] else None} for r in rows]
    def llm_calls(self, limit: int = 100) -> list[dict]:
        return [dict(r) for r in self.store.all("SELECT id,ts,task,tier,prompt_id,prompt_version,model,provider,parsed_ok,cached,latency_ms,tokens_in,tokens_out,error FROM llm_call ORDER BY ts DESC LIMIT ?", (limit,))]
    def verify_chain(self) -> bool: return self.store.verify_audit_chain()


class Mistabra:
    def __init__(self, data_dir: str | Path, settings: Settings | None = None, provider=None, embedder=None,
                 resolver=None, ontology_id: str = "default", log: Callable[[str], None] | None = None):
        self.settings = settings or Settings.from_env(data_dir)
        self.settings.data_dir = Path(data_dir)
        self.log = log or (lambda m: None)
        self.store = Store(self.settings.data_dir)
        self.callbacks = Callbacks()
        self.provider = provider or make_provider(self.settings)
        self.gateway = Gateway(self.store, self.settings, self.provider)
        self.embedder = embedder or HashingEmbedder(self.settings.embed_dim)
        self.vectors = VectorIndex(self.settings.data_dir, self.embedder, self.settings.vector_backend)
        self.jobs = Jobs(self.store, self.log)
        self.ontology_id = ontology_id
        self.ontology = OntologyAPI(self.store, OntologyBuilder(self.gateway, self.settings.chunk_chars, self.log), self.settings.data_dir, self.callbacks)
        self.sources = SourcesAPI(self.store)
        self.episodes = EpisodesAPI(self.store, self.jobs, self.callbacks)
        self.entities = EntitiesAPI(self.store, resolver or SimpleResolver(self.gateway), self.callbacks, self.vectors)
        self.extractor = Extractor(self.store, self.gateway, self.current_ontology, self.vectors, self.jobs)
        self.audit = AuditAPI(self.store)
        self.jobs.register("extract", lambda p: self.extractor.run(p["episode_id"]))
        self.jobs.register("resolve", lambda p: self.entities.resolve_mentions(p["mention_ids"]))
        atexit.register(Mistabra._close_ref, weakref.ref(self))

    @staticmethod
    def _close_ref(ref):
        m = ref()
        if m is not None: m.close()

    def current_ontology(self):
        return self.ontology.latest_version(self.ontology_id)

    def process(self) -> dict:
        """Run all queued jobs (extraction, resolution) to completion."""
        return self.jobs.run_pending()

    def stats(self) -> dict:
        c = lambda t: self.store.one(f"SELECT COUNT(*) AS n FROM {t}")["n"]
        return {"episodes": c("episode"), "observations": c("observation"), "mentions": c("entity_mention"), "entities": c("entity"),
                "unresolved_mentions": self.store.one("SELECT COUNT(*) AS n FROM entity_mention WHERE resolution='unresolved'")["n"],
                "ontology_versions": c("ontology_version"), "audit_events": c("audit_event"), "llm_calls": c("llm_call"),
                "llm": dict(self.gateway.stats), "provider": self.provider.name, "vector_backend": self.vectors.backend}

    def close(self) -> None:
        """Release embedded resources in a safe order: vectors first, then the store."""
        if getattr(self, "_closed", False): return
        self._closed = True
        try: self.vectors.close()
        finally: self.store.close()
