"""Public entry point."""
from __future__ import annotations
import json
from pathlib import Path
from .settings import Settings
from .store import Store
from .llm import Gateway, make_provider
from .engine import Engine
from .context import ContextAPI
from .risks import RisksAPI, RadarsAPI
from .episodes import SourcesAPI, EpisodesAPI
from .signals import SignalsAPI, TrajectoriesAPI, HypothesesAPI


class AuditAPI:
    def __init__(self, store: Store): self.store = store

    def events(self, *, object_type=None, object_id=None, actor=None, since=None, limit=200) -> list[dict]:
        q, p = "SELECT * FROM audit WHERE 1=1", []
        for col, v in (("object_type", object_type), ("object_id", object_id), ("actor", actor)):
            if v: q += f" AND {col}=?"; p.append(v)
        if since: q += " AND ts>=?"; p.append(since)
        return [{**dict(r), "before": json.loads(r["before_json"]) if r["before_json"] else None, "after": json.loads(r["after_json"]) if r["after_json"] else None}
                for r in self.store.all(q + " ORDER BY seq DESC LIMIT ?", [*p, limit])]

    def trace(self, signal_id: str) -> dict:
        """Everything behind one signal: audit events on it and its trajectories, and the model calls that produced them."""
        s = self.store.one("SELECT * FROM signal WHERE id=?", (signal_id,))
        if s is None: raise KeyError(signal_id)
        tids = [r["id"] for r in self.store.all("SELECT id FROM trajectory WHERE signal_id=?", (signal_id,))]
        ids = [signal_id, *tids]
        events = [dict(r) for r in self.store.all(f"SELECT * FROM audit WHERE object_id IN ({','.join('?' * len(ids))}) ORDER BY seq", ids)]
        calls = [dict(r) for r in self.store.all("SELECT id,ts,task,prompt_version,provider,model,ok,cached,latency_ms,tokens_in,tokens_out,error FROM llm_call "
                                                 "WHERE (subject_type='signal' AND subject_id=?) OR (subject_type='episode' AND subject_id=?) ORDER BY ts", (signal_id, s["episode_id"]))]
        return {"signal_id": signal_id, "episode_id": s["episode_id"], "events": events, "llm_calls": calls}

    def verify(self) -> bool:
        return self.store.verify_audit()


class LearnAPI:
    def __init__(self, engine: Engine, store: Store): self.engine, self.store = engine, store

    def replay(self, signal_ids, actor: str, *, model: str | None = None, prompt_version: str | None = None) -> dict:
        """Re-reason the given signals with the current (or specified) model/prompts without persisting; diff against stored."""
        rows = []
        for sid in signal_ids:
            s = self.store.one("SELECT strength_system, metrics_json FROM signal WHERE id=?", (sid,))
            new = self.engine.reason(sid, actor, persist=False, model=model, prompt_version=prompt_version)
            rows.append({"signal_id": sid, "stored": s["strength_system"], "replayed": new["strength"],
                         "delta": None if s["strength_system"] is None else round(new["strength"] - s["strength_system"], 3),
                         "stored_metrics": json.loads(s["metrics_json"]) if s["metrics_json"] else None, "replayed_metrics": new["metrics"]})
        deltas = [abs(r["delta"]) for r in rows if r["delta"] is not None]
        return {"rows": rows, "mean_abs_delta": round(sum(deltas) / len(deltas), 3) if deltas else None, "model": model, "prompt_version": prompt_version}


class Mistabra:
    def __init__(self, data_dir: str | Path, settings: Settings | None = None, *, provider=None, log=None):
        self.settings = settings or Settings.from_env(data_dir)
        self.settings.data_dir = Path(data_dir)
        self.log = log or (lambda m: None)
        self.store = Store(self.settings.data_dir, blind=self.settings.blind)
        self.provider = provider or make_provider(self.settings)
        self.gateway = Gateway(self.store, self.settings, self.provider)
        self.engine = Engine(self.store, self.gateway, self.settings, self.log)
        self.context = ContextAPI(self.store)
        self.risks = RisksAPI(self.store)
        self.radars = RadarsAPI(self.store, self.gateway, self.engine, self.context)
        self.sources = SourcesAPI(self.store)
        self.episodes = EpisodesAPI(self.store, self.engine)
        self.signals = SignalsAPI(self.store, self.engine, self.settings)
        self.trajectories = TrajectoriesAPI(self.store)
        self.hypotheses = HypothesesAPI(self.store)
        self.audit = AuditAPI(self.store)
        self.learn = LearnAPI(self.engine, self.store)

    def stats(self) -> dict:
        c = self.store.count
        return {"context": c("context_item"), "risks": c("risk"), "radars": c("radar"), "episodes": c("episode"), "signals": c("signal"),
                "signals_lambda": self.store.one("SELECT COUNT(*) AS n FROM signal WHERE via='lambda'")["n"],
                "trajectories": c("trajectory"), "hypotheses": self.store.one("SELECT COUNT(*) AS n FROM hypothesis WHERE status!='ordinary'")["n"],
                "audit_events": c("audit"), "llm_calls": c("llm_call"), "llm": dict(self.gateway.stats),
                "provider": self.provider.name, "model": self.settings.model, "fast_model": self.settings.fast_model, "blind": self.settings.blind}

    def close(self):
        self.store.close()
