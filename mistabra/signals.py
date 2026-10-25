"""Signals, trajectories, hypotheses. The blind rule is applied in reads."""
from __future__ import annotations
import json
from .store import Store, new_id, now, dumps
from .models import Signal, SignalContent, Metrics, Trajectory, Hypothesis, Coverage, CoverageItem, PosteriorView


def _post(r) -> PosteriorView:
    return PosteriorView(decision=r["decision"], model=r["model"], prompt_version=r["prompt_version"], distribution=json.loads(r["distribution_json"]),
                         info_gain=r["info_gain"], strength=r["strength"], rationale=r["rationale"], created_at=r["created_at"])


def _traj(r) -> Trajectory:
    return Trajectory(id=r["id"], signal_id=r["signal_id"], episode_id=r["episode_id"], risk_id=r["risk_id"], radar_id=r["radar_id"],
                      hypothesis_id=r["hypothesis_id"], outcome=r["outcome"], ordinary_outcome=r["ordinary_outcome"], reach=r["reach"], fit=r["fit"],
                      rationale=r["rationale"], status=r["status"], origin=r["origin"], created_at=r["created_at"])


def _hyp(r) -> Hypothesis:
    return Hypothesis(id=r["id"], risk_id=r["risk_id"], claim=r["claim"], entities=json.loads(r["entities_json"]), status=r["status"], origin=r["origin"],
                      is_lambda=bool(r["is_lambda"]), predicts=json.loads(r["predicts_json"]), created_at=r["created_at"])


class SignalsAPI:
    def __init__(self, store: Store, engine, settings): self.store, self.engine, self.settings = store, engine, settings

    def _build(self, r, *, unblind: bool = False) -> Signal:
        hidden = self.settings.blind and r["strength_analyst"] is None and not unblind
        trs = [_traj(t) for t in self.store.all("SELECT * FROM trajectory WHERE signal_id=? ORDER BY outcome DESC, fit DESC", (r["id"],))]
        risk_t = next((t for t in trs if t.outcome == "risk"), None)
        pr = self.store.one("SELECT * FROM posterior WHERE signal_id=? ORDER BY created_at DESC LIMIT 1", (r["id"],))
        return Signal(id=r["id"], episode_id=r["episode_id"], risk_id=r["risk_id"], origin=r["origin"], via=r["via"], radar_id=r["radar_id"],
                      content=SignalContent.model_validate_json(r["content_json"]), note=r["note"],
                      strength_system=None if hidden else r["strength_system"], strength_analyst=r["strength_analyst"],
                      metrics=None if hidden or not r["metrics_json"] else Metrics(**json.loads(r["metrics_json"])),
                      missing=json.loads(r["missing_json"]), status=r["status"],
                      trajectory=None if hidden else risk_t, baselines=[] if hidden else [t for t in trs if t.outcome == "ordinary"],
                      posterior=None if hidden or pr is None else _post(pr),
                      versions=json.loads(r["versions_json"]), created_at=r["created_at"])

    def create(self, episode_id: str, risk_id: str, actor: str, *, content: dict | None = None, note: str | None = None) -> Signal:
        """The analyst raises a hunch. Trajectories follow as a consequence."""
        ep = self.store.one("SELECT * FROM episode WHERE id=?", (episode_id,))
        risk = self.store.one("SELECT * FROM risk WHERE id=?", (risk_id,))
        if ep is None or risk is None: raise KeyError("episode or risk not found")
        c = SignalContent.model_validate(content or {"summary": note or ""})
        sid = new_id()
        with self.store.tx(actor) as tx:
            tx.execute("INSERT INTO signal(id,episode_id,risk_id,origin,via,radar_id,content_json,note,missing_json,status,versions_json,created_at)"
                       " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (sid, episode_id, risk_id, "analyst", "analyst", None, c.model_dump_json(), note, "[]", "open",
                                                           dumps({"pipeline": "poc-1"}), now()))
            tx.audit("signal.created", "signal", sid, after={"episode_id": episode_id, "risk_id": risk_id, "via": "analyst", "note": note})
        self.engine.reason(sid, actor="system")
        return self.get(sid)

    def get(self, signal_id: str) -> Signal:
        r = self.store.one("SELECT * FROM signal WHERE id=?", (signal_id,))
        if r is None: raise KeyError(signal_id)
        return self._build(r)

    def list(self, *, risk_id: str | None = None, status: str | None = None, via: str | None = None, episode_id: str | None = None,
             min_strength: float | None = None, since: str | None = None) -> list[Signal]:
        q, p = "SELECT * FROM signal WHERE 1=1", []
        for col, v in (("risk_id", risk_id), ("status", status), ("via", via), ("episode_id", episode_id)):
            if v: q += f" AND {col}=?"; p.append(v)
        if since: q += " AND created_at>=?"; p.append(since)
        out = [self._build(r) for r in self.store.all(q + " ORDER BY created_at", p)]
        if min_strength is not None:
            out = [s for s in out if (s.strength_analyst if s.strength_analyst is not None else (s.strength_system or 0)) >= min_strength]
        return out

    def update(self, signal_id: str, actor: str, *, strength: float | None = None, status: str | None = None, note: str | None = None) -> Signal:
        with self.store.tx(actor) as tx:
            r = tx.one("SELECT * FROM signal WHERE id=?", (signal_id,))
            if r is None: raise KeyError(signal_id)
            before = {"strength_analyst": r["strength_analyst"], "status": r["status"], "note": r["note"]}
            if strength is not None:
                if not 0 <= strength <= 1: raise ValueError("strength must be in [0,1]")
                tx.execute("UPDATE signal SET strength_analyst=? WHERE id=?", (strength, signal_id))
            if status is not None:
                if status not in ("open", "reasoned", "confirmed", "dismissed"): raise ValueError(status)
                tx.execute("UPDATE signal SET status=? WHERE id=?", (status, signal_id))
            if note is not None: tx.execute("UPDATE signal SET note=? WHERE id=?", (note, signal_id))
            tx.audit("signal.updated", "signal", signal_id, before=before, after={"strength_analyst": strength, "status": status, "note": note,
                     "system_was": r["strength_system"] if strength is not None else None})
        return self.get(signal_id)

    def remove(self, signal_id: str, actor: str, reason: str) -> None:
        self.update(signal_id, actor, status="dismissed", note=reason)


class TrajectoriesAPI:
    def __init__(self, store: Store): self.store = store

    def get(self, trajectory_id: str) -> Trajectory:
        r = self.store.one("SELECT * FROM trajectory WHERE id=?", (trajectory_id,))
        if r is None: raise KeyError(trajectory_id)
        return _traj(r)

    def list(self, *, signal_id: str | None = None, risk_id: str | None = None, status: str | None = None) -> list[Trajectory]:
        q, p = "SELECT * FROM trajectory WHERE 1=1", []
        for col, v in (("signal_id", signal_id), ("risk_id", risk_id), ("status", status)):
            if v: q += f" AND {col}=?"; p.append(v)
        return [_traj(r) for r in self.store.all(q + " ORDER BY created_at", p)]

    def update(self, trajectory_id: str, actor: str, *, status: str) -> Trajectory:
        """confirm → precedent (context) and hypothesis support; reject → recorded."""
        if status not in ("confirmed", "rejected", "proposed"): raise ValueError(status)
        with self.store.tx(actor) as tx:
            r = tx.one("SELECT t.*, h.claim FROM trajectory t JOIN hypothesis h ON h.id=t.hypothesis_id WHERE t.id=?", (trajectory_id,))
            if r is None: raise KeyError(trajectory_id)
            tx.execute("UPDATE trajectory SET status=? WHERE id=?", (status, trajectory_id))
            tx.audit("trajectory.updated", "trajectory", trajectory_id, before={"status": r["status"]}, after={"status": status})
            if status == "confirmed":
                text = f"Confirmed trajectory ({r['outcome']}{': ' + r['ordinary_outcome'] if r['ordinary_outcome'] else ''}). Claim: {r['claim']}. {r['rationale']}"
                tx.execute("INSERT INTO context_item(id,kind,title,text,tags_json,origin,ref_type,ref_id,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                           (new_id(), "precedent", f"precedent for signal {r['signal_id']}", text, dumps([r["risk_id"]]), "system", "trajectory", trajectory_id, now()))
                if r["outcome"] == "risk":
                    tx.execute("UPDATE hypothesis SET status='supported' WHERE id=? AND status='open'", (r["hypothesis_id"],))
                    tx.execute("UPDATE signal SET status='confirmed' WHERE id=?", (r["signal_id"],))
        return self.get(trajectory_id)


class HypothesesAPI:
    def __init__(self, store: Store): self.store = store

    def get(self, hypothesis_id: str) -> Hypothesis:
        r = self.store.one("SELECT * FROM hypothesis WHERE id=?", (hypothesis_id,))
        if r is None: raise KeyError(hypothesis_id)
        return _hyp(r)

    def list(self, *, risk_id: str | None = None, status: str | None = None, include_ordinary: bool = False, is_lambda: bool | None = None) -> list[Hypothesis]:
        q, p = "SELECT * FROM hypothesis WHERE 1=1", []
        if risk_id: q += " AND risk_id=?"; p.append(risk_id)
        if status: q += " AND status=?"; p.append(status)
        elif not include_ordinary: q += " AND status!='ordinary'"
        if is_lambda is not None: q += " AND is_lambda=?"; p.append(int(is_lambda))
        return [_hyp(r) for r in self.store.all(q + " ORDER BY created_at", p)]

    def signals(self, hypothesis_id: str) -> list[dict]:
        """Supporting signals (risk trajectories on this hypothesis), ordered by occurrence time then creation."""
        rows = self.store.all("SELECT s.id AS signal_id, e.occurred_start, e.occurred_kind, t.reach, t.fit, t.status FROM trajectory t JOIN signal s ON s.id=t.signal_id "
                              "JOIN episode e ON e.id=s.episode_id WHERE t.hypothesis_id=? AND t.outcome='risk' AND s.status!='dismissed' "
                              "ORDER BY COALESCE(e.occurred_start, '9999'), s.created_at", (hypothesis_id,))
        return [dict(r) for r in rows]

    def belief(self, risk_id: str) -> dict[str, float]:
        """Latest posterior for the risk, as hypothesis id -> probability (includes baselines and 'none')."""
        r = self.store.one("SELECT distribution_json FROM posterior WHERE risk_id=? ORDER BY created_at DESC LIMIT 1", (risk_id,))
        return json.loads(r["distribution_json"]) if r else {}

    def coverage(self, risk_id: str) -> Coverage:
        items, bel = [], self.belief(risk_id)
        for h in self.list(risk_id=risk_id):
            sup = self.signals(h.id)
            if not sup: continue
            items.append(CoverageItem(hypothesis=h, support=len(sup), max_reach=max(s["reach"] for s in sup),
                                      first_seen=sup[0]["occurred_start"], last_seen=sup[-1]["occurred_start"], belief=bel.get(h.id)))
        items.sort(key=lambda i: (i.max_reach, i.support))
        return Coverage(risk_id=risk_id, items=items, path_reach=max((i.max_reach for i in items), default=0.0))
