"""Consequences of an episode (or an analyst-raised signal): radar reads, Lambda-signal check,
signals, trajectories (risk + baselines), hypothesis matching, strength. Inline, synchronous."""
from __future__ import annotations
import json
from .store import Store, new_id, now, dumps
from .llm import Gateway, LLMError
from .models import SignalContent

PIPELINE_VERSION = "poc-1"


def strength_from(reach: float, fit: float, gap: float, specificity: float, credibility: float) -> float:
    """Deterministic, documented, replayable. gap < 0 means an ordinary explanation fits better."""
    s = 0.4 * fit + 0.4 * max(gap, 0.0) + 0.2 * reach
    if gap < 0:
        s *= max(0.0, 1.0 + gap)        # ordinary explanation wins: scale down
    s *= 0.85 + 0.15 * specificity
    s *= 0.7 + 0.3 * credibility
    return round(min(1.0, max(0.0, s)), 3)


class Engine:
    def __init__(self, store: Store, gw: Gateway, settings, log=None):
        self.store, self.gw, self.settings, self.log = store, gw, settings, (log or (lambda m: None))

    # ---- context helpers ----
    def context_snippets(self, query: str, k: int = 6) -> list[str]:
        from .context import search_items
        return [f"[{it['kind']}] {it['text'][:600]}" for it in search_items(self.store, query, k)]

    def precedents(self, risk_id: str, k: int = 5) -> list[dict]:
        rows = self.store.all("SELECT t.*, h.claim FROM trajectory t JOIN hypothesis h ON h.id=t.hypothesis_id "
                              "WHERE t.risk_id=? AND t.status='confirmed' ORDER BY t.created_at DESC LIMIT ?", (risk_id, k))
        return [{"claim": r["claim"], "outcome": r["outcome"], "ordinary_outcome": r["ordinary_outcome"], "reach": r["reach"], "fit": r["fit"], "rationale": r["rationale"]} for r in rows]

    # ---- episode consequences ----
    def process_episode(self, episode_id: str, actor: str = "system") -> list[str]:
        ep = self.store.one("SELECT * FROM episode WHERE id=?", (episode_id,))
        src = self.store.one("SELECT * FROM source WHERE id=?", (ep["source_id"],))
        risks = self.store.all("SELECT * FROM risk WHERE status='active'")
        created = []
        for risk in risks:
            hit = None
            for radar in self.store.all("SELECT * FROM radar WHERE risk_id=? AND enabled=1", (risk["id"],)):
                try:
                    out, call_id = self.gw.call("radar.read", {"risk_text": risk["text"], "radar_query": radar["query"], "episode_text": ep["text"],
                                                               "occurred_kind": ep["occurred_kind"], "source": src["name"]},
                                                subject=("episode", episode_id))
                except LLMError as e:
                    self.log(f"radar.read failed: {e}"); continue
                if out.relevant and out.confidence >= 0.5:
                    hit = (radar, out, call_id); break
            if hit:
                radar, out, call_id = hit
                sid = self._create_signal(ep, risk, "system", "radar", radar["id"], out.content, out.missing, actor, call_id)
                created.append(sid)
            else:
                # No radar matched: Lambda-signal check against this risk, on the operational tier.
                try:
                    out, call_id = self.gw.call("signal.lambda_check", {"risk_text": risk["text"], "episode_text": ep["text"], "context": self.context_snippets(risk["text"], 3)},
                                                subject=("episode", episode_id))
                except LLMError as e:
                    self.log(f"lambda_check failed: {e}"); continue
                if out.relevant and out.confidence >= 0.5:
                    sid = self._create_signal(ep, risk, "system", "lambda", None, out.content, [], actor, call_id, note=out.why)
                    created.append(sid)
        for sid in created:
            self.reason(sid, actor)
        return created

    def _create_signal(self, ep, risk, origin, via, radar_id, content, missing, actor, call_id, note=None) -> str:
        sid = new_id()
        c = SignalContent.model_validate(content.model_dump() if content is not None and hasattr(content, "model_dump") else (content or {}))
        with self.store.tx(actor) as tx:
            tx.execute("INSERT INTO signal(id,episode_id,risk_id,origin,via,radar_id,content_json,note,missing_json,status,versions_json,created_at)"
                       " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                       (sid, ep["id"], risk["id"], origin, via, radar_id, c.model_dump_json(), note, dumps(list(missing)), "open",
                        dumps({"pipeline": PIPELINE_VERSION, "model": self.gw.settings.model, "fast_model": self.gw.settings.fast_model}), now()))
            tx.audit("signal.created", "signal", sid, after={"episode_id": ep["id"], "risk_id": risk["id"], "via": via, "radar_id": radar_id}, llm_call_id=call_id)
        return sid

    # ---- reasoning: trajectories, hypotheses, strength ----
    def reason(self, signal_id: str, actor: str = "system", *, persist: bool = True, model: str | None = None, prompt_version: str | None = None) -> dict:
        s = self.store.one("SELECT * FROM signal WHERE id=?", (signal_id,))
        ep = self.store.one("SELECT * FROM episode WHERE id=?", (s["episode_id"],))
        risk = self.store.one("SELECT * FROM risk WHERE id=?", (s["risk_id"],))
        src = self.store.one("SELECT * FROM source WHERE id=?", (ep["source_id"],))
        content = json.loads(s["content_json"])
        existing = [{"id": h["id"], "claim": h["claim"]} for h in self.store.all("SELECT id, claim FROM hypothesis WHERE risk_id=? AND status!='rejected'", (risk["id"],))]
        base_vars = {"risk_text": risk["text"], "episode_text": ep["text"], "occurred_kind": ep["occurred_kind"], "content": content,
                     "context": self.context_snippets(ep["text"], 4), "precedents": self.precedents(risk["id"])}
        rt, c1 = self.gw.call("trajectory.risk", {**base_vars, "existing_hypotheses": existing}, subject=("signal", signal_id), model=model, prompt_version=prompt_version)
        bl, c2 = self.gw.call("trajectory.baseline", {**base_vars, "max": self.settings.max_baselines}, subject=("signal", signal_id), model=model, prompt_version=prompt_version)
        best_baseline = max((b.fit for b in bl.baselines), default=0.0)
        specificity = 1.0 if content.get("target") else (0.5 if content.get("entities") else 0.2)
        metrics = {"reach": round(rt.reach, 3), "fit": round(rt.fit, 3), "gap": round(rt.fit - best_baseline, 3), "specificity": specificity, "credibility": src["credibility"]}
        strength = strength_from(**metrics)
        result = {"metrics": metrics, "strength": strength, "hypothesis": rt.hypothesis_id or rt.claim, "rationale": rt.rationale,
                  "baselines": [b.model_dump() for b in bl.baselines], "missing": sorted(set(json.loads(s["missing_json"])) | set(rt.missing))}
        if not persist:
            return result
        with self.store.tx(actor) as tx:
            hid = rt.hypothesis_id if rt.hypothesis_id and tx.one("SELECT 1 FROM hypothesis WHERE id=?", (rt.hypothesis_id,)) else None
            if hid is None:
                claim = (rt.claim or "unspecified connection to the risk").strip()
                row = tx.one("SELECT id FROM hypothesis WHERE risk_id=? AND lower(claim)=lower(?)", (risk["id"], claim))
                if row: hid = row["id"]
                else:
                    hid = new_id()
                    tx.execute("INSERT INTO hypothesis(id,risk_id,claim,entities_json,status,origin,created_at) VALUES(?,?,?,?,?,?,?)",
                               (hid, risk["id"], claim, dumps(rt.entities), "open", "system", now()))
                    tx.audit("hypothesis.created", "hypothesis", hid, after={"claim": claim, "risk_id": risk["id"]}, llm_call_id=c1)
            tx.execute("DELETE FROM trajectory WHERE signal_id=? AND status='proposed'", (signal_id,))
            tid = new_id()
            tx.execute("INSERT INTO trajectory(id,signal_id,episode_id,risk_id,radar_id,hypothesis_id,outcome,ordinary_outcome,reach,fit,rationale,status,origin,created_at)"
                       " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (tid, signal_id, ep["id"], risk["id"], s["radar_id"], hid, "risk", None, rt.reach, rt.fit, rt.rationale, "proposed", "system", now()))
            tx.audit("trajectory.proposed", "trajectory", tid, after={"signal_id": signal_id, "hypothesis_id": hid, "outcome": "risk", "reach": rt.reach, "fit": rt.fit}, llm_call_id=c1)
            for b in bl.baselines:
                brow = tx.one("SELECT id FROM hypothesis WHERE risk_id=? AND lower(claim)=lower(?)", (risk["id"], b.claim.strip()))
                bhid = brow["id"] if brow else new_id()
                if not brow:
                    tx.execute("INSERT INTO hypothesis(id,risk_id,claim,entities_json,status,origin,created_at) VALUES(?,?,?,?,?,?,?)",
                               (bhid, risk["id"], b.claim.strip(), "[]", "ordinary", "system", now()))
                btid = new_id()
                tx.execute("INSERT INTO trajectory(id,signal_id,episode_id,risk_id,radar_id,hypothesis_id,outcome,ordinary_outcome,reach,fit,rationale,status,origin,created_at)"
                           " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (btid, signal_id, ep["id"], risk["id"], s["radar_id"], bhid, "ordinary", b.ordinary_outcome, 0.0, b.fit, b.rationale, "proposed", "system", now()))
                tx.audit("trajectory.proposed", "trajectory", btid, after={"signal_id": signal_id, "outcome": "ordinary", "ordinary_outcome": b.ordinary_outcome, "fit": b.fit}, llm_call_id=c2)
            tx.execute("UPDATE signal SET strength_system=?, metrics_json=?, missing_json=?, status=CASE WHEN status='open' THEN 'reasoned' ELSE status END WHERE id=?",
                       (strength, dumps(metrics), dumps(result["missing"]), signal_id))
            tx.audit("signal.reasoned", "signal", signal_id, after={"strength_system": strength, "metrics": metrics}, llm_call_id=c1)
        return result
