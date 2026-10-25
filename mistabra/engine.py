"""Consequences of an episode (or an analyst-raised signal): radar reads, Lambda-signal check,
signals, then reasoning — abduction (risk trajectory + baselines), the decision model (posterior + strength),
Lambda hypotheses, persistence. Inline, synchronous."""
from __future__ import annotations
import json
from .store import Store, new_id, now, dumps
from .llm import Gateway, LLMError
from .models import SignalContent
from .decision import (NONE, Abduction, AttachedSignal, HypothesisRef, HypothesisSet, Posterior, State, make_decision)
from .decision.pairwise import strength_from  # noqa: F401  (re-exported for tests and replay tooling)

PIPELINE_VERSION = "poc-2"
NEW = "new"


class Engine:
    def __init__(self, store: Store, gw: Gateway, settings, log=None):
        self.store, self.gw, self.settings, self.log = store, gw, settings, (log or (lambda m: None))
        self._decisions: dict = {}

    def decision(self, name: str | None = None):
        name = name or self.settings.decision
        if name not in self._decisions:
            self._decisions[name] = make_decision(name, self.gw, self.settings)
        return self._decisions[name]

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
                        dumps({"pipeline": PIPELINE_VERSION, "model": self.gw.settings.model, "fast_model": self.gw.settings.fast_model, "decision": self.settings.decision}), now()))
            tx.audit("signal.created", "signal", sid, after={"episode_id": ep["id"], "risk_id": risk["id"], "via": via, "radar_id": radar_id}, llm_call_id=call_id)
        return sid

    # ---- abduction ----
    def abduce(self, s, ep, risk, src, content, existing: list[dict], *, force_new: bool = False, model=None, prompt_version=None) -> Abduction:
        base_vars = {"risk_text": risk["text"], "episode_text": ep["text"], "occurred_kind": ep["occurred_kind"], "content": content,
                     "context": self.context_snippets(ep["text"], 4), "precedents": self.precedents(risk["id"])}
        rt, c1 = self.gw.call("trajectory.risk", {**base_vars, "existing_hypotheses": existing, "force_new": force_new}, subject=("signal", s["id"]), model=model, prompt_version=prompt_version)
        bl, c2 = self.gw.call("trajectory.baseline", {**base_vars, "max": self.settings.max_baselines}, subject=("signal", s["id"]), model=model, prompt_version=prompt_version)
        known = {h["id"] for h in existing}
        hid = rt.hypothesis_id if (rt.hypothesis_id in known and not force_new) else NEW
        claim = (rt.claim or next((h["claim"] for h in existing if h["id"] == hid), None) or "unspecified connection to the risk").strip()
        specificity = 1.0 if content.get("target") else (0.5 if content.get("entities") else 0.2)
        return Abduction(hypothesis_id=hid, claim=claim, rationale=rt.rationale, reach=rt.reach, fit=rt.fit, entities=tuple(rt.entities), missing=tuple(rt.missing),
                         predicts=tuple(rt.predicts), baselines=tuple(b.model_dump() for b in bl.baselines), specificity=specificity, credibility=src["credibility"],
                         llm_call_ids=(c1, c2))

    # ---- state and hypothesis set ----
    def hypothesis_set(self, risk_id: str, abduction: Abduction, existing: list[dict], before: str = "9999") -> HypothesisSet:
        hyps = [HypothesisRef(id=h["id"], claim=h["claim"], kind="risk", is_lambda=bool(h.get("is_lambda")), predicts=tuple(h.get("predicts", ())), status=h.get("status", "open")) for h in existing]
        if abduction.hypothesis_id == NEW:
            hyps.append(HypothesisRef(id=NEW, claim=abduction.claim, kind="risk", is_lambda=True, predicts=abduction.predicts))
        seen, bases = set(), []
        for b in abduction.baselines:
            key = b["claim"].strip().lower()
            row = self.store.one("SELECT id FROM hypothesis WHERE risk_id=? AND lower(claim)=?", (risk_id, key))
            bid = row["id"] if row else f"baseline:{len(bases)}"
            if key not in seen:
                seen.add(key); bases.append(HypothesisRef(id=bid, claim=b["claim"].strip(), kind="baseline", status="ordinary"))
        for r in self.store.all("SELECT id, claim FROM hypothesis WHERE risk_id=? AND status='ordinary' AND created_at<? ORDER BY created_at DESC LIMIT 6", (risk_id, before)):
            if r["claim"].strip().lower() not in seen:
                seen.add(r["claim"].strip().lower()); bases.append(HypothesisRef(id=r["id"], claim=r["claim"], kind="baseline", status="ordinary"))
        return HypothesisSet(risk_id=risk_id, hypotheses=tuple(hyps), baselines=tuple(bases))

    def state_for(self, risk_id: str, signal) -> State:
        """The set's state as of this signal: signals created before it (so a replay sees what the original run saw) and the last posterior before it."""
        rows = self.store.all("SELECT t.hypothesis_id, t.signal_id, t.reach, t.fit, s.status, s.content_json, e.occurred_start FROM trajectory t "
                              "JOIN signal s ON s.id=t.signal_id JOIN episode e ON e.id=s.episode_id WHERE t.risk_id=? AND t.outcome='risk' AND s.created_at<? "
                              "AND s.status!='dismissed' ORDER BY COALESCE(e.occurred_start,'9999'), s.created_at", (risk_id, signal["created_at"]))
        attached: dict[str, list] = {}
        for r in rows:
            attached.setdefault(r["hypothesis_id"], []).append(AttachedSignal(signal_id=r["signal_id"], summary=json.loads(r["content_json"]).get("summary", "")[:200],
                                                                               occurred=r["occurred_start"], reach=r["reach"], fit=r["fit"], status=r["status"]))
        last = self.store.one("SELECT p.* FROM posterior p JOIN signal s ON s.id=p.signal_id WHERE p.risk_id=? AND s.created_at<? ORDER BY p.created_at DESC LIMIT 1", (risk_id, signal["created_at"]))
        lp = Posterior(distribution=json.loads(last["distribution_json"]), info_gain=last["info_gain"], rationale=last["rationale"], decision=last["decision"],
                       model=last["model"], prompt_version=last["prompt_version"], state_hash=last["state_hash"]) if last else None
        return State(attached={k: tuple(v) for k, v in attached.items()}, last=lp)

    # ---- reasoning: abduction → decision → persistence ----
    def reason(self, signal_id: str, actor: str = "system", *, persist: bool = True, model: str | None = None, prompt_version: str | None = None,
               decision: str | None = None) -> dict:
        s = self.store.one("SELECT * FROM signal WHERE id=?", (signal_id,))
        ep = self.store.one("SELECT * FROM episode WHERE id=?", (s["episode_id"],))
        risk = self.store.one("SELECT * FROM risk WHERE id=?", (s["risk_id"],))
        src = self.store.one("SELECT * FROM source WHERE id=?", (ep["source_id"],))
        content = json.loads(s["content_json"])
        existing = [{"id": h["id"], "claim": h["claim"], "status": h["status"], "is_lambda": bool(h["is_lambda"]), "predicts": json.loads(h["predicts_json"])}
                    for h in self.store.all("SELECT * FROM hypothesis WHERE risk_id=? AND status NOT IN ('rejected','ordinary') AND created_at<?", (risk["id"], s["created_at"]))]
        # Hypotheses are filtered to those that existed when the signal arrived, so a replay rebuilds the set the original run scored.
        dm = self.decision(decision)
        episode = {"text": ep["text"], "occurred_kind": ep["occurred_kind"], "occurred": ep["occurred_start"], "source": src["name"], "credibility": src["credibility"], "content": content}
        state = self.state_for(risk["id"], s)
        ab = self.abduce(s, ep, risk, src, content, [{"id": h["id"], "claim": h["claim"]} for h in existing], model=model, prompt_version=prompt_version)
        hset = self.hypothesis_set(risk["id"], ab, existing, s["created_at"])
        ctx = self.context_snippets(ep["text"], 4)
        scored = dm.score(ctx, state, episode, hset, ab, model=model, prompt_version=prompt_version, subject=("signal", signal_id))
        rescored = False
        if scored.unexplained and ab.hypothesis_id != NEW:
            # The set cannot explain the signal: the abducer proposes a Lambda hypothesis and the set is scored once more.
            ab = self.abduce(s, ep, risk, src, content, [{"id": h["id"], "claim": h["claim"]} for h in existing], force_new=True, model=model, prompt_version=prompt_version)
            hset = self.hypothesis_set(risk["id"], ab, existing, s["created_at"])
            scored = dm.score(ctx, state, episode, hset, ab, model=model, prompt_version=prompt_version, subject=("signal", signal_id))
            rescored = True
        result = {"metrics": scored.metrics, "strength": scored.strength, "hypothesis": ab.hypothesis_id if ab.hypothesis_id != NEW else ab.claim, "rationale": ab.rationale,
                  "baselines": list(ab.baselines), "missing": sorted(set(json.loads(s["missing_json"])) | set(ab.missing)), "decision": dm.name,
                  "posterior": scored.posterior.distribution, "info_gain": scored.posterior.info_gain, "none": scored.posterior.none,
                  "lambda_hypothesis": ab.hypothesis_id == NEW, "rescored": rescored}
        if not persist:
            return result
        c1 = ab.llm_call_ids[0] if ab.llm_call_ids else None
        c2 = ab.llm_call_ids[1] if len(ab.llm_call_ids) > 1 else None
        with self.store.tx(actor) as tx:
            if ab.hypothesis_id != NEW:
                hid = ab.hypothesis_id
            else:
                row = tx.one("SELECT id FROM hypothesis WHERE risk_id=? AND lower(claim)=lower(?)", (risk["id"], ab.claim))
                if row: hid = row["id"]
                else:
                    # No existing hypothesis explains the signal: the abducer creates a Lambda hypothesis.
                    hid = new_id()
                    tx.execute("INSERT INTO hypothesis(id,risk_id,claim,entities_json,status,origin,is_lambda,predicts_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                               (hid, risk["id"], ab.claim, dumps(list(ab.entities)), "open", "system", 1, dumps(list(ab.predicts)), now()))
                    tx.audit("hypothesis.created", "hypothesis", hid, after={"claim": ab.claim, "risk_id": risk["id"], "is_lambda": True, "predicts": list(ab.predicts),
                                                                               "reason": "none_of_these" if rescored else "abducer"}, llm_call_id=c1)
            tx.execute("DELETE FROM trajectory WHERE signal_id=? AND status='proposed'", (signal_id,))
            tid = new_id()
            tx.execute("INSERT INTO trajectory(id,signal_id,episode_id,risk_id,radar_id,hypothesis_id,outcome,ordinary_outcome,reach,fit,rationale,status,origin,created_at)"
                       " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (tid, signal_id, ep["id"], risk["id"], s["radar_id"], hid, "risk", None, ab.reach, ab.fit, ab.rationale, "proposed", "system", now()))
            tx.audit("trajectory.proposed", "trajectory", tid, after={"signal_id": signal_id, "hypothesis_id": hid, "outcome": "risk", "reach": ab.reach, "fit": ab.fit}, llm_call_id=c1)
            for b in ab.baselines:
                brow = tx.one("SELECT id FROM hypothesis WHERE risk_id=? AND lower(claim)=lower(?)", (risk["id"], b["claim"].strip()))
                bhid = brow["id"] if brow else new_id()
                if not brow:
                    tx.execute("INSERT INTO hypothesis(id,risk_id,claim,entities_json,status,origin,created_at) VALUES(?,?,?,?,?,?,?)",
                               (bhid, risk["id"], b["claim"].strip(), "[]", "ordinary", "system", now()))
                btid = new_id()
                tx.execute("INSERT INTO trajectory(id,signal_id,episode_id,risk_id,radar_id,hypothesis_id,outcome,ordinary_outcome,reach,fit,rationale,status,origin,created_at)"
                           " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (btid, signal_id, ep["id"], risk["id"], s["radar_id"], bhid, "ordinary", b["ordinary_outcome"], 0.0, b["fit"], b["rationale"], "proposed", "system", now()))
                tx.audit("trajectory.proposed", "trajectory", btid, after={"signal_id": signal_id, "outcome": "ordinary", "ordinary_outcome": b["ordinary_outcome"], "fit": b["fit"]}, llm_call_id=c2)
            # Posterior: keys "new" and "baseline:n" are resolved to the stored ids now that they exist.
            dist = {}
            for k, v in scored.posterior.distribution.items():
                if k == NEW: dist[hid] = v
                elif k.startswith("baseline:"):
                    ref = hset.by_id(k)
                    row = tx.one("SELECT id FROM hypothesis WHERE risk_id=? AND lower(claim)=lower(?)", (risk["id"], ref.claim)) if ref else None
                    dist[row["id"] if row else k] = v
                else: dist[k] = v
            pid = new_id()
            tx.execute("INSERT INTO posterior(id,signal_id,risk_id,decision,model,prompt_version,distribution_json,info_gain,strength,rationale,state_hash,created_at)"
                       " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (pid, signal_id, risk["id"], dm.name, scored.posterior.model, scored.posterior.prompt_version, dumps(dist),
                                                         scored.posterior.info_gain, scored.strength, scored.posterior.rationale, scored.posterior.state_hash, now()))
            tx.audit("posterior.written", "posterior", pid, after={"signal_id": signal_id, "decision": dm.name, "none": scored.posterior.none, "info_gain": scored.posterior.info_gain,
                                                                   "state_hash": scored.posterior.state_hash, "rescored": rescored}, llm_call_id=scored.llm_call_ids[-1] if scored.llm_call_ids else None)
            tx.execute("UPDATE signal SET strength_system=?, metrics_json=?, missing_json=?, status=CASE WHEN status='open' THEN 'reasoned' ELSE status END WHERE id=?",
                       (scored.strength, dumps(scored.metrics), dumps(result["missing"]), signal_id))
            tx.audit("signal.reasoned", "signal", signal_id, after={"strength_system": scored.strength, "metrics": scored.metrics, "decision": dm.name}, llm_call_id=c1)
        return result
