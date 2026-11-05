"""The reference implementation of `assess` (spec v3 §4–5).

    core = Core(Settings.from_env())
    result = core.assess(mission, operational, TrajectoryInput(episode=..., hypothesis=..., risk=...))

Flow per candidate risk: signal (given | radar → predefined | articulated) → abduction (complete the hypothesis,
match stored ones) → baselines (stored first, new only when uncovered) → score (formula, the episode alone) →
standing (the hypothesis with everything attached, with and without this episode) → provenance.

The layer reads both contexts and writes neither. Its only state is derived: radars per (mission version, risk,
stored claims) and the model-output cache, so identical inputs return identical results.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading

from .llm import Gateway, LLMAuthError, LLMError, Recorder, make_backend
from .llm.prompts import TrajectoryDraft
from .scoring import noisy_or, score_from, uses, weight_of
from .settings import Settings
from .types import (AssessedTrajectory, Assessment, AssessmentGroup, Episode, Hypothesis, HypothesisDraft, HypothesisRef,
                    MissionContext, OperationalContext, Provenance, Risk, RiskRef, SignalContent, SignalOut, Standing,
                    TrajectoryInput)

IMPLEMENTATION = "reference"
VERSION = "0.3.0"
_TOKEN = re.compile(r"\w+")


def _norm(s: str) -> str:
    return " ".join(_TOKEN.findall((s or "").lower()))


class _Run:
    """Bookkeeping for one `assess` call: model calls and operational items read."""
    def __init__(self):
        self.rec = Recorder()
        self.read: dict[str, int] = {}

    def saw(self, *items):
        for it in items:
            self.read[it.id] = it.version


class Core:
    def __init__(self, settings: Settings | None = None, *, provider=None):
        self.settings = settings or Settings.from_env()
        self.provider = provider or make_backend(self.settings)
        self.gw = Gateway(self.settings, self.provider)
        self._radars: dict[tuple, list[str]] = {}
        self._lock = threading.Lock()

    # ================================================================ public
    def assess(self, mission: MissionContext, operational: OperationalContext, trajectory: TrajectoryInput, *,
               scope: str = "given", detail: str = "full") -> Assessment:
        if scope not in ("given", "open"):
            raise ValueError("scope must be 'given' or 'open'")
        if detail not in ("full", "scores"):
            raise ValueError("detail must be 'full' or 'scores'")
        run = _Run()
        groups = self._groups(mission, operational, trajectory, run, primary=True)
        if scope == "open" and (trajectory.hypothesis is not None or trajectory.risk is not None):
            stripped = TrajectoryInput(episode=trajectory.episode, signal=trajectory.signal)
            seen = {self._key(g) for g in groups}
            for g in self._groups(mission, operational, stripped, run, primary=False, include_rejected=True):
                if self._key(g) not in seen:
                    groups.append(g)
                    seen.add(self._key(g))
        if detail == "scores":
            groups = [self._strip(g) for g in groups]
        prov = Provenance(
            implementation=IMPLEMENTATION, implementation_version=VERSION,
            models=tuple(sorted({c.model for c in run.rec.calls})), prompts={c.task: c.prompt_version for c in run.rec.calls},
            mission_version=mission.version, operational_items=dict(sorted(run.read.items())),
            input_hash=hashlib.sha256(json.dumps({"t": trajectory.model_dump(mode="json", by_alias=True), "scope": scope,
                                                  "mission": mission.version}, sort_keys=True).encode()).hexdigest()[:16],
            deterministic=True, calls=tuple(run.rec.calls))
        return Assessment(groups=tuple(groups), pattern=trajectory.pattern, provenance=prov)

    # ================================================================ groups
    @staticmethod
    def _key(g: AssessmentGroup) -> str:
        h = g.target.hypothesis
        return f"{g.target.risk.risk_id}|{h.id or _norm(h.claim)}"

    def _groups(self, mission: MissionContext, op: OperationalContext, ti: TrajectoryInput, run: _Run, *,
                primary: bool, include_rejected: bool = False) -> list[AssessmentGroup]:
        given_h: Hypothesis | None = None
        given_claim: str | None = None
        if isinstance(ti.hypothesis, str):
            given_h = op.hypothesis(ti.hypothesis)
            if given_h is None:
                raise KeyError(f"hypothesis {ti.hypothesis!r} is not in the operational context")
            run.saw(given_h)
            given_claim = given_h.claim
        elif isinstance(ti.hypothesis, HypothesisDraft):
            given_claim = ti.hypothesis.claim

        inconsistency = None
        if ti.risk is not None:
            r = mission.risk(ti.risk)
            if r is None:
                raise KeyError(f"risk {ti.risk!r} is not in the mission context")
            risks, risk_source = [r], "given"
            if given_h is not None and given_h.risk_id != r.id:
                inconsistency = f"hypothesis {given_h.id} belongs to risk {given_h.risk_id}, not to the given risk {r.id}"
        elif given_h is not None:
            r = mission.risk(given_h.risk_id)
            if r is None:
                raise KeyError(f"hypothesis {given_h.id} refers to risk {given_h.risk_id!r}, which is not in the mission context")
            risks, risk_source = [r], "predefined"
        else:
            risks, risk_source = list(mission.active_risks), "predefined"

        pointed = ti.hypothesis is not None or ti.risk is not None
        candidates = []
        for risk in risks:
            candidates += self._for_risk(mission, op, ti, risk, risk_source, given_h, given_claim, pointed, run, include_rejected)

        if given_claim is not None and not given_h and len(risks) > 1 and candidates:
            # a draft hypothesis without a risk: one group, under the risk it fits best
            candidates = [max(candidates, key=lambda g: (g.target.fit * (0.5 + g.target.reach), g.target.score))]
        if inconsistency:
            candidates = [g.model_copy(update={"inconsistency": inconsistency}) for g in candidates]
        given_first = sorted(candidates, key=lambda g: (g.target.hypothesis.source != "given", -g.target.score))
        return given_first

    # ---------------------------------------------------------------- one risk
    def _for_risk(self, mission, op: OperationalContext, ti: TrajectoryInput, risk: Risk, risk_source, given_h, given_claim,
                  pointed: bool, run: _Run, include_rejected: bool) -> list[AssessmentGroup]:
        ep = ti.episode
        ctx = self._snippets(mission, ep.text + " " + risk.text, 4)

        # 1. signal
        if ti.signal is not None:
            sig_content = ti.signal.content or SignalContent(summary=ti.signal.note)
            signal = SignalOut(content=sig_content, source="given")
        else:
            signal = None
            for q in self._radar_queries(mission, op, risk, run):
                try:
                    out = self.gw.call("radar.read", {"risk_text": risk.text, "radar_query": q, "episode_text": ep.text, "source": ep.source},
                                       recorder=run.rec)
                except LLMAuthError:
                    raise
                except LLMError:
                    continue
                if out.relevant and out.confidence >= self.settings.relevance_threshold:
                    signal = SignalOut(content=SignalContent(**self._content(out.content)), source="predefined", radar=q)
                    break
            if signal is None and not pointed:
                out = self.gw.call("signal.lambda_check", {"risk_text": risk.text, "episode_text": ep.text, "context": ctx}, recorder=run.rec)
                if not (out.relevant and out.confidence >= self.settings.relevance_threshold):
                    return []
                signal = SignalOut(content=SignalContent(**self._content(out.content)), source="articulated")

        # 2. abduction
        stored = op.primaries(risk.id)
        rejected = [h for h in op.primaries(risk.id, include_rejected=True) if h.status == "rejected"]
        run.saw(*stored, *rejected)
        precedents = op.precedents(risk.id)
        run.saw(*precedents)
        variables = {
            "risk_text": risk.text, "context": ctx,
            "episode": {"text": ep.text, "source": ep.source, "credibility": ep.credibility, "occurred": ep.occurred},
            "signal": None if ti.signal is None else {**ti.signal.content.model_dump(), "note": ti.signal.note} if ti.signal.content else {"note": ti.signal.note},
            "detected_signal": None if signal is None or signal.source == "given" else signal.content.model_dump(),
            "given_hypothesis": given_claim,
            "stored_hypotheses": [{"id": h.id, "claim": h.claim, "status": h.status} for h in stored if h.id != (given_h.id if given_h else None)],
            "rejected_hypotheses": [{"id": h.id, "claim": h.claim} for h in rejected],
            "precedents": [{"claim": op.hypothesis(t.hypothesis_id).claim if op.hypothesis(t.hypothesis_id) else "", "rationale": t.rationale,
                            "fit": t.fit, "reach": t.reach} for t in precedents],
            "max": 1 if given_claim else self.settings.max_trajectories,
        }
        ab = self.gw.call("trajectory.abduce", variables, recorder=run.rec)
        if given_claim is None and (not ab.relevant or not ab.trajectories):
            return []
        if signal is None:   # pointed at by the analyst, no radar matched: the layer articulates the signal
            signal = SignalOut(content=SignalContent(**self._content(ab.signal)) if ab.signal else SignalContent(summary=ep.text[:140]),
                               source="articulated")
        elif signal.source == "articulated" and ab.signal is not None:
            signal = SignalOut(content=SignalContent(**self._content(ab.signal)), source="articulated")

        drafts = ab.trajectories[:1] if given_claim else ab.trajectories[: self.settings.max_trajectories]
        if given_claim and not drafts:
            drafts = [TrajectoryDraft(hypothesis_id="given", rationale="The model returned no trajectory for the given hypothesis.")]

        groups, seen = [], set()
        all_known = {h.id: h for h in op.primaries(risk.id, include_rejected=True)}
        for d in drafts:
            href = self._resolve(d, given_h, given_claim, all_known, rejected)
            if href is None:
                continue
            if href.status == "rejected" and href.source != "given" and not include_rejected:
                continue
            k = href.id or _norm(href.claim)
            if k in seen:
                continue
            seen.add(k)
            groups.append(self._group(op, ti, risk, risk_source, signal, href, d, run))
        return groups

    def _resolve(self, d, given_h, given_claim, known: dict[str, Hypothesis], rejected: list[Hypothesis]) -> HypothesisRef | None:
        if given_claim is not None:
            if given_h is not None:
                return HypothesisRef(source="given", id=given_h.id, version=given_h.version, claim=given_h.claim,
                                     entities=given_h.entities, status=given_h.status)
            return HypothesisRef(source="given", claim=given_claim, entities=tuple(d.entities))
        if d.hypothesis_id and d.hypothesis_id in known and d.match_confidence >= self.settings.match_threshold:
            h = known[d.hypothesis_id]
            return HypothesisRef(source="predefined", id=h.id, version=h.version, claim=h.claim, entities=h.entities,
                                 match_confidence=round(d.match_confidence, 3), status=h.status)
        claim = (d.claim or (known[d.hypothesis_id].claim if d.hypothesis_id in known else "")).strip()
        if not claim:
            return None
        for h in rejected:   # never articulate a rejected claim again
            if _norm(h.claim) == _norm(claim):
                return HypothesisRef(source="predefined", id=h.id, version=h.version, claim=h.claim, entities=h.entities,
                                     match_confidence=1.0, status="rejected")
        return HypothesisRef(source="articulated", claim=claim, entities=tuple(d.entities))

    # ---------------------------------------------------------------- one group
    def _group(self, op: OperationalContext, ti: TrajectoryInput, risk: Risk, risk_source, signal: SignalOut, href: HypothesisRef,
               d, run: _Run) -> AssessmentGroup:
        ep = ti.episode
        stored_b = op.baselines_of(href.id) if href.id else []
        run.saw(*stored_b)
        bl = self.gw.call("trajectory.baselines", {
            "risk_text": risk.text, "episode": {"text": ep.text, "source": ep.source, "occurred": ep.occurred},
            "signal": signal.content.model_dump(), "hypothesis": href.claim,
            "stored_baselines": [{"id": b.id, "claim": b.claim} for b in stored_b],
            "max_new": self.settings.max_baselines if not stored_b else 1}, recorder=run.rec)
        by_id = {b.id: b for b in stored_b}
        rref = RiskRef(risk_id=risk.id, source=risk_source)
        baselines = []
        for a in bl.assessed:
            b = by_id.get(a.baseline_id)
            if b is None:
                continue
            baselines.append(AssessedTrajectory(
                episode_id=ep.id, role="baseline", signal=signal, risk=rref,
                hypothesis=HypothesisRef(source="predefined", id=b.id, version=b.version, claim=b.claim, match_confidence=1.0, status=b.status),
                score=round(a.fit, 3), reach=0.0, fit=round(a.fit, 3), rationale=a.rationale, ordinary_outcome=a.ordinary_outcome or b.claim))
        known = {_norm(b.hypothesis.claim) for b in baselines}
        for n in bl.new[: self.settings.max_baselines]:
            if _norm(n.claim) in known:
                continue
            known.add(_norm(n.claim))
            baselines.append(AssessedTrajectory(
                episode_id=ep.id, role="baseline", signal=signal, risk=rref, hypothesis=HypothesisRef(source="articulated", claim=n.claim),
                score=round(n.fit, 3), reach=0.0, fit=round(n.fit, 3), rationale=n.rationale, ordinary_outcome=n.ordinary_outcome))

        best_b = max((b.fit for b in baselines), default=0.0)
        fit, reach = round(d.fit, 3), round(d.reach, 3)
        score = score_from(reach, fit, fit - best_b, signal.content.specificity, ep.credibility)
        target = AssessedTrajectory(episode_id=ep.id, role="primary", signal=signal, hypothesis=href, risk=rref, score=score,
                                    reach=reach, fit=fit, missing=tuple(d.missing), rationale=d.rationale)
        return AssessmentGroup(target=target, baselines=tuple(baselines), standing=self._standing(op, ti, risk, target, baselines, run))

    # ---------------------------------------------------------------- standing
    def _standing(self, op: OperationalContext, ti: TrajectoryInput, risk: Risk, target: AssessedTrajectory, baselines, run: _Run) -> Standing:
        pw = self.settings.proposed_weight
        attached = op.attached(target.hypothesis.id)[-self.settings.max_attached:] if target.hypothesis.id else []
        attached = [t for t in attached if t.episode_id != ti.episode.id]   # re-assessing an episode replaces its own trajectory
        run.saw(*attached)
        items = [{"id": t.id, "status": t.status, "weight": weight_of(t.status, pw), "score": t.score, "fit": t.fit, "reach": t.reach,
                  "occurred": t.occurred, "summary": t.signal.summary, "rationale": t.rationale} for t in attached]
        analysts_own = ti.signal is not None and ti.hypothesis is not None
        this = {"id": f"this:{ti.episode.id}", "status": "this", "weight": 1.0 if analysts_own else pw, "score": target.score,
                "fit": target.fit, "reach": target.reach, "occurred": ti.episode.occurred if isinstance(ti.episode.occurred, str) else None,
                "summary": target.signal.content.summary, "rationale": target.rationale}
        without = self._judge(risk, target, baselines, items, run) if any(i["weight"] > 0 for i in items) else (0.0, None)
        value = self._judge(risk, target, baselines, items + [this], run)
        return Standing(value=value[0], without=without[0], delta=round(value[0] - without[0], 3),
                        used=uses(items + [this]), rationale=value[1])

    def _judge(self, risk: Risk, target, baselines, items: list[dict], run: _Run) -> tuple[float, str | None]:
        if self.settings.standing == "noisy_or":
            return noisy_or(items), "noisy-OR of weighted scores"
        out = self.gw.call("standing.judge", {
            "risk_text": risk.text, "hypothesis": target.hypothesis.claim,
            "baselines": sorted({b.hypothesis.claim for b in baselines}),
            "trajectories": [{k: i[k] for k in ("status", "weight", "score", "fit", "reach", "occurred", "summary", "rationale")} for i in items]},
            recorder=run.rec)
        return round(max(0.0, min(1.0, out.value)), 3), out.rationale

    # ================================================================ helpers
    def _radar_queries(self, mission: MissionContext, op: OperationalContext, risk: Risk, run: _Run) -> list[str]:
        claims = sorted(h.claim for h in op.primaries(risk.id))
        key = (mission.version, risk.id, hashlib.sha256("|".join(claims).encode()).hexdigest()[:12])
        with self._lock:
            if key in self._radars:
                return self._radars[key]
        out = self.gw.call("radar.derive", {"risk_text": risk.text, "context": self._snippets(mission, risk.text, 6), "hypotheses": claims},
                           recorder=run.rec)
        qs = [r.query.strip() for r in out.radars if r.query.strip()]
        with self._lock:
            self._radars[key] = qs
        return qs

    @staticmethod
    def _snippets(mission: MissionContext, query: str, k: int) -> list[str]:
        q = {t.lower() for t in _TOKEN.findall(query) if len(t) > 2}
        scored = []
        for d in mission.documents:
            tl = [t.lower() for t in _TOKEN.findall(d.title + " " + d.text)]
            hits = sum(1 for t in tl if t in q)
            if hits:
                scored.append((hits / (1 + len(tl) ** 0.5), d))
        scored.sort(key=lambda x: -x[0])
        return [f"[{d.title or d.id}] {d.text[:600]}" for _, d in scored[:k]]

    @staticmethod
    def _content(c) -> dict:
        if c is None:
            return {}
        d = c.model_dump() if hasattr(c, "model_dump") else dict(c)
        d["modifiers"] = tuple(d.get("modifiers") or ())
        d["entities"] = tuple(d.get("entities") or ())
        return {k: v for k, v in d.items() if k in SignalContent.model_fields}

    @staticmethod
    def _strip(g: AssessmentGroup) -> AssessmentGroup:
        return g.model_copy(update={
            "target": g.target.model_copy(update={"rationale": None}),
            "baselines": tuple(b.model_copy(update={"rationale": None}) for b in g.baselines),
            "standing": g.standing.model_copy(update={"rationale": None})})


_default: Core | None = None


def assess(mission: MissionContext, operational: OperationalContext, trajectory: TrajectoryInput, *,
           scope: str = "given", detail: str = "full") -> Assessment:
    """Module-level convenience: one shared Core built from the environment (MISTABRA_* variables)."""
    global _default
    if _default is None:
        _default = Core()
    return _default.assess(mission, operational, trajectory, scope=scope, detail=detail)
