"""The reference implementation of `assess` (spec v3 §4–5).

    core = Core(Settings.from_env())
    result = core.assess(mission, operational, TrajectoryInput(episode=..., hypothesis=..., risk=...))

Flow per candidate risk, three model calls:
  1. episode.triage (fast): is the episode relevant — directly, or through a held hypothesis — and which radar does it
     answer? Gives the signal (given | predefined | articulated). Skipped when the analyst gives the signal; an
     irrelevant episode stops here unless the analyst pointed it at a hypothesis or risk.
  2. episode.assess (reasoning): a verdict on every held hypothesis (supports | weakens | unrelated), new hypotheses
     only for mechanisms the held ones miss, and for each supported hypothesis its ordinary explanations (baselines).
  3. standing.judge (reasoning): every hypothesis the episode touched, judged together, before and after this episode.
Then score (formula, the episode alone) and provenance. radar.derive runs once per mission version and risk.

Candidate risks are assessed in parallel; results are combined in a fixed order, so parallelism never changes an
answer.

The layer reads both contexts and writes neither. Its only state is derived: radars per (mission version, risk,
stored claims) and the model-output cache, so identical inputs return identical results.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from .llm import Gateway, LLMAuthError, LLMError, Recorder, make_backend
from .llm.prompts import BaselinesOut, GivenVerdict
from .scoring import score_margin, noisy_or, score_from, uses, weight_of
from .settings import Settings
from .types import (AssessedTrajectory, Assessment, AssessmentGroup, Episode, Hypothesis, HypothesisDraft, HypothesisRef,
                    MissionContext, OperationalContext, Provenance, Risk, RiskRef, SignalContent, SignalOut, Standing,
                    TrajectoryInput)

IMPLEMENTATION = "reference"
VERSION = "0.3.0"
_TOKEN = re.compile(r"\w+")


def _norm(s: str) -> str:
    return " ".join(_TOKEN.findall((s or "").lower()))


def _pmap(fn, items: list) -> list:
    """Map in parallel, results in input order. Exceptions propagate (the first one, after all finish)."""
    items = list(items)
    if len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(len(items), 8)) as ex:
        return list(ex.map(fn, items))


def _clip(x: float) -> float:
    return round(max(0.0, min(1.0, float(x))), 3)


@dataclass
class _Entry:
    """One hypothesis the episode bears on: its reference, the stance, and the model's assessment of it."""
    href: HypothesisRef
    stance: str
    verdict: object


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
        run.rec.label = trajectory.episode.id
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
            risks, risk_origin = [r], "given"
            if given_h is not None and given_h.risk_id != r.id:
                inconsistency = f"hypothesis {given_h.id} belongs to risk {given_h.risk_id}, not to the given risk {r.id}"
        elif given_h is not None:
            r = mission.risk(given_h.risk_id)
            if r is None:
                raise KeyError(f"hypothesis {given_h.id} refers to risk {given_h.risk_id!r}, which is not in the mission context")
            risks, risk_origin = [r], "predefined"
        else:
            risks, risk_origin = list(mission.active_risks), "predefined"

        pointed = ti.hypothesis is not None or ti.risk is not None
        candidates = [g for gs in _pmap(lambda risk: self._for_risk(mission, op, ti, risk, risk_origin, given_h, given_claim, pointed,
                                                                     run, include_rejected), risks) for g in gs]

        if given_claim is not None and not given_h and len(risks) > 1 and candidates:
            # a draft hypothesis without a risk: one group, under the risk it fits best
            candidates = [max(candidates, key=lambda g: (g.target.fit * (0.5 + g.target.reach), g.target.score))]
        if inconsistency:
            candidates = [g.model_copy(update={"inconsistency": inconsistency}) for g in candidates]
        given_first = sorted(candidates, key=lambda g: (g.target.hypothesis.origin != "given", -g.target.score))
        return given_first

    # ---------------------------------------------------------------- one risk
    def _for_risk(self, mission, op: OperationalContext, ti: TrajectoryInput, risk: Risk, risk_origin, given_h, given_claim,
                  pointed: bool, run: _Run, include_rejected: bool) -> list[AssessmentGroup]:
        ep = ti.episode
        ctx = self._snippets(mission, ep.text + " " + risk.text, 4)
        known = {h.id: h for h in op.primaries(risk.id, include_rejected=True)}
        held = [h for h in known.values() if given_h is None or h.id != given_h.id]
        run.saw(*held)

        # 1. triage → signal
        if ti.signal is not None:
            signal = SignalOut(content=ti.signal.content or SignalContent(summary=ti.signal.note), origin="given")
        else:
            queries = self._radar_queries(mission, op, risk, run)
            tr = self.gw.call("episode.triage", {
                "risk_text": risk.text, "episode": {"text": ep.text, "source": ep.source},
                "radars": [{"index": i, "query": q} for i, q in enumerate(queries)],
                "held_hypotheses": [{"id": h.id, "claim": h.claim, "status": h.status} for h in held]}, recorder=run.rec)
            relevant = tr.relevant and tr.confidence >= self.settings.relevance_threshold
            if not relevant and not pointed:
                return []
            signal = None
            if relevant:
                radar = queries[tr.radar] if tr.radar is not None and 0 <= tr.radar < len(queries) else None
                signal = SignalOut(content=SignalContent(**self._content(tr.content)), origin="predefined" if radar else "articulated",
                                   radar=radar)

        # 2. assess: verdicts on held hypotheses, new ones, baselines
        precedents = op.precedents(risk.id)
        run.saw(*precedents)
        baselines_of = {h.id: op.baselines_of(h.id) for h in [*held, *([given_h] if given_h else [])]}
        run.saw(*(b for bs in baselines_of.values() for b in bs))
        def evidence(hid):   # what is already on file for a hypothesis, neutral facts included: they can combine with this episode
            return [f"({t.stance}) {t.signal.summary}" for t in op.attached(hid)
                    if t.status != "rejected" and t.episode_id != ep.id][-self.settings.max_evidence:]

        def held_view(h):
            return {"id": h.id, "claim": h.claim, "status": h.status, "refines": h.refines, "evidence": evidence(h.id),
                    "baselines": [{"id": b.id, "claim": b.claim} for b in baselines_of.get(h.id, [])]}
        out = self.gw.call("episode.assess", {
            "risk_text": risk.text, "context": ctx, "pointed": pointed,
            "episode": {"text": ep.text, "source": ep.source, "credibility": ep.credibility, "occurred": ep.occurred},
            "signal": None if ti.signal is None else ({**ti.signal.content.model_dump(), "note": ti.signal.note} if ti.signal.content else {"note": ti.signal.note}),
            "detected_signal": None if signal is None or signal.origin == "given" else signal.content.model_dump(),
            "given_hypothesis": None if given_claim is None else {
                "claim": given_claim, "evidence": evidence(given_h.id) if given_h else [],
                "baselines": [{"id": b.id, "claim": b.claim} for b in baselines_of.get(given_h.id if given_h else "", [])]},
            "held_hypotheses": [] if given_claim else [held_view(h) for h in held],
            "precedents": [{"claim": known[t.hypothesis_id].claim if t.hypothesis_id in known else "", "rationale": t.rationale,
                            "fit": t.fit, "reach": t.reach} for t in precedents],
            "max_new": 0 if given_claim else self.settings.max_trajectories,
            "max_new_baselines": self.settings.max_baselines}, recorder=run.rec)

        entries: list[_Entry] = []
        if given_claim is not None:
            v = out.given or GivenVerdict(rationale="The model returned no assessment of the given hypothesis.")
            href = (HypothesisRef(origin="given", id=given_h.id, version=given_h.version, claim=given_h.claim, entities=given_h.entities,
                                  status=given_h.status) if given_h else HypothesisRef(origin="given", claim=given_claim, entities=tuple(v.entities)))
            entries.append(_Entry(href, "neutral" if v.stance == "unrelated" else v.stance, v))   # the analyst asked: always answered
        else:
            seen = set()
            for v in out.stored:
                h = known.get(v.hypothesis_id)
                if h is None or v.stance == "unrelated" or h.id in seen:
                    continue
                if v.stance == "supports" and v.confidence < self.settings.match_threshold:
                    continue
                seen.add(h.id)
                entries.append(_Entry(HypothesisRef(origin="predefined", id=h.id, version=h.version, claim=h.claim, entities=h.entities,
                                                    match_confidence=round(v.confidence, 3), status=h.status), v.stance, v))
            by_claim = {_norm(h.claim): h for h in known.values()}
            for v in out.stored:   # a held hypothesis whose mechanism the evidence shows works differently: the corrected one is new
                h, rc = known.get(v.hypothesis_id), (v.revised_claim or "").strip()
                if h is None or not rc or _norm(rc) in by_claim or _norm(rc) in seen:
                    continue
                seen.add(_norm(rc))
                entries.append(_Entry(HypothesisRef(origin="articulated", claim=rc, entities=h.entities, refines=h.id), "supports", v))
            for n in out.new[: self.settings.max_trajectories]:
                claim = (n.claim or "").strip()
                if not claim:
                    continue
                h = by_claim.get(_norm(claim))
                if h is not None:      # restated a held (or rejected) hypothesis: it is that one
                    if h.id in seen:
                        continue
                    seen.add(h.id)
                    entries.append(_Entry(HypothesisRef(origin="predefined", id=h.id, version=h.version, claim=h.claim, entities=h.entities,
                                                        match_confidence=1.0, status=h.status), "supports", n))
                    continue
                if _norm(claim) in seen:
                    continue
                seen.add(_norm(claim))
                entries.append(_Entry(HypothesisRef(origin="articulated", claim=claim, entities=tuple(n.entities),
                                                    refines=n.refines if n.refines in known else None), "supports", n))
            entries = [e for e in entries if include_rejected or e.href.status != "rejected"]
            entries += self._sharpen(op, ep, risk, known, entries, seen, run)
            if not entries:
                return []
        if signal is None:   # pointed at by the analyst, triage did not see a signal: the layer articulates it
            signal = SignalOut(content=SignalContent(**self._content(out.signal)) if out.signal else SignalContent(summary=ep.text[:140]),
                               origin="articulated")
        elif signal.origin == "articulated" and out.signal is not None:
            signal = SignalOut(content=SignalContent(**self._content(out.signal)), origin="articulated")

        # 3. targets and baselines, then one standing call for all of them
        rref = RiskRef(risk_id=risk.id, origin=risk_origin)
        built = [self._target(ep, signal, rref, e, baselines_of.get(e.href.id, []) if e.href.id else []) for e in entries]
        items = [self._attached_items(op, ti, e.href, run) for e in entries]
        anchors = [h.standing if e.href.id and (h := op.hypothesis(e.href.id)) is not None else None for e in entries]
        standings = self._standings(ti, risk, built, items, anchors, run)
        return [AssessmentGroup(target=t, baselines=tuple(b), standing=st) for (t, b), st in zip(built, standings)]

    # ---------------------------------------------------------------- sharpening
    def _sharpen(self, op: OperationalContext, ep: Episode, risk: Risk, known: dict, entries: list, seen: set, run: _Run) -> list:
        """A held hypothesis that this episode supports and that now has `sharpen_after` supporting facts gets one question:
        do the facts together show a more specific mechanism? Live runs showed the model articulates a hypothesis from the
        first facts and then attaches every later, more telling fact to it unchanged ("a chemical leak near Mill Road"
        absorbed the plant's overdue backflow test and the pressure drops); an optional revised claim in the assessment
        was used 2 times in 799 verdicts. A sharper claim becomes a new hypothesis that refines the held one and inherits
        its evidence (spec §5.2b). Hypotheses that already have a refinement are not asked again."""
        refined = {h.refines for h in known.values() if h.refines}
        cands = []
        for e in entries:
            h = known.get(e.href.id or "")
            if h is None or e.stance != "supports" or h.id in refined or h.status == "rejected":
                continue
            facts = [t for t in op.attached(h.id) if t.stance == "supports" and t.status != "rejected" and t.episode_id != ep.id]
            if len(facts) + 1 >= self.settings.sharpen_after:
                cands.append((e, h, facts))

        def ask(c):
            e, h, facts = c
            return self.gw.call("hypothesis.sharpen", {
                "risk_text": risk.text, "claim": h.claim,
                "facts": [{"summary": t.signal.summary, "rationale": t.rationale} for t in facts[-self.settings.max_evidence:]]
                         + [{"new": True, "episode": ep.text, "rationale": e.verdict.rationale}]}, recorder=run.rec)
        out = []
        for (e, h, _), r in zip(cands, _pmap(ask, cands)):
            claim = (r.sharper_claim or "").strip()
            if not claim or _norm(claim) in seen or any(_norm(claim) == _norm(k.claim) for k in known.values()):
                continue
            seen.add(_norm(claim))
            out.append(_Entry(HypothesisRef(origin="articulated", claim=claim, entities=h.entities, refines=h.id), "supports", e.verdict))
        return out

    # ---------------------------------------------------------------- one target
    def _target(self, ep: Episode, signal: SignalOut, rref: RiskRef, e: "_Entry", stored_b: list[Hypothesis]):
        v = e.verdict
        bl = v.baselines if e.stance == "supports" else BaselinesOut()
        by_id = {b.id: b for b in stored_b}
        baselines, known = [], set()
        for a in bl.assessed:
            b = by_id.get(a.baseline_id)
            if b is None or b.id in known:
                continue
            known.add(b.id)
            known.add(_norm(b.claim))
            baselines.append(AssessedTrajectory(
                episode_id=ep.id, role="baseline", signal=signal, risk=rref,
                hypothesis=HypothesisRef(origin="predefined", id=b.id, version=b.version, claim=b.claim, match_confidence=1.0, status=b.status),
                score=round(a.fit, 3), reach=0.0, fit=round(a.fit, 3), rationale=a.rationale, ordinary_outcome=a.ordinary_outcome or b.claim))
        for n in bl.new[: self.settings.max_baselines if not stored_b else 1]:
            if not n.claim.strip() or _norm(n.claim) in known:
                continue
            known.add(_norm(n.claim))
            baselines.append(AssessedTrajectory(
                episode_id=ep.id, role="baseline", signal=signal, risk=rref, hypothesis=HypothesisRef(origin="articulated", claim=n.claim),
                score=round(n.fit, 3), reach=0.0, fit=round(n.fit, 3), rationale=n.rationale, ordinary_outcome=n.ordinary_outcome))
        fit, reach = round(v.fit, 3), round(v.reach, 3)
        if e.stance in ("weakens", "neutral"):   # no support: standing weighs counter-evidence against it, and ignores neutral
            fit, reach, score = round(min(v.fit, 0.3), 3), 0.0, 0.0
        else:
            best_b = max((b.fit for b in baselines), default=0.0)
            tailored = e.href.origin == "articulated" and not e.href.refines   # written from this episode alone
            if tailored and self.settings.new_hypothesis_score == "margin":
                score = score_margin(reach, fit - best_b, signal.content.specificity, ep.credibility)
            else:
                score = score_from(reach, fit, fit - best_b, signal.content.specificity, ep.credibility)
        target = AssessedTrajectory(episode_id=ep.id, role="primary", signal=signal, hypothesis=e.href, risk=rref, score=score,
                                    reach=reach, fit=fit, missing=tuple(v.missing), rationale=v.rationale, stance=e.stance)
        return target, baselines

    # ---------------------------------------------------------------- standing
    def _attached_items(self, op: OperationalContext, ti: TrajectoryInput, href: HypothesisRef, run: _Run) -> list[dict]:
        """Trajectories that count toward a hypothesis's standing. A refinement inherits its parent's: the facts that made
        the layer sharpen the claim support the sharper claim too (spec §5.2b)."""
        pw = self.settings.proposed_weight
        own = op.attached(href.id) if href.id else []
        h = op.hypothesis(href.id) if href.id else None
        parent = href.refines or (h.refines if h is not None else None)
        mine = {t.episode_id for t in own}
        inherited = [t for t in op.attached(parent) if t.episode_id not in mine] if parent else []
        attached = (inherited + list(own))[-self.settings.max_attached:]
        attached = [t for t in attached if t.episode_id != ti.episode.id]   # re-assessing an episode replaces its own trajectory
        run.saw(*attached)
        return [{"id": t.id, "status": t.status, "stance": t.stance, "weight": weight_of(t.status, pw), "score": t.score, "fit": t.fit,
                 "reach": t.reach, "occurred": t.occurred, "summary": t.signal.summary, "rationale": t.rationale} for t in attached]

    def _this(self, ti: TrajectoryInput, target: AssessedTrajectory) -> dict:
        analysts_own = ti.signal is not None and ti.hypothesis is not None
        return {"id": f"this:{ti.episode.id}", "status": "this", "stance": target.stance,
                "weight": 1.0 if analysts_own else self.settings.proposed_weight, "score": target.score, "fit": target.fit,
                "reach": target.reach, "occurred": ti.episode.occurred if isinstance(ti.episode.occurred, str) else None,
                "summary": target.signal.content.summary, "rationale": target.rationale}

    def _standings(self, ti: TrajectoryInput, risk: Risk, built: list, items: list[list[dict]], anchors: list[float | None],
                   run: _Run) -> list[Standing]:
        """Standing of every touched hypothesis, before and after this episode (spec §5.3, §5.2a). Judged together in one
        call; a supporting trajectory never lowers standing, a weakening one never raises it. When the other layer kept the
        standing last reported for a hypothesis (v3.2), that is `before`: standing becomes a chain of updates instead of
        being re-judged from scratch, which drifted by up to 0.25 between consecutive episodes."""
        anchors = [a if a is not None and any(i["weight"] > 0 for i in it) else None for a, it in zip(anchors, items)]
        thises = [self._this(ti, t) for t, _ in built]
        if self.settings.standing == "noisy_or":
            raw = [(noisy_or(it), noisy_or(it + [th]), "noisy-OR of weighted scores") for it, th in zip(items, thises)]   # deterministic: no anchor needed
        else:
            keys = [f"h{i + 1}" for i in range(len(built))]
            fields = ("status", "stance", "weight", "score", "fit", "reach", "occurred", "summary", "rationale")
            out = self.gw.call("standing.judge", {"risk_text": risk.text, "hypotheses": [
                {"key": k, "claim": t.hypothesis.claim, "baselines": sorted({b.hypothesis.claim for b in bl}),
                 "trajectories": [{f: i[f] for f in fields} for i in it], "this": {f: th[f] for f in fields},
                 **({"before": a} if a is not None else {})}
                for k, (t, bl), it, th, a in zip(keys, built, items, thises, anchors)]}, recorder=run.rec)
            got = {x.key: x for x in out.standings}
            raw = []
            for k, it, a in zip(keys, items, anchors):
                x = got.get(k)
                before = a if a is not None else (_clip(x.before) if x is not None and any(i["weight"] > 0 for i in it) else 0.0)
                raw.append((before, _clip(x.value) if x is not None else before, x.rationale if x is not None else "no judgment returned"))
        res, cap = [], self.settings.unconfirmed_cap
        for (before, value, why), (t, _), it, th in zip(raw, built, items, thises):
            if not (th["weight"] >= 1.0 or any(i["status"] == "confirmed" for i in it)):   # the layer's own proposals only
                before, value = min(before, cap), min(value, cap)
            if t.stance == "neutral":    # the episode does not bear on the hypothesis: standing unchanged
                value = before
            elif t.stance == "weakens":    # the judge must lower it; if it did not, damp like the noisy-OR baseline does
                value = value if value < before else round(before * (1 - 0.5 * th["weight"]), 3)
            else:
                value = max(value, before)
            res.append(Standing(value=value, without=before, delta=round(value - before, 3), used=uses(it + [th]), rationale=why))
        return res

    # ================================================================ helpers
    def _radar_queries(self, mission: MissionContext, op: OperationalContext, risk: Risk, run: _Run) -> list[str]:
        """Radars come from the risk and the mission documents only (spec §5.2c). Deriving them from the held hypotheses
        made them look only where the analyst already looks; the hypotheses reach triage by a separate path instead."""
        key = (mission.version, risk.id)
        with self._lock:
            if key in self._radars:
                return self._radars[key]
        out = self.gw.call("radar.derive", {"risk_text": risk.text, "context": self._snippets(mission, risk.text, 6)}, recorder=run.rec)
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
