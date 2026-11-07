"""A stand-in for the other layer, and the scenario runner.

The core layer reads both contexts and writes neither (spec v3 §2). Something has to keep what it returns between
calls — that is `OtherLayer`, the smallest version of the other layer that makes sequences meaningful:

- every returned group is kept, weak ones included (spec §5.5): new hypotheses are stored with status `proposed`
  (analyst drafts as `open`), new baselines with `baseline_of` the target, every trajectory with status `proposed`;
- a claim identical (after normalisation) to a stored one reuses it — the other layer's own exact-match dedupe;
  anything looser is the core layer's matching, which is what A4 measures;
- analyst actions in a step's `after` are applied: confirm the step's target trajectory, reject or support a hypothesis.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from mistabra_core.llm.base import LLMOutputError
from mistabra_core import Core
from mistabra_core.types import (Assessment, Hypothesis, HypothesisRef, OperationalContext, StoredTrajectory, TrajectoryInput)

from .scenario import Scenario, Step


def norm(s: str) -> str:
    return " ".join(re.findall(r"\w+", (s or "").lower()))


class OtherLayer:
    def __init__(self, scenario: Scenario):
        self.hyps: list[Hypothesis] = list(scenario.operational.hypotheses)
        self.trs: list[StoredTrajectory] = list(scenario.operational.trajectories)
        self.n = 0

    @property
    def op(self) -> OperationalContext:
        return OperationalContext(hypotheses=tuple(self.hyps), trajectories=tuple(self.trs))

    def _find(self, hid: str) -> int:
        return next(i for i, h in enumerate(self.hyps) if h.id == hid)

    def _hyp_id(self, ref: HypothesisRef, risk_id: str, role: str, baseline_of: str | None = None) -> str:
        if ref.id:
            if baseline_of:
                i = self._find(ref.id)
                h = self.hyps[i]
                if baseline_of not in h.baseline_of:
                    self.hyps[i] = h.model_copy(update={"baseline_of": h.baseline_of + (baseline_of,)})
            return ref.id
        same = next((h for h in self.hyps if h.risk_id == risk_id and h.role == role and norm(h.claim) == norm(ref.claim)), None)
        if same:
            return self._hyp_id(ref.model_copy(update={"id": same.id}), risk_id, role, baseline_of)
        self.n += 1
        hid = f"{'B' if role == 'baseline' else 'N'}{self.n}"
        given = ref.origin == "given"
        self.hyps.append(Hypothesis(id=hid, risk_id=risk_id, claim=ref.claim, entities=ref.entities, author="analyst" if given else "system",
                                    role=role, is_lambda=(ref.origin == "articulated" and role == "primary"),
                                    status="open" if given else "proposed", baseline_of=(baseline_of,) if baseline_of else (),
                                    refines=ref.refines))
        return hid

    def keep(self, a: Assessment, step: Step) -> list[str]:
        """Store what the assessment returned. Returns the stored id of each group's target hypothesis."""
        ids = []
        for g in a.groups:
            t = g.target
            risk_id = t.risk.risk_id
            hid = self._hyp_id(t.hypothesis, risk_id, "primary")
            ids.append(hid)
            i = self._find(hid)
            self.hyps[i] = self.hyps[i].model_copy(update={"standing": g.standing.value})   # kept to anchor the next judgment
            self.trs = [x for x in self.trs if not (x.episode_id == t.episode_id and x.hypothesis_id == hid)]
            occ = step.episode.occurred if isinstance(step.episode.occurred, str) else None
            self.trs.append(StoredTrajectory(id=f"T{len(self.trs) + 1}-{t.episode_id}-{hid}", episode_id=t.episode_id, occurred=occ,
                                             hypothesis_id=hid, risk_id=risk_id, role="primary", signal=t.signal.content, score=t.score,
                                             fit=t.fit, reach=t.reach, rationale=t.rationale or "", stance=t.stance, status="proposed"))
            for b in g.baselines:
                bid = self._hyp_id(b.hypothesis, risk_id, "baseline", baseline_of=hid)
                self.trs.append(StoredTrajectory(id=f"T{len(self.trs) + 1}-{t.episode_id}-{bid}", episode_id=t.episode_id, occurred=occ,
                                                 hypothesis_id=bid, risk_id=risk_id, role="baseline", signal=b.signal.content, score=b.score,
                                                 fit=b.fit, reach=0.0, rationale=b.rationale or "", ordinary_outcome=b.ordinary_outcome,
                                                 status="proposed"))
        return ids

    def resolve(self, ref: str, created: dict[str, list[str]]) -> str | None:
        if ref.startswith("step:"):
            got = created.get(ref[5:]) or []
            return got[0] if got else None
        return ref if any(h.id == ref for h in self.hyps) else None

    def after(self, step: Step, ids: list[str], created: dict[str, list[str]]):
        act = step.after
        if act.get("confirm") and ids:
            self.trs = [x.model_copy(update={"status": "confirmed"}) if x.episode_id == step.episode.id and x.hypothesis_id == ids[0]
                        and x.role == "primary" else x for x in self.trs]
            i = self._find(ids[0])
            self.hyps[i] = self.hyps[i].model_copy(update={"standing": None})   # evidence weights changed: re-judge from scratch
        for key, status in (("reject_hypothesis", "rejected"), ("support_hypothesis", "supported"), ("accept_hypothesis", "open")):
            if act.get(key):
                hid = self.resolve(str(act[key]), created)
                if hid:
                    i = self._find(hid)
                    self.hyps[i] = self.hyps[i].model_copy(update={"status": status, "version": self.hyps[i].version + 1})


# ---------------------------------------------------------------- runner
@dataclass
class GroupRec:
    risk: str
    hypothesis_id: str
    origin: str
    claim: str
    score: float
    fit: float
    reach: float
    lambda_signal: bool
    lambda_hypothesis: bool
    match_confidence: float | None
    standing: float
    without: float
    delta: float
    baselines: int
    inconsistency: str | None
    missing: list[str] = field(default_factory=list)
    stance: str = "supports"


@dataclass
class StepRec:
    label: str
    episode_id: str
    pattern: int
    given_hypothesis: str | None
    groups: list[GroupRec]
    expect: dict
    calls: int
    cached: int
    seconds: float
    error: str | None = None          # the layer failed on this episode; the scenario goes on without it


@dataclass
class ScenarioResult:
    id: str
    kind: str
    control: bool
    steps: list[StepRec] = field(default_factory=list)
    hypotheses: dict[str, dict] = field(default_factory=dict)     # stored primary id -> {claim, lambda, author, status, standing}
    final: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def run_scenario(core: Core, sc: Scenario, log=print) -> ScenarioResult:
    layer = OtherLayer(sc)
    res = ScenarioResult(id=sc.id, kind=sc.kind, control=sc.control, final=sc.final)
    created: dict[str, list[str]] = {}
    last_standing: dict[str, float] = {}
    initial = [h.id for h in sc.operational.hypotheses if h.role == "primary"]
    t0 = time.time()
    for i, step in enumerate(sc.steps, 1):
        runs = [step.model_copy(update={"hypothesis": hid}) for hid in initial] if sc.per_hypothesis else [step]
        for st in runs:
            ts = time.perf_counter()
            try:
                a = core.assess(sc.mission, layer.op, TrajectoryInput(episode=st.episode, signal=st.signal, hypothesis=st.hypothesis, risk=st.risk),
                                scope=st.scope)
            except LLMOutputError as e:   # one unusable model answer costs this episode, not the whole scenario
                log(f"  [{sc.id}] {st.label}: {e}")
                res.steps.append(StepRec(label=st.label, episode_id=st.episode.id, pattern=0, given_hypothesis=None, groups=[],
                                         expect=st.expect, calls=0, cached=0, seconds=round(time.perf_counter() - ts, 2),
                                         error=f"{type(e).__name__}: {e}"))
                continue
            ids = layer.keep(a, st)
            created.setdefault(st.label, []).extend(ids)
            groups = []
            for g, hid in zip(a.groups, ids):
                t = g.target
                groups.append(GroupRec(risk=t.risk.risk_id, hypothesis_id=hid, origin=t.hypothesis.origin, claim=t.hypothesis.claim,
                                       score=t.score, fit=t.fit, reach=t.reach, lambda_signal=t.lambda_signal, lambda_hypothesis=t.lambda_hypothesis,
                                       match_confidence=t.hypothesis.match_confidence, standing=g.standing.value, without=g.standing.without,
                                       delta=g.standing.delta, baselines=len(g.baselines), inconsistency=g.inconsistency,
                                       missing=list(t.missing), stance=t.stance))
                last_standing[hid] = g.standing.value
            calls = a.provenance.calls
            res.steps.append(StepRec(label=st.label, episode_id=st.episode.id, pattern=a.pattern,
                                     given_hypothesis=st.hypothesis if isinstance(st.hypothesis, str) else (st.hypothesis.claim if st.hypothesis else None),
                                     groups=groups, expect=st.expect, calls=len(calls), cached=sum(c.cached for c in calls),
                                     seconds=round(time.perf_counter() - ts, 2)))
            layer.after(st, ids, created)
        if i % 5 == 0 or i == len(sc.steps):
            log(f"  [{sc.id}] step {i}/{len(sc.steps)} ({100 * i // len(sc.steps)}%)  [{time.time() - t0:.1f}s]")
    for h in layer.hyps:
        if h.role == "primary":
            res.hypotheses[h.id] = {"claim": h.claim, "lambda": h.is_lambda, "author": h.author, "status": h.status, "refines": h.refines,
                                    "risk": h.risk_id, "standing": last_standing.get(h.id)}
    res.final = {**sc.final, "_created": created}
    if sc.final.get("jointly_strong"):
        # "each alone is weak" means alone: every episode assessed against the starting context only, not after the others
        # have shaped the hypothesis it attaches to (a hypothesis articulated from the first episode can already name the
        # second's condition, and then the second scores high against it).
        solo = {}
        for st in sc.steps:
            a = core.assess(sc.mission, sc.operational, TrajectoryInput(episode=st.episode, signal=st.signal, hypothesis=st.hypothesis,
                                                                         risk=st.risk), scope=st.scope)
            solo[st.label] = max((g.target.score for g in a.groups if g.target.stance == "supports"), default=0.0)
        res.final["_solo_scores"] = solo
    return res
