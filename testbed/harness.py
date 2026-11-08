"""The scenario runner.

Each scenario is fed episode by episode through a `Casebook` (mistabra_core/casebook.py), the layer above the core that
keeps the case between calls: every returned group is kept (new hypotheses as proposed, baselines with `baseline_of`,
trajectories as proposed), an identical claim reuses the stored one, and when a hypothesis appears the look-back
re-reads recent episodes that bore on nothing. The runner adds only what a scenario scripts: analyst actions in a
step's `after` (confirm the step's target trajectory, reject / support / reopen a hypothesis), applied through the
Casebook's decision methods, and the records the metrics read.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from mistabra_core.llm.base import LLMOutputError
from mistabra_core import Casebook, Core
from mistabra_core.types import (Assessment, Hypothesis, HypothesisRef, OperationalContext, StoredTrajectory, TrajectoryInput)

from .scenario import Scenario, Step


def norm(s: str) -> str:
    return " ".join(re.findall(r"\w+", (s or "").lower()))


def _resolve(case: Casebook, ref: str, created: dict[str, list[str]]) -> str | None:
    if ref.startswith("step:"):
        got = created.get(ref[5:]) or []
        return got[0] if got else None
    return ref if case.hypothesis(ref) is not None else None


def _after(case: Casebook, step: Step, got, created: dict[str, list[str]]):
    """Apply the analyst actions a scenario step scripts, through the Casebook's decision methods."""
    act = step.after
    if act.get("confirm") and got.trajectories:
        case.confirm(got.trajectories[0], by="scenario")
    for key, fn in (("reject_hypothesis", case.reject_hypothesis), ("support_hypothesis", case.support_hypothesis),
                    ("accept_hypothesis", case.reopen_hypothesis)):
        if act.get(key):
            hid = _resolve(case, str(act[key]), created)
            if hid:
                fn(hid, by="scenario")


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
    lookback: list[dict] = field(default_factory=list)   # earlier episodes re-submitted against hypotheses this step created


@dataclass
class ScenarioResult:
    id: str
    kind: str
    control: bool
    steps: list[StepRec] = field(default_factory=list)
    hypotheses: dict[str, dict] = field(default_factory=dict)     # stored primary id -> {claim, lambda, author, status, standing}
    final: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


LOOKBACK_PER_SCENARIO = 12     # re-assessments per scenario: each costs 2-3 model calls


def _rec(g, hid) -> GroupRec:
    t = g.target
    return GroupRec(risk=t.risk.risk_id, hypothesis_id=hid, origin=t.hypothesis.origin, claim=t.hypothesis.claim,
                    score=t.score, fit=t.fit, reach=t.reach, lambda_signal=t.lambda_signal, lambda_hypothesis=t.lambda_hypothesis,
                    match_confidence=t.hypothesis.match_confidence, standing=g.standing.value, without=g.standing.without,
                    delta=g.standing.delta, baselines=len(g.baselines), inconsistency=g.inconsistency,
                    missing=list(t.missing), stance=t.stance)


def run_scenario(core: Core, sc: Scenario, log=print, lookback: bool = True, window: int = 8) -> ScenarioResult:
    """Feed a scenario to the core through a Casebook (the layer above the core), episode by episode."""
    case = Casebook(core, sc.mission, sc.operational, lookback=lookback and not sc.per_hypothesis, window=window,
                    lookback_budget=LOOKBACK_PER_SCENARIO)
    res = ScenarioResult(id=sc.id, kind=sc.kind, control=sc.control, final=sc.final)
    created: dict[str, list[str]] = {}
    last_standing: dict[str, float] = {}     # the last standing reported per hypothesis (the Casebook clears its own on confirm)
    initial = [h.id for h in sc.operational.hypotheses if h.role == "primary"]
    label_of = {st.episode.id: st.label for st in sc.steps}
    t0 = time.time()
    for i, step in enumerate(sc.steps, 1):
        runs = [step.model_copy(update={"hypothesis": hid}) for hid in initial] if sc.per_hypothesis else [step]
        for st in runs:
            ts = time.perf_counter()
            try:
                got = case.ingest(st.episode, signal=st.signal, hypothesis=st.hypothesis, risk=st.risk, scope=st.scope)
            except LLMOutputError as e:   # one unusable model answer costs this episode, not the whole scenario
                log(f"  [{sc.id}] {st.label}: {e}")
                res.steps.append(StepRec(label=st.label, episode_id=st.episode.id, pattern=0, given_hypothesis=None, groups=[],
                                         expect=st.expect, calls=0, cached=0, seconds=round(time.perf_counter() - ts, 2),
                                         error=f"{type(e).__name__}: {e}"))
                continue
            a = got.assessment
            created.setdefault(st.label, []).extend(got.hypothesis_ids)
            groups = [_rec(g, hid) for g, hid in zip(a.groups, got.hypothesis_ids)]
            looked = [{"episode": label_of.get(lb.episode_id, lb.episode_id), **_rec(lb.group, lb.hypothesis_id).__dict__} for lb in got.lookback]
            for g in groups + [GroupRec(**{k: v for k, v in x.items() if k != "episode"}) for x in looked]:
                last_standing[g.hypothesis_id] = g.standing
            res.steps.append(StepRec(label=st.label, episode_id=st.episode.id, pattern=a.pattern,
                                     given_hypothesis=st.hypothesis if isinstance(st.hypothesis, str) else (st.hypothesis.claim if st.hypothesis else None),
                                     groups=groups, expect=st.expect, calls=len(got.calls), cached=sum(c.cached for c in got.calls),
                                     seconds=round(time.perf_counter() - ts, 2), lookback=looked))
            _after(case, st, got, created)
        if i % 5 == 0 or i == len(sc.steps):
            log(f"  [{sc.id}] step {i}/{len(sc.steps)} ({100 * i // len(sc.steps)}%)  [{time.time() - t0:.1f}s]")
    for h in case.hyps:
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
