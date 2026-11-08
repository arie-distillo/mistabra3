"""Casebook — the layer above the core: it keeps a case and drives the core episode by episode.

The core (`Core.assess`) is stateless by design: it assesses one episode against the contexts it is given and returns
what it found. Something has to keep the case between calls. The Casebook does:

- keeps the operational context: hypotheses (the analyst's and the layer's) and every trajectory returned;
- `ingest(episode)`: calls the core with the current context and stores what comes back;
- look-back: when a hypothesis appears, re-submits recent episodes that bore on nothing against it (a clue that came
  before the hypothesis it supports would otherwise never be seen again);
- records analyst decisions (confirm / reject a trajectory, support / reject / reopen a hypothesis) with who, when
  and why, next to the assessment they are about: the audit trail, and later the data a decision model learns from;
- persists to SQLite when given a path, or lives in memory.

    from mistabra_core import Core, Settings
    from mistabra_core.casebook import Casebook

    case = Casebook(Core(Settings.from_env()), mission, path="case.sqlite")
    h = case.add_hypothesis("The fleet stalled because the routing server was attacked.", risk="fleet")
    result = case.ingest(Episode(id="e1", text="..."))
    case.confirm(result.trajectories[0], by="arie", note="matches the service tickets")
    case.export("feedback.jsonl")    # assessments joined with the analyst's decisions on them
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import Field

from .core import Core
from .llm.prompts import Out, Prompt
from .types import (Assessment, Episode, Hypothesis, HypothesisRef, MissionContext, OperationalContext, SignalDraft,
                    StoredTrajectory, TrajectoryInput)


def _norm(s: str) -> str:
    return " ".join(re.findall(r"\w+", (s or "").lower()))


class _Pick(Out):
    indices: list[int] = Field(default_factory=list)
    why: str = ""


LOOKBACK = Prompt("lookback.select", "1", "You help an analyst re-read earlier reports in the light of a new hypothesis.",
                  "A new hypothesis has just been formed. Below are earlier episodes that, when they arrived, bore on nothing. Which of "
                  "them could bear on this hypothesis, for or against it, now that it exists? Pick only those with a concrete link "
                  "(the same place, system, actor, object or mechanism); most will have none. Return their indices, best first, at "
                  "most `max`. Schema: {\"indices\":[int],\"why\":str}", _Pick, "fast")


@dataclass
class LookBack:
    episode_id: str
    hypothesis_id: str
    group: object                       # the AssessmentGroup the re-assessment returned


@dataclass
class Ingested:
    """What one `ingest` did: the core's assessment, the stored id of each group's hypothesis and trajectory (in the
    order of `assessment.groups`), the hypotheses it created, and what the look-back re-attached."""
    assessment: Assessment
    hypothesis_ids: list[str]
    trajectories: list[str]
    created: list[str]
    lookback: list[LookBack] = field(default_factory=list)
    calls: list = field(default_factory=list)     # model call records of the assessment and of the look-back


class Casebook:
    def __init__(self, core: Core, mission: MissionContext, operational: OperationalContext | None = None, *,
                 path: str | Path | None = None, lookback: bool = True, window: int = 8, per_hypothesis: int = 3,
                 lookback_budget: int | None = None):
        """`lookback_budget`: at most this many re-assessments over the Casebook's life (each costs 2-3 model calls);
        None = no limit. `window`: how many recent unattached episodes are candidates; `per_hypothesis`: how many of them
        a new hypothesis re-reads at most."""
        self.core, self.mission = core, mission
        self.lookback_on, self.window, self.per_hypothesis = lookback, window, per_hypothesis
        self.lookback_budget = lookback_budget
        self.hyps: list[Hypothesis] = list(operational.hypotheses) if operational else []
        self.trs: list[StoredTrajectory] = list(operational.trajectories) if operational else []
        self.unattached: list[dict] = []          # {"episode": Episode json, "risk": str|None}
        self.assessments: list[dict] = []         # audit: one record per core call
        self.decisions: list[dict] = []           # audit: one record per analyst decision
        self.traj_assessment: dict[str, str] = {}  # trajectory id -> assessment id it came from
        self._n = {"H": 0, "N": 0, "B": 0, "T": 0, "A": 0, "D": 0}
        self._lock = threading.Lock()
        self.path = Path(path) if path else None
        if self.path and self.path.exists():
            self._load()

    # ================================================================ the case
    @property
    def op(self) -> OperationalContext:
        return OperationalContext(hypotheses=tuple(self.hyps), trajectories=tuple(self.trs))

    def hypothesis(self, hid: str) -> Hypothesis | None:
        return next((h for h in self.hyps if h.id == hid), None)

    def hypotheses(self, risk: str | None = None, *, include_baselines: bool = False) -> list[Hypothesis]:
        return [h for h in self.hyps if (risk is None or h.risk_id == risk) and (include_baselines or h.role == "primary")]

    def trajectories(self, hid: str | None = None) -> list[StoredTrajectory]:
        return [t for t in self.trs if hid is None or t.hypothesis_id == hid]

    def add_hypothesis(self, claim: str, *, risk: str, entities: tuple[str, ...] = (), author: str = "analyst") -> str:
        """An analyst's own hypothesis (status open)."""
        with self._lock:
            hid = self._next("H")
            self.hyps.append(Hypothesis(id=hid, risk_id=risk, claim=claim, entities=tuple(entities), author=author, role="primary",
                                        status="open"))
            self._save()
        return hid

    # ================================================================ ingest
    def ingest(self, episode: Episode, *, signal: SignalDraft | None = None, hypothesis=None, risk: str | None = None,
               scope: str = "given", detail: str = "full") -> Ingested:
        """Assess one episode against the case as it stands, keep what comes back, then look back if it created a
        hypothesis. Model errors propagate (nothing is stored for a failed call)."""
        before = {h.id for h in self.hyps}
        a = self.core.assess(self.mission, self.op, TrajectoryInput(episode=episode, signal=signal, hypothesis=hypothesis, risk=risk),
                             scope=scope, detail=detail)
        with self._lock:
            hids, tids = self._keep(a, episode)
            if not a.groups and hypothesis is None:
                self.unattached.append({"episode": episode.model_dump(mode="json"), "risk": risk})
            created = [h.id for h in self.hyps if h.id not in before and h.role == "primary"]
            self._save()
        out = Ingested(assessment=a, hypothesis_ids=hids, trajectories=tids, created=created, calls=list(a.provenance.calls))
        if self.lookback_on and created:
            out.lookback = self._look_back([self.hypothesis(h) for h in created if self.hypothesis(h).status != "rejected"], out.calls)
        return out

    def _look_back(self, new: list[Hypothesis], calls: list) -> list[LookBack]:
        found = []
        for h in new:
            if self.lookback_budget is not None and self.lookback_budget <= 0:
                break
            cands = [u for u in self.unattached[-self.window:]]
            if not cands:
                break
            picked = self._pick(h, cands)
            if self.lookback_budget is not None:
                picked = picked[: self.lookback_budget]
                self.lookback_budget -= len(picked)
            for u in picked:
                ep = Episode.model_validate(u["episode"])
                b = self.core.assess(self.mission, self.op, TrajectoryInput(episode=ep, hypothesis=h.id, risk=h.risk_id))
                calls += b.provenance.calls
                kept = [g for g in b.groups if g.target.stance in ("supports", "weakens")
                        and (g.target.hypothesis.id == h.id or g.target.hypothesis.refines == h.id)]   # + a sharpening it caused
                if not kept:
                    continue
                b = b.model_copy(update={"groups": tuple(kept)})
                with self._lock:
                    hids, _ = self._keep(b, ep, lookback_for=h.id)
                    if any(g.target.stance == "supports" for g in kept) and u in self.unattached:
                        self.unattached.remove(u)   # now evidence of that hypothesis (and reaches sharpening with it)
                    self._save()
                found += [LookBack(episode_id=ep.id, hypothesis_id=hid, group=g) for g, hid in zip(b.groups, hids)]
        return found

    def _pick(self, h: Hypothesis, cands: list[dict]) -> list[dict]:
        """One fast-model call picks which earlier episodes to re-assess (a full re-assessment costs 2-3 calls each)."""
        risk_text = next((r.text for r in self.mission.risks if r.id == h.risk_id), "")
        out = self.core.gw.call(LOOKBACK.id, {"risk_text": risk_text, "hypothesis": h.claim, "max": self.per_hypothesis,
                                              "episodes": [{"index": i, "text": c["episode"]["text"]} for i, c in enumerate(cands)]},
                                prompt=LOOKBACK)
        return [cands[i] for i in dict.fromkeys(out.indices) if 0 <= i < len(cands)][: self.per_hypothesis]

    # ================================================================ keeping what the core returned
    def _keep(self, a: Assessment, episode: Episode, lookback_for: str | None = None) -> tuple[list[str], list[str]]:
        aid = self._next("A")
        occ = episode.occurred if isinstance(episode.occurred, str) else None
        hids, tids = [], []
        for g in a.groups:
            t = g.target
            risk_id = t.risk.risk_id
            hid = self._hyp_id(t.hypothesis, risk_id, "primary")
            hids.append(hid)
            i = self._find(hid)
            self.hyps[i] = self.hyps[i].model_copy(update={"standing": g.standing.value})   # anchors the next judgment (§5.2b)
            self.trs = [x for x in self.trs if not (x.episode_id == t.episode_id and x.hypothesis_id == hid)]
            tid = f"{self._next('T')}-{t.episode_id}-{hid}"
            self.trs.append(StoredTrajectory(id=tid, episode_id=t.episode_id, occurred=occ, hypothesis_id=hid, risk_id=risk_id,
                                             role="primary", signal=t.signal.content, score=t.score, fit=t.fit, reach=t.reach,
                                             rationale=t.rationale or "", stance=t.stance, status="proposed"))
            tids.append(tid)
            self.traj_assessment[tid] = aid
            for b in g.baselines:
                bid = self._hyp_id(b.hypothesis, risk_id, "baseline", baseline_of=hid)
                btid = f"{self._next('T')}-{t.episode_id}-{bid}"
                self.trs.append(StoredTrajectory(id=btid, episode_id=t.episode_id, occurred=occ, hypothesis_id=bid, risk_id=risk_id,
                                                 role="baseline", signal=b.signal.content, score=b.score, fit=b.fit, reach=0.0,
                                                 rationale=b.rationale or "", ordinary_outcome=b.ordinary_outcome, status="proposed"))
        self.assessments.append({"id": aid, "at": _now(), "episode": episode.model_dump(mode="json"), "lookback_for": lookback_for,
                                 "assessment": a.model_dump(mode="json", by_alias=True), "trajectories": tids, "hypotheses": hids})
        return hids, tids

    def _hyp_id(self, ref: HypothesisRef, risk_id: str, role: str, baseline_of: str | None = None) -> str:
        if ref.id and self.hypothesis(ref.id) is not None:
            if baseline_of:
                i = self._find(ref.id)
                h = self.hyps[i]
                if baseline_of not in h.baseline_of:
                    self.hyps[i] = h.model_copy(update={"baseline_of": h.baseline_of + (baseline_of,)})
            return ref.id
        same = next((h for h in self.hyps if h.risk_id == risk_id and h.role == role and _norm(h.claim) == _norm(ref.claim)), None)
        if same:
            return same.id
        given = ref.origin == "given"
        hid = self._next("B" if role == "baseline" else ("H" if given else "N"))
        self.hyps.append(Hypothesis(id=hid, risk_id=risk_id, claim=ref.claim, entities=ref.entities, author="analyst" if given else "system",
                                    role=role, is_lambda=(ref.origin == "articulated" and role == "primary"),
                                    status="open" if given else "proposed", baseline_of=(baseline_of,) if baseline_of else (),
                                    refines=ref.refines))
        return hid

    # ================================================================ analyst decisions
    def confirm(self, trajectory_id: str, *, by: str | None = None, note: str = "") -> None:
        """The analyst confirms a trajectory: it now counts fully (weight 1) toward its hypothesis."""
        self._set_trajectory(trajectory_id, "confirmed", by, note)

    def reject(self, trajectory_id: str, *, by: str | None = None, note: str = "") -> None:
        """The analyst rejects a trajectory: it no longer counts."""
        self._set_trajectory(trajectory_id, "rejected", by, note)

    def support_hypothesis(self, hid: str, *, by: str | None = None, note: str = "") -> None:
        self._set_hypothesis(hid, "supported", by, note)

    def reject_hypothesis(self, hid: str, *, by: str | None = None, note: str = "") -> None:
        """Rejected hypotheses are never articulated again."""
        self._set_hypothesis(hid, "rejected", by, note)

    def reopen_hypothesis(self, hid: str, *, by: str | None = None, note: str = "") -> None:
        self._set_hypothesis(hid, "open", by, note)

    def _set_trajectory(self, tid: str, status: str, by, note):
        with self._lock:
            i = next((k for k, t in enumerate(self.trs) if t.id == tid), None)
            if i is None:
                raise KeyError(f"no trajectory {tid!r}")
            t = self.trs[i]
            self.trs[i] = t.model_copy(update={"status": status})
            h = self._find(t.hypothesis_id)
            self.hyps[h] = self.hyps[h].model_copy(update={"standing": None})   # evidence weights changed: re-judge (§5.2b)
            self._decide("trajectory", tid, status, by, note, previous=t.status)

    def _set_hypothesis(self, hid: str, status: str, by, note):
        with self._lock:
            i = self._find(hid)
            h = self.hyps[i]
            self.hyps[i] = h.model_copy(update={"status": status, "version": h.version + 1})
            self._decide("hypothesis", hid, status, by, note, previous=h.status)

    def _decide(self, kind, target, status, by, note, previous):
        self.decisions.append({"id": self._next("D"), "at": _now(), "by": by, "kind": kind, "target": target, "status": status,
                               "previous": previous, "note": note, "assessment": self.traj_assessment.get(target)})
        self._save()

    # ================================================================ export
    def export(self, path: str | Path) -> int:
        """One JSON line per analyst decision on a trajectory: the decision, the trajectory, and the assessment it came
        from (episode and full output). This is the feedback a decision model can later be trained and calibrated on.
        Returns the number of lines written."""
        by_id = {a["id"]: a for a in self.assessments}
        trs = {t.id: t for t in self.trs}
        n = 0
        with Path(path).open("w", encoding="utf-8") as f:
            for d in self.decisions:
                if d["kind"] != "trajectory":
                    continue
                a = by_id.get(d["assessment"] or "")
                t = trs.get(d["target"])
                f.write(json.dumps({"decision": d, "trajectory": t.model_dump(mode="json") if t else None,
                                    "hypothesis": (h.model_dump(mode="json") if t and (h := self.hypothesis(t.hypothesis_id)) else None),
                                    "episode": a["episode"] if a else None, "assessment": a["assessment"] if a else None},
                                   ensure_ascii=False) + "\n")
                n += 1
        return n

    # ================================================================ persistence
    def _next(self, prefix: str) -> str:
        self._n[prefix] += 1
        return f"{prefix}{self._n[prefix]}"

    def _find(self, hid: str) -> int:
        return next(i for i, h in enumerate(self.hyps) if h.id == hid)

    def _state(self) -> dict:
        return {"hypotheses": [h.model_dump(mode="json", by_alias=True) for h in self.hyps],
                "trajectories": [t.model_dump(mode="json") for t in self.trs], "unattached": self.unattached,
                "assessments": self.assessments, "decisions": self.decisions, "traj_assessment": self.traj_assessment,
                "counters": self._n, "mission_version": self.mission.version}

    def _save(self):
        if self.path is None:
            return
        db = sqlite3.connect(self.path)
        try:
            db.execute("CREATE TABLE IF NOT EXISTS casebook (key TEXT PRIMARY KEY, value TEXT)")
            db.executemany("INSERT OR REPLACE INTO casebook VALUES (?, ?)", [(k, json.dumps(v)) for k, v in self._state().items()])
            db.commit()
        finally:
            db.close()

    def _load(self):
        db = sqlite3.connect(self.path)
        try:
            rows = dict(db.execute("SELECT key, value FROM casebook").fetchall())
        except sqlite3.OperationalError:
            return
        finally:
            db.close()
        st = {k: json.loads(v) for k, v in rows.items()}
        self.hyps = [Hypothesis.model_validate(h) for h in st.get("hypotheses", [])]
        self.trs = [StoredTrajectory.model_validate(t) for t in st.get("trajectories", [])]
        self.unattached, self.assessments = st.get("unattached", []), st.get("assessments", [])
        self.decisions, self.traj_assessment = st.get("decisions", []), st.get("traj_assessment", {})
        self._n.update(st.get("counters", {}))


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
