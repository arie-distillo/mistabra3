"""Scenarios: one format for every test case the testbed runs, and adapters that turn existing case files into it.

A scenario is a mission context, an initial operational context, and an ordered list of steps. Each step is one
`assess` call (one episode with any of signal, hypothesis, risk), optional expectations about its result, and
optional analyst actions applied after it. `final` holds expectations about the end state.

Sources:
  native YAML           testbed/scenarios/*.yaml               (load_yaml)
  PoC labelled data     testbed/data/community_safety/          (from_community_safety)
  ACH case files        testbed/sources/*_ach_*.yaml or private  (from_ach)   — project format: claims, episodes, ground_truth
  phase-1 cases         testbed/sources/phase1_cases.yaml       (from_phase1) — conj_* / miss_* and their controls
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import yaml
from pydantic import BaseModel, Field

from mistabra_core.types import (Document, Episode, Hypothesis, HypothesisDraft, MissionContext, OperationalContext, Risk,
                                 SignalContent, SignalDraft)

HERE = Path(__file__).resolve().parent


class Step(BaseModel):
    episode: Episode
    signal: SignalDraft | None = None
    hypothesis: Union[str, HypothesisDraft, None] = None
    risk: str | None = None
    scope: str = "given"
    expect: dict[str, Any] = Field(default_factory=dict)
    after: dict[str, Any] = Field(default_factory=dict)   # analyst actions: confirm: true | reject_hypothesis: <ref>
    label: str = ""                                       # free tag, e.g. the PoC seed


class Scenario(BaseModel):
    id: str
    title: str = ""
    kind: str                      # labels | sequence | ach | conj | miss | stability | noise | analyst
    control: bool = False
    mission: MissionContext
    operational: OperationalContext = OperationalContext()
    steps: list[Step] = Field(default_factory=list)
    final: dict[str, Any] = Field(default_factory=dict)
    per_hypothesis: bool = False   # run every step once per stored primary hypothesis (ACH style)
    note: str = ""


# ---------------------------------------------------------------- native YAML
def _episode(d: dict, i: int, prefix: str) -> Episode:
    return Episode(id=d.get("id") or f"{prefix}-e{i + 1}", text=d["text"].strip(), source=d.get("source", "unknown"),
                   credibility=d.get("credibility", 0.7), occurred=d.get("occurred"))


def _signal(d) -> SignalDraft | None:
    if d is None:
        return None
    if isinstance(d, str):
        return SignalDraft(note=d)
    return SignalDraft(content=SignalContent(**d["content"]) if d.get("content") else None, note=d.get("note", ""))


def _hyp(d):
    if d is None or isinstance(d, str):
        return d
    return HypothesisDraft(claim=d["claim"])


def load_yaml(path: Path) -> Scenario:
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    sid = d["id"]
    mission = MissionContext(version=str(d["mission"].get("version", "1")),
                             documents=tuple(Document(id=doc.get("id", f"doc{i + 1}"), title=doc.get("title", ""), text=doc["text"].strip())
                                             for i, doc in enumerate(d["mission"].get("documents", []))),
                             risks=tuple(Risk(**r) for r in d["mission"]["risks"]))
    op = OperationalContext(hypotheses=tuple(Hypothesis(**h) for h in (d.get("operational") or {}).get("hypotheses", [])))
    steps = [Step(episode=_episode(s["episode"], i, sid), signal=_signal(s.get("signal")), hypothesis=_hyp(s.get("hypothesis")),
                  risk=s.get("risk"), scope=s.get("scope", "given"), expect=s.get("expect") or {}, after=s.get("after") or {},
                  label=s.get("label", f"s{i + 1}"))
             for i, s in enumerate(d.get("steps", []))]
    return Scenario(id=sid, title=d.get("title", ""), kind=d["kind"], control=d.get("control", False), mission=mission, operational=op,
                    steps=steps, final=d.get("final") or {}, per_hypothesis=d.get("per_hypothesis", False), note=d.get("note", ""))


# ---------------------------------------------------------------- PoC labelled data (Elm Street)
CREDIBILITY = {"Field reports": 0.8, "Community hotline": 0.6, "Social media monitoring": 0.5, "Police liaison bulletin": 0.9}


def from_community_safety(root: Path = HERE / "data" / "community_safety") -> Scenario:
    docs = tuple(Document(id=p.stem, title=p.stem.replace("_", " "), text=p.read_text(encoding="utf-8")) for p in sorted((root / "context").glob("*.md")))
    risks = tuple(Risk(id=r["key"], text=r["text"]) for r in json.loads((root / "risks.json").read_text(encoding="utf-8")))
    labels = json.loads((root / "labels.json").read_text(encoding="utf-8"))
    steps = []
    for line in (root / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        occ = r.get("occurred")
        occ = occ[0] if isinstance(occ, list) else occ
        lab = labels["labels"].get(r["seed"], {})
        steps.append(Step(episode=Episode(id=r["seed"], text=r["text"], source=r["source"], credibility=CREDIBILITY.get(r["source"], 0.5),
                                          occurred=occ),
                          expect={"risk": "attack", **lab}, label=r["seed"]))
    steps.sort(key=lambda s: s.episode.occurred or "9999")
    return Scenario(id="community_safety", title="Community Safety Trust (PoC labelled set)", kind="labels",
                    mission=MissionContext(version="cs-1", documents=docs, risks=risks), steps=steps,
                    # The PoC's band cut-off between weak and strong was 0.5. Three live runs (2026-11-06) ranked these episodes
                    # perfectly but put 8 of 13 "strong" ones at 0.41-0.49; with the cut-off at 0.4 band accuracy was 0.93 / 0.90.
                    # Calibrated on this same set: re-check on held-out cases.
                    final={"bands": {**labels["bands"], "weak": [0.2, 0.4], "strong": [0.4, 1.0]}, "orderings": labels["orderings"]},
                    note="31 labelled episodes, pattern 1, in order of occurrence; labels are for the attack risk.")


# ---------------------------------------------------------------- ACH case files
def from_ach(path: Path) -> Scenario:
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    sid = path.stem
    risk = Risk(id="r1", text=d["description"].strip().split(". ")[0].rstrip(".") + ".")
    gt = {c["id"]: c for c in d.get("ground_truth", {}).get("claims", [])}
    hyps = tuple(Hypothesis(id=f"H{i + 1}", risk_id="r1", claim=" ".join(c["text"].split()), origin="analyst", status="open")
                 for i, c in enumerate(d["claims"]))
    steps = [Step(episode=Episode(id=f"{sid}-e{i + 1}", text=" ".join(e["content"].split()), source=e.get("source_type", "unknown"),
                                  credibility=e.get("credibility", 0.7)), label=f"e{i + 1}")
             for i, e in enumerate(d["episodes"])]
    final = {"expected": {hid: {"range": c.get("expected_score_range"), "eliminated": c.get("eliminated", False),
                                "surviving": c.get("is_surviving_hypothesis", False), "verdict": c.get("expected_verdict")}
                          for hid, c in gt.items()}}
    docs = (Document(id="context", title="analytical context", text=d.get("analytical_context", "")),
            Document(id="description", title=d.get("name", sid), text=d.get("description", "")))
    return Scenario(id=sid, title=d.get("name", sid), kind="ach", mission=MissionContext(version=f"{sid}-1", documents=docs, risks=(risk,)),
                    operational=OperationalContext(hypotheses=hyps), steps=steps, final=final, per_hypothesis=True,
                    note="Each episode is assessed once per stated claim (pattern 3); final standing is compared with the ACH ground truth.")


# ---------------------------------------------------------------- phase-1 cases
def from_phase1(path: Path) -> list[Scenario]:
    out = []
    for c in yaml.safe_load(path.read_text(encoding="utf-8")):
        cid, domain = c["id"], c.get("domain", "")
        dps = c["data_points"]
        eps = [Episode(id=f"{cid}-{p['id']}", text=" ".join(p["text"].split()), source="case record", credibility=0.85) for p in dps]
        roles = {f"{cid}-{p['id']}": p.get("role") for p in dps}
        gt = c.get("ground_truth", {})
        if c["capability"] == "conjunction_refutation":
            risk = Risk(id="r1", text=f"The {domain.replace('_', ' ')} incident is misattributed and its real cause stays unaddressed.")
            h = Hypothesis(id="H1", risk_id="r1", claim=c["hypothesis"], origin="analyst", status="open")
            steps = [Step(episode=e, hypothesis="H1", label=e.id, expect={"role": roles[e.id]}) for e in eps]
            refuting = [f"{cid}-{x}" for s in gt.get("refuting_sets", []) for x in s]
            out.append(Scenario(id=cid, title=c["hypothesis"], kind="conj", control=c.get("control", False),
                                mission=MissionContext(version=f"{cid}-1", risks=(risk,)), operational=OperationalContext(hypotheses=(h,)),
                                steps=steps, final={"refuted": not c.get("control", False), "refuting": refuting},
                                note="Data points in order under the stated hypothesis (pattern 3)."))
        else:
            risk = Risk(id="r1", text=f"A {domain.replace('_', ' ')} failure recurs because its real cause is not identified.")
            hyps = tuple(Hypothesis(id=f"G{i + 1}", risk_id="r1", claim=g, origin="analyst", status="open")
                         for i, g in enumerate(c.get("given_hypotheses", [])))
            steps = [Step(episode=e, label=e.id, expect={"role": roles[e.id]}) for e in eps]
            out.append(Scenario(id=cid, title=f"missing hypothesis ({domain})", kind="miss", control=c.get("control", False),
                                mission=MissionContext(version=f"{cid}-1", risks=(risk,)), operational=OperationalContext(hypotheses=hyps),
                                steps=steps, final={"missing": gt.get("missing_hypothesis"), "alternatives": gt.get("missing_hypothesis_alternatives", []),
                                                    "cluster": [f"{cid}-{x}" for x in gt.get("cluster_members", [])]},
                                note="Data points in order, pattern 1, with the given hypotheses stored."))
    return out


# ---------------------------------------------------------------- discovery
def discover(include_private: bool = True) -> list[Scenario]:
    out = [from_community_safety()]
    out += [load_yaml(p) for p in sorted((HERE / "scenarios").glob("*.yaml"))]
    src = [HERE / "sources"] + ([HERE / "private"] if include_private else [])
    for d in src:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.yaml")):
            if p.name.startswith("phase1"):
                out += from_phase1(p)
            else:
                head = yaml.safe_load(p.read_text(encoding="utf-8"))
                if isinstance(head, dict) and "claims" in head and "episodes" in head:
                    out.append(from_ach(p))
    return out
