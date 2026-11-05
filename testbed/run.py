"""Testbed runner.

    python -m testbed.run                         # fake provider, offline, free: proves the plumbing end to end
    python -m testbed.run --live                  # provider/model/key from MISTABRA_* env: the quality measurements
    python -m testbed.run --live --compare        # judged standing (the frame) vs noisy-OR standing (baseline), same inputs
    python -m testbed.run --only miss_ --only conj_01
    python -m testbed.run --live --cache .cache/mistabra.sqlite   # re-runs reuse model outputs

Writes a JSON report and a Markdown summary to testbed/reports/ (or --out). Reported, never gated.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .harness import run_scenario
from .metrics import all_metrics
from .scenario import discover

HERE = Path(__file__).resolve().parent

HEADLINE = [
    ("A1_score", "band_accuracy", "score in the labelled band"), ("A1_score", "hard_negative_auroc", "positives above hard negatives (AUROC)"),
    ("A1_score", "orderings_satisfied", "ordering constraints held"), ("A1_score", "signal_recall", "labelled positives raised a group"),
    ("A1_score", "false_signal_rate", "everyday episodes raised a group"),
    ("A2_lambda", "missing_articulated", "missing hypothesis articulated"), ("A2_lambda", "missing_established", "… and reached standing ≥ 0.5"),
    ("A2_lambda", "control_false_alarm", "controls with an established Lambda hypothesis"), ("A2_lambda", "flag_accuracy", "Lambda flags as expected"),
    ("A3_combination", "jointly_strong_detected", "weak episodes combined into high standing"), ("A3_combination", "control_false_joint", "… in controls"),
    ("A3_combination", "refutation_auroc", "refuted vs supported stated hypotheses (AUROC)"),
    ("A4_matching", "match_accuracy", "matched the right stored hypothesis"), ("A4_matching", "false_new_rate", "articulated a duplicate instead"),
    ("A5_stability", "mean_spread", "score spread across paraphrases"), ("A6_self_reinforcement", "held", "noise did not build a hypothesis"),
    ("R1_ach", "mean_spearman", "ACH: standing vs ground-truth order (Spearman)"), ("R1_ach", "survivor_above_eliminated", "ACH: survivors above eliminated"),
    ("E_expectations", "held", "other step expectations held"),
]


def run(settings_overrides: dict, only: list[str], log=print) -> dict:
    from mistabra_core import Core, Settings
    settings = Settings.from_env(**settings_overrides)
    core = Core(settings)
    scenarios = [s for s in discover() if not only or any(o in s.id for o in only)]
    log(f"testbed provider={core.provider.name} model={settings.model} fast={settings.fast_model} standing={settings.standing} "
        f"scenarios={len(scenarios)} steps={sum(len(s.steps) * (max(1, len(s.operational.hypotheses)) if s.per_hypothesis else 1) for s in scenarios)}")
    results, t0 = [], time.time()
    for i, sc in enumerate(scenarios, 1):
        log(f"[{i}/{len(scenarios)}] {sc.id} ({sc.kind}{', control' if sc.control else ''}) — {len(sc.steps)} steps  [{time.time() - t0:.0f}s]")
        try:
            results.append(run_scenario(core, sc, log=log))
        except Exception as e:  # noqa: BLE001 — a failed scenario is reported, the run continues
            from .harness import ScenarioResult
            results.append(ScenarioResult(id=sc.id, kind=sc.kind, control=sc.control, error=f"{type(e).__name__}: {e}"))
            log(f"  ERROR {type(e).__name__}: {e}")
    metrics = all_metrics([r for r in results if r.error is None])
    return {"settings": {"provider": core.provider.name, "model": settings.model, "fast_model": settings.fast_model, "standing": settings.standing,
                         "proposed_weight": settings.proposed_weight}, "llm": dict(core.gw.stats), "seconds": round(time.time() - t0, 1),
            "metrics": metrics, "scenarios": [asdict(r) for r in results]}


def markdown(reports: list[dict]) -> str:
    out = ["# Mistabra Core — testbed report", ""]
    for r in reports:
        s = r["settings"]
        out.append(f"- **{s['standing']}** standing: provider `{s['provider']}`, model `{s['model']}`, fast `{s['fast_model']}`; "
                   f"{r['llm']['calls']} model calls, {r['llm']['cached']} cached, {r['llm']['errors']} errors, {r['seconds']} s")
    if reports and reports[0]["settings"]["provider"] == "fake":
        out += ["", "> Fake provider: these numbers prove the plumbing, not the method. Run with `--live` for measurements."]
    out += ["", "| Metric | " + " | ".join(r["settings"]["standing"] for r in reports) + " |", "| --- |" + " --- |" * len(reports)]
    for sec, key, label in HEADLINE:
        out.append(f"| {sec.split('_')[0]} {label} | " + " | ".join(str(r["metrics"][sec].get(key)) for r in reports) + " |")
    m = reports[0]["metrics"]
    out += ["", "## Lambda hypotheses (A2)", "", "| Case | Expected | Articulated | Similarity | Standing |", "| --- | --- | --- | --- | --- |"]
    out += [f"| {e['case']} | {e['expected']} | {e['best']} | {e['similarity']} | {e['standing']} |" for e in m["A2_lambda"]["examples"]]
    out += ["", "## ACH cases (R1)", ""]
    for c in m["R1_ach"]["cases"]:
        out.append(f"- {c['case']}: Spearman {c['spearman']}; standing {c['standing']}; expected midpoints {c['expected_mid']}")
    out += ["", "## Failures (first report)", ""]
    for sec, v in m.items():
        for f in v.get("failures", [])[:25]:
            out.append(f"- {sec}: {f}")
    errs = [x for x in reports[0]["scenarios"] if x["error"]]
    if errs:
        out += ["", "## Scenario errors", ""] + [f"- {x['id']}: {x['error']}" for x in errs]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m testbed.run")
    ap.add_argument("--live", action="store_true", help="use the provider configured in MISTABRA_* env (default: fake, offline)")
    ap.add_argument("--standing", choices=["judged", "noisy_or"], default=None)
    ap.add_argument("--compare", action="store_true", help="run judged and noisy_or standing on the same inputs")
    ap.add_argument("--only", action="append", default=[], help="run scenarios whose id contains this text (repeatable)")
    ap.add_argument("--cache", help="SQLite file caching model outputs across runs")
    ap.add_argument("--out", default=str(HERE / "reports"))
    a = ap.parse_args(argv)
    if not a.live:
        os.environ["MISTABRA_FORCE_FAKE"] = "1"
    over = {"cache_path": Path(a.cache)} if a.cache else {}
    modes = ["judged", "noisy_or"] if a.compare else [a.standing or os.environ.get("MISTABRA_STANDING", "judged")]
    reports = [run({**over, "standing": m}, a.only) for m in modes]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"{stamp}-{reports[0]['settings']['provider']}-{'compare' if a.compare else modes[0]}"
    (out / f"{tag}.json").write_text(json.dumps(reports, indent=1, default=str))
    md = markdown(reports)
    (out / f"{tag}.md").write_text(md)
    print("\n" + md)
    print(f"report: {out / tag}.json, .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
