"""Testbed runner.

    python -m testbed.run                         # fake provider, offline, free: proves the plumbing end to end
    python -m testbed.run --live                  # provider/model/key from MISTABRA_* env: the quality measurements
    python -m testbed.run --live --compare        # judged standing (the frame) vs noisy-OR standing (baseline), same inputs
    python -m testbed.run --only miss_ --only conj_01
    python -m testbed.run --live --cache .cache/mistabra.sqlite   # re-runs reuse model outputs
    python -m testbed.run --live --jobs 8         # scenarios in parallel (default 4); steps within a scenario stay in order

Writes a JSON report and a Markdown summary to testbed/reports/ (or --out). Reported, never gated.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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


def run(settings_overrides: dict, only: list[str], log=print, jobs: int = 4) -> dict:
    from mistabra_core import Core, Settings
    from mistabra_core.llm import LLMAuthError
    settings = Settings.from_env(**settings_overrides)
    core = Core(settings)
    scenarios = [s for s in discover() if not only or any(o in s.id for o in only)]
    from mistabra_core.settings import describe_key
    log(f"testbed provider={settings.provider} ({describe_key(settings.provider) if settings.uses_real_llm else 'offline'}) model={settings.model} fast={settings.fast_model} reasoning={settings.reasoning or 'default'} standing={settings.standing} "
        f"scenarios={len(scenarios)} steps={sum(len(s.steps) * (max(1, len(s.operational.hypotheses)) if s.per_hypothesis else 1) for s in scenarios)}")
    t0 = time.time()
    stop = []

    def one(i_sc):
        i, sc = i_sc
        if stop:
            return None
        log(f"[{i}/{len(scenarios)}] {sc.id} ({sc.kind}{', control' if sc.control else ''}) — {len(sc.steps)} steps  [{time.time() - t0:.0f}s]")
        try:
            r = run_scenario(core, sc, log=log)
            log(f"[{i}/{len(scenarios)}] {sc.id} done  [{time.time() - t0:.0f}s]")
            return r
        except LLMAuthError as e:
            stop.append(e)
            return None
        except Exception as e:  # noqa: BLE001 — a failed scenario is reported, the run continues
            from .harness import ScenarioResult
            log(f"  [{sc.id}] ERROR {type(e).__name__}: {e}")
            return ScenarioResult(id=sc.id, kind=sc.kind, control=sc.control, error=f"{type(e).__name__}: {e}")

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:   # scenarios are independent; each keeps its steps in order
        results = list(ex.map(one, enumerate(scenarios, 1)))
    if stop:
        log(f"STOPPED: the provider rejected the API key — {stop[0]}\n"
            f"Check the key in .env (MISTABRA_API_KEY, or the provider's own variable) belongs to provider '{settings.provider}'.")
        raise SystemExit(3)
    metrics = all_metrics([r for r in results if r.error is None])
    return {"settings": {"provider": core.provider.name, "model": settings.model, "fast_model": settings.fast_model, "standing": settings.standing,
                         "proposed_weight": settings.proposed_weight}, "llm": dict(core.gw.stats), "seconds": round(time.time() - t0, 1),
            "metrics": metrics, "scenarios": [asdict(r) for r in results]}


def markdown(reports: list[dict]) -> str:
    out = ["# Mistabra Core — testbed report", ""]
    for r in reports:
        s = r["settings"]
        out.append(f"- **{s['standing']}** standing: provider `{s['provider']}`, model `{s['model']}`, fast `{s['fast_model']}`; "
                   f"{r['llm']['calls']} model calls, {r['llm']['cached']} cached, {r['llm'].get('retries', 0)} validation retries, "
                   f"{r['llm']['errors']} errors, {r['seconds']} s")
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
    ap.add_argument("--jobs", type=int, default=4, help="scenarios run in parallel (default 4)")
    ap.add_argument("--out", default=str(HERE / "reports"))
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):   # Windows consoles default to cp1252; never crash on a character
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if not a.live:
        os.environ["MISTABRA_FORCE_FAKE"] = "1"
    else:
        from mistabra_core import Settings
        s = Settings.from_env()
        if not s.uses_real_llm:
            print("ERROR: --live needs a model. Set MISTABRA_PROVIDER (anthropic | openrouter | openai), MISTABRA_MODEL, and the key in "
                  "MISTABRA_API_KEY or the provider's own variable (ANTHROPIC_API_KEY | OPENROUTER_API_KEY | OPENAI_API_KEY; another "
                  "service's key is never used), or MISTABRA_BASE_URL for a local endpoint; MISTABRA_FORCE_FAKE must be unset. "
                  f"Now: provider={s.provider!r}, key={'set' if s.api_key else 'missing'}, base_url={s.base_url!r}, "
                  f"force_fake={s.force_fake}.", file=sys.stderr)
            return 2
    over = {"cache_path": Path(a.cache)} if a.cache else {}
    over["failure_log"] = Path(a.out) / "failures.jsonl"   # every failed model call, with prompt and raw output
    stamp = time.strftime("%Y%m%d-%H%M%S")
    calls = Path(a.out) / f"{stamp}-calls.jsonl"
    if a.live:
        over["call_log"] = calls                           # every model call, with prompt and output: the "why" behind the report
    modes = ["judged", "noisy_or"] if a.compare else [a.standing or os.environ.get("MISTABRA_STANDING", "judged")]
    reports = [run({**over, "standing": m}, a.only, jobs=a.jobs) for m in modes]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{stamp}-{reports[0]['settings']['provider']}-{'compare' if a.compare else modes[0]}"
    (out / f"{tag}.json").write_text(json.dumps(reports, indent=1, default=str), encoding="utf-8")
    md = markdown(reports)
    (out / f"{tag}.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"report: {out / tag}.json, .md")
    if calls.exists():
        print(f"model calls (prompt and output of each): {calls}")
    if (out / "failures.jsonl").exists():
        print(f"failed model calls (prompt, raw output, error): {out / 'failures.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
