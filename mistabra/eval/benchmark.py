"""Benchmark runner: loads the test data into a Mistabra instance, then measures — reported, never gated.

Measures: signal recall on labelled positives, false signals on easy negatives, band accuracy and mean absolute
error against labels, pairwise ordering constraints, Lambda detection, missing-information flags, accumulation order.
"""
from __future__ import annotations
import json, time
from pathlib import Path


def load_testdata(m, testdata: Path | str, actor: str = "bench", log=print) -> dict:
    td = Path(testdata)
    for p in sorted((td / "context").glob("*.md")):
        m.context.create_file(p, actor)
    risks = {}
    for r in json.loads((td / "risks.json").read_text()):
        risk = m.risks.create(r["text"], actor, open=r.get("open", False))
        n = len(m.radars.derive(risk.id, actor))
        risks[r["key"]] = risk.id
        log(f"[bench] risk {r['key']}: {n} radars derived")
    creds = {"Field reports": 0.8, "Community hotline": 0.6, "Social media monitoring": 0.5, "Police liaison bulletin": 0.9}
    for name, c in creds.items():
        m.sources.create(name, actor, credibility=c)
    episodes, t0 = {}, time.time()
    recs = [json.loads(l) for l in (td / "episodes.jsonl").read_text().splitlines() if l.strip()]
    for i, r in enumerate(recs, 1):
        occ = tuple(r["occurred"]) if isinstance(r.get("occurred"), list) else r.get("occurred")
        ep = m.episodes.create(r["text"], actor, source=r["source"], occurred=occ, metadata={"seed": r["seed"]})
        episodes[r["seed"]] = ep.id
        if i % 5 == 0 or i == len(recs):
            log(f"[bench] episodes {i}/{len(recs)} ({100 * i // len(recs)}%)  [{time.time() - t0:.1f}s]")
    return {"risks": risks, "episodes": episodes}


def run_benchmark(m, testdata: Path | str, log=print) -> dict:
    td = Path(testdata)
    ids = load_testdata(m, td, log=log)
    labels = json.loads((td / "benchmark" / "labels.json").read_text())
    bands, risk_id = labels["bands"], ids["risks"]["attack"]
    rows = {r["id"]: r for r in m.store.all("SELECT * FROM signal WHERE risk_id=?", (risk_id,))}
    by_ep = {}
    for r in rows.values():
        by_ep.setdefault(r["episode_id"], []).append(r)

    def band_of(x):
        return next((b for b, (lo, hi) in bands.items() if lo <= x <= hi and (x < hi or b == "strong")), "none")

    res = {"recall": [], "false_signals": [], "band_hits": [], "abs_err": [], "orderings": [], "lambda": None, "missing": [], "per_episode": {}}
    strength = {}
    for seed, lab in labels["labels"].items():
        eid = ids["episodes"][seed]
        sigs = by_ep.get(eid, [])
        best = max((s["strength_system"] or 0.0 for s in sigs), default=None)
        strength[seed] = best
        res["per_episode"][seed] = {"signals": len(sigs), "strength": best, "via": [s["via"] for s in sigs], "label": lab.get("band")}
        if lab.get("expect_signal") is True:
            res["recall"].append(bool(sigs))
        if lab.get("expect_signal") is False:
            res["false_signals"].append(bool(sigs))
        if lab.get("band") is not None:
            x = best if best is not None else 0.0
            res["band_hits"].append(band_of(x) == lab["band"])
            lo, hi = bands[lab["band"]]
            res["abs_err"].append(0.0 if lo <= x <= hi else min(abs(x - lo), abs(x - hi)))
        if lab.get("lambda"):
            res["lambda"] = any(s["via"] == "lambda" for s in sigs)
        if lab.get("expect_missing"):
            res["missing"].append(any(json.loads(s["missing_json"]) for s in sigs) if sigs else None)
    for chain in labels["orderings"]:
        vals = [strength.get(s) or 0.0 for s in chain]
        res["orderings"].append(all(a < b for a, b in zip(vals, vals[1:])))
    # accumulation order
    cov = m.hypotheses.coverage(risk_id)
    res["accumulation"] = {"hypotheses_supported": len(cov.items), "path_reach": cov.path_reach}
    # lambda precision: lambda signals that are labelled positive / all lambda signals
    lam = [s for s in rows.values() if s["via"] == "lambda"]
    seed_of = {v: k for k, v in ids["episodes"].items()}
    lam_pos = sum(1 for s in lam if labels["labels"].get(seed_of.get(s["episode_id"]), {}).get("expect_signal"))
    summary = {
        "signal_recall": _ratio(res["recall"]), "false_signal_rate": _ratio(res["false_signals"]),
        "band_accuracy": _ratio(res["band_hits"]), "mean_abs_band_error": round(sum(res["abs_err"]) / len(res["abs_err"]), 3) if res["abs_err"] else None,
        "orderings_satisfied": _ratio(res["orderings"]), "lambda_detected": res["lambda"],
        "lambda_precision": round(lam_pos / len(lam), 2) if lam else None, "lambda_signals": len(lam),
        "missing_flagged": _ratio([x for x in res["missing"] if x is not None]), "accumulation": res["accumulation"],
        "llm": m.stats()["llm"], "provider": m.provider.name, "model": m.settings.model,
    }
    for k, v in summary.items():
        log(f"[bench] {k}: {v}")
    return {"summary": summary, "per_episode": res["per_episode"], "ids": ids}


def _ratio(xs):
    return round(sum(1 for x in xs if x) / len(xs), 2) if xs else None
