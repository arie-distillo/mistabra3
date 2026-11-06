"""Quality measurements over scenario results (spec v3 §8).

A1  score: band accuracy, ordering constraints, hard negatives low, recall / false signals     (labelled steps)
A2  Lambda: missing hypothesis articulated and established; none in controls; Lambda flags as expected
A3  combination: jointly-strong sequences reach high standing from weak scores; refuting sets lower standing
A4  matching: articulated claims that should have matched a stored hypothesis
A5  stability: score spread across paraphrases of one episode
A6  self-reinforcement: no hypothesis built from proposed trajectories alone on noise
R1  ACH: final standing orders the stated claims as the case's ground truth does

Every metric reports its value, the number of items it rests on, and the items that failed, so a number can
always be traced back to scenario steps. Nothing here is gated: the runner reports.
"""
from __future__ import annotations

import re
import statistics

from .harness import ScenarioResult, StepRec

_STOP = set("the a an of to in on at by for and or is was were be been because due with from that this its their it as".split())


def words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", (s or "").lower()) if len(w) > 2 and w not in _STOP}


def jaccard(a: str, b: str) -> float:
    x, y = words(a), words(b)
    return len(x & y) / len(x | y) if x | y else 0.0


def ratio(xs) -> float | None:
    xs = list(xs)
    return round(sum(1 for x in xs if x) / len(xs), 3) if xs else None


def auroc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(num / den, 3) if den else None


def _best(step: StepRec, risk: str | None = None):
    gs = [g for g in step.groups if risk is None or g.risk == risk]
    return max(gs, key=lambda g: g.score) if gs else None


def _created(res: ScenarioResult, label: str) -> list[str]:
    """Every hypothesis the step with this label returned (a step may raise several)."""
    return list(dict.fromkeys(res.final.get("_created", {}).get(label) or []))


def _wanted(res: ScenarioResult, ref: str) -> list[str]:
    """`step:<label>` means any hypothesis that step returned; anything else is a stored id."""
    return _created(res, ref[5:]) if str(ref).startswith("step:") else [ref]


def _match_group(step: StepRec, want: list[str]):
    """The group that answers a `matches` expectation: one on a wanted hypothesis if any, else the first."""
    return next((g for g in step.groups if g.hypothesis_id in want), step.groups[0] if step.groups else None)


# ---------------------------------------------------------------- A1
def a1_score(results: list[ScenarioResult]) -> dict:
    hits, errs, recall, false_sig, hard, pos, fails, orderings, missing = [], [], [], [], [], [], [], [], []
    for res in results:
        bands = res.final.get("bands") or {"none": [0, 0.2], "weak": [0.2, 0.5], "strong": [0.5, 1.0]}
        by_label = {}
        for s in res.steps:
            e = s.expect
            if not e:
                continue
            g = _best(s, e.get("risk"))
            x = g.score if g else 0.0
            by_label[s.label] = x
            if e.get("band"):
                lo, hi = bands[e["band"]]
                ok = lo <= x <= hi
                hits.append(ok)
                errs.append(0.0 if ok else min(abs(x - lo), abs(x - hi)))
                if not ok:
                    fails.append(f"{res.id}:{s.label} band {e['band']} got {x}")
            if e.get("expect_signal") is True:
                recall.append(g is not None)
            if e.get("expect_signal") is False:
                false_sig.append(g is not None)
            if e.get("hard_negative"):
                hard.append(x)
            elif e.get("band") in ("weak", "strong"):
                pos.append(x)
            if e.get("expect_missing"):
                missing.append(bool(g and g.missing))
        for chain in res.final.get("orderings", []):
            vals = [by_label.get(c, 0.0) for c in chain]
            ok = all(a < b for a, b in zip(vals, vals[1:]))
            orderings.append(ok)
            if not ok:
                fails.append(f"{res.id}: ordering {' < '.join(chain)} got {vals}")
    return {"band_accuracy": ratio(hits), "mean_band_error": round(statistics.mean(errs), 3) if errs else None, "n": len(hits),
            "signal_recall": ratio(recall), "false_signal_rate": ratio(false_sig), "orderings_satisfied": ratio(orderings),
            "hard_negative_mean": round(statistics.mean(hard), 3) if hard else None, "positive_mean": round(statistics.mean(pos), 3) if pos else None,
            "hard_negative_auroc": auroc(pos, hard), "missing_flagged": ratio(missing), "failures": fails}


# ---------------------------------------------------------------- A2
def a2_lambda(results: list[ScenarioResult], established: float = 0.5, match: float = 0.2) -> dict:
    found, est, false_alarm, flags, fails, examples = [], [], [], [], [], []
    for res in results:
        if res.kind == "miss":
            targets = [t for t in [res.final.get("missing"), *res.final.get("alternatives", [])] if t]
            lam = [h for h in res.hypotheses.values() if h["lambda"]]
            if res.control:
                fa = [h for h in lam if (h["standing"] or 0) >= established]
                false_alarm.append(bool(fa))
                if fa:
                    fails.append(f"{res.id} (control): Lambda hypothesis established: {fa[0]['claim']}")
            else:
                best = max(lam, key=lambda h: max(jaccard(h["claim"], t) for t in targets), default=None)
                sim = max((jaccard(best["claim"], t) for t in targets), default=0.0) if best else 0.0
                ok = best is not None and sim >= match
                found.append(ok)
                est.append(ok and (best["standing"] or 0) >= established)
                examples.append({"case": res.id, "expected": res.final.get("missing"), "best": best["claim"] if best else None,
                                 "similarity": round(sim, 2), "standing": best["standing"] if best else None})
                if not ok:
                    fails.append(f"{res.id}: missing hypothesis not articulated (best: {best['claim'] if best else None})")
        for s in res.steps:
            for key in ("lambda_signal", "lambda_hypothesis"):
                if key in s.expect:
                    g = s.groups[0] if s.groups else None
                    ok = g is not None and getattr(g, key) == s.expect[key]
                    flags.append(ok)
                    if not ok:
                        fails.append(f"{res.id}:{s.label} expected {key}={s.expect[key]}, got {getattr(g, key) if g else 'no group'}")
    return {"missing_articulated": ratio(found), "missing_established": ratio(est), "control_false_alarm": ratio(false_alarm),
            "n_cases": len(found), "n_controls": len(false_alarm), "flag_accuracy": ratio(flags), "n_flags": len(flags),
            "examples": examples, "failures": fails}


# ---------------------------------------------------------------- A3
def a3_combination(results: list[ScenarioResult], high: float = 0.5, low: float = 0.35) -> dict:
    joint, joint_ctrl, refuted, supported, fails, rows = [], [], [], [], [], []
    for res in results:
        js = res.final.get("jointly_strong")
        if js:
            cands = _wanted(res, js["hypothesis"])   # the strongest of them at the end
            hid = max(cands, key=lambda c: res.hypotheses.get(c, {}).get("standing") or 0.0, default=None)
            h = res.hypotheses.get(hid or "", {})
            scores = [g.score for s in res.steps for g in s.groups if g.hypothesis_id == hid]
            val = h.get("standing") or 0.0
            strong = val >= js.get("min_standing", high) and (max(scores, default=0) < js.get("max_single", high))
            (joint_ctrl if res.control else joint).append(strong)
            rows.append({"scenario": res.id, "control": res.control, "standing": val, "scores": scores})
            if strong == res.control:
                fails.append(f"{res.id}: jointly strong={strong}, standing {val}, single scores {scores}")
        if res.kind == "conj":
            val = res.hypotheses.get("H1", {}).get("standing") or 0.0
            (supported if res.control else refuted).append(val)
            if (not res.control and val > low) or (res.control and val < high):
                fails.append(f"{res.id}: {'control' if res.control else 'refuted'} hypothesis standing {val}")
    return {"jointly_strong_detected": ratio(joint), "control_false_joint": ratio(joint_ctrl), "rows": rows,
            "refuted_mean_standing": round(statistics.mean(refuted), 3) if refuted else None,
            "supported_mean_standing": round(statistics.mean(supported), 3) if supported else None,
            "refutation_auroc": auroc(supported, refuted), "n_refuted": len(refuted), "n_supported": len(supported), "failures": fails}


# ---------------------------------------------------------------- A4
def a4_matching(results: list[ScenarioResult]) -> dict:
    ok, false_new, wrong, fails = [], [], [], []
    for res in results:
        for s in res.steps:
            m = s.expect.get("matches")
            if not m:
                continue
            want = _wanted(res, m)
            g = _match_group(s, want)
            good = g is not None and g.hypothesis_id in want and g.source == "predefined"
            ok.append(good)
            false_new.append(not good and g is not None and g.source == "articulated")
            wrong.append(not good and g is not None and g.source == "predefined")
            if not good:
                fails.append(f"{res.id}:{s.label} expected match {want}, got {(g.source, g.hypothesis_id, g.claim) if g else 'no group'}")
    return {"match_accuracy": ratio(ok), "false_new_rate": ratio(false_new), "wrong_match_rate": ratio(wrong), "n": len(ok), "failures": fails}


# ---------------------------------------------------------------- A5
def a5_stability(results: list[ScenarioResult], tolerance: float = 0.15) -> dict:
    groups: dict[str, list[float]] = {}
    for res in results:
        for s in res.steps:
            v = s.expect.get("variant_of")
            if v:
                g = s.groups[0] if s.groups else None
                groups.setdefault(f"{res.id}:{v}", []).append(g.score if g else 0.0)
    spreads = {k: round(max(v) - min(v), 3) for k, v in groups.items() if len(v) > 1}
    return {"mean_spread": round(statistics.mean(spreads.values()), 3) if spreads else None, "within_tolerance": ratio(x <= tolerance for x in spreads.values()),
            "tolerance": tolerance, "spreads": spreads, "failures": [f"{k}: spread {x}" for k, x in spreads.items() if x > tolerance]}


# ---------------------------------------------------------------- A6
def a6_self_reinforcement(results: list[ScenarioResult]) -> dict:
    ok, fails, peaks = [], [], {}
    for res in results:
        cap = res.final.get("max_standing")
        if cap is None:
            continue
        peak = max(((h["standing"] or 0.0, h["claim"]) for h in res.hypotheses.values() if h["origin"] == "system"), default=(0.0, None))
        peaks[res.id] = peak[0]
        ok.append(peak[0] <= cap)
        if peak[0] > cap:
            fails.append(f"{res.id}: system hypothesis reached standing {peak[0]} > {cap}: {peak[1]}")
    return {"held": ratio(ok), "peaks": peaks, "n": len(ok), "failures": fails}


# ---------------------------------------------------------------- R1
def r1_ach(results: list[ScenarioResult]) -> dict:
    rows, pair_ok, fails = [], [], []
    for res in results:
        exp = res.final.get("expected")
        if res.kind != "ach" or not exp:
            continue
        hids = [h for h in exp if h in res.hypotheses and exp[h].get("range")]
        got = [res.hypotheses[h]["standing"] or 0.0 for h in hids]
        mid = [sum(exp[h]["range"]) / 2 for h in hids]
        rho = spearman(got, mid)
        surv = [res.hypotheses[h]["standing"] or 0.0 for h in hids if not exp[h]["eliminated"]]
        elim = [res.hypotheses[h]["standing"] or 0.0 for h in hids if exp[h]["eliminated"]]
        pairs = [s > e for s in surv for e in elim]
        pair_ok += pairs
        rows.append({"case": res.id, "spearman": rho, "standing": dict(zip(hids, got)), "expected_mid": dict(zip(hids, mid))})
        if rho is not None and rho < 0.5:
            fails.append(f"{res.id}: Spearman {rho}")
    rhos = [r["spearman"] for r in rows if r["spearman"] is not None]
    return {"mean_spearman": round(statistics.mean(rhos), 3) if rhos else None, "survivor_above_eliminated": ratio(pair_ok), "cases": rows, "failures": fails}


# ---------------------------------------------------------------- step expectations (any scenario)
def e_expectations(results: list[ScenarioResult]) -> dict:
    """Per-step expectations not covered above: hypothesis source, inconsistency notes, rejected claims not re-articulated."""
    ok, fails = [], []
    for res in results:
        for s in res.steps:
            e = s.expect
            g = _match_group(s, _wanted(res, e["matches"])) if e.get("matches") else (s.groups[0] if s.groups else None)
            if "hypothesis_source" in e:
                good = g is not None and g.source == e["hypothesis_source"]
                ok.append(good)
                if not good:
                    fails.append(f"{res.id}:{s.label} hypothesis source {g.source if g else None}, expected {e['hypothesis_source']}")
            if "inconsistency" in e:
                good = g is not None and bool(g.inconsistency) == bool(e["inconsistency"])
                ok.append(good)
                if not good:
                    fails.append(f"{res.id}:{s.label} inconsistency {g.inconsistency if g else None}")
            if "stance" in e:   # with `matches`, the stance of the group on the matched hypothesis
                good = g is not None and g.stance == e["stance"] and (not e.get("matches") or g.hypothesis_id in _wanted(res, e["matches"]))
                if good and e["stance"] == "weakens" and g.without > 0:
                    good = g.standing < g.without      # counter-evidence must lower the standing
                ok.append(good)
                if not good:
                    got = f"{g.stance} on {g.hypothesis_id}, standing {g.without} -> {g.standing}" if g else "no group"
                    fails.append(f"{res.id}:{s.label} stance {got}, expected {e['stance']}")
            if "not_articulated_like" in e:
                claim = res.hypotheses.get(e["not_articulated_like"], {}).get("claim", "")
                dup = [x.claim for x in s.groups if x.source == "articulated" and jaccard(x.claim, claim) >= 0.5]
                ok.append(not dup)
                if dup:
                    fails.append(f"{res.id}:{s.label} re-articulated a rejected claim: {dup[0]}")
    return {"held": ratio(ok), "n": len(ok), "failures": fails}


def all_metrics(results: list[ScenarioResult]) -> dict:
    return {"A1_score": a1_score(results), "A2_lambda": a2_lambda(results), "A3_combination": a3_combination(results),
            "A4_matching": a4_matching(results), "A5_stability": a5_stability(results), "A6_self_reinforcement": a6_self_reinforcement(results),
            "R1_ach": r1_ach(results), "E_expectations": e_expectations(results)}
