"""Acceptance: the PoC demonstration list from the API spec, driven through the public API only.

  python -m tests.acceptance.run            # fake provider, offline, free
  python -m tests.acceptance.run --live     # provider/model/key from MISTABRA_* env
Exit 0 = every check passed, 1 = failures. Logs every step with timing.
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, tempfile, time
from pathlib import Path


class Log:
    def __init__(self): self.t0, self.passes, self.fails = time.time(), 0, []
    def step(self, n, title): print(f"\n[{n}] {title}", flush=True)
    def info(self, msg): print(f"    {msg}", flush=True)
    def check(self, ok, label, actual=None):
        if ok: self.passes += 1; print(f"    [PASS] {label}", flush=True)
        else: self.fails.append(label); print(f"    [FAIL] {label}" + (f"  actual={actual!r}" if actual is not None else ""), flush=True)
    def timed(self, label, fn):
        t = time.perf_counter(); r = fn(); print(f"    → {label} ({(time.perf_counter() - t) * 1000:.0f} ms)", flush=True); return r
    def summary(self):
        print(f"\n=== {self.passes} passed, {len(self.fails)} failed, {time.time() - self.t0:.1f}s ===", flush=True)
        for f in self.fails: print(f"  FAILED: {f}", flush=True)
        return 1 if self.fails else 0


def run(m, testdata: Path, log: Log, live: bool):
    actor = "acc:analyst"
    log.step(1, "Context in, risks from a sentence, radars derived")
    for p in sorted((testdata / "context").glob("*.md")): m.context.create_file(p, actor)
    risks = json.loads((testdata / "risks.json").read_text())
    attack = m.risks.create(risks[0]["text"], actor, open=True)
    radars = log.timed("radars.derive", lambda: m.radars.derive(attack.id, actor))
    log.info("radars: " + " | ".join(r.query[:60] for r in radars))
    log.check(3 <= len(radars) <= 6 and all(r.origin == "system" for r in radars), "3-6 radars derived from the risk sentence", len(radars))

    log.step(2, "Episodes drive everything: signals with structured content")
    for name, c in (("Field reports", 0.8), ("Community hotline", 0.6), ("Social media monitoring", 0.5), ("Police liaison bulletin", 0.9)):
        m.sources.create(name, actor, credibility=c)
    recs = [json.loads(l) for l in (testdata / "episodes.jsonl").read_text().splitlines() if l.strip()]
    eps, t0 = {}, time.time()
    for i, r in enumerate(recs, 1):
        occ = tuple(r["occurred"]) if isinstance(r.get("occurred"), list) else r.get("occurred")
        eps[r["seed"]] = m.episodes.create(r["text"], actor, source=r["source"], occurred=occ, metadata={"seed": r["seed"]}).id
        if i % 8 == 0 or i == len(recs): log.info(f"episodes {i}/{len(recs)} [{time.time() - t0:.1f}s]")
    sig = {seed: m.signals.list(episode_id=eid, risk_id=attack.id) for seed, eid in eps.items()}
    labels = json.loads((testdata / "benchmark" / "labels.json").read_text())["labels"]
    pos = [s for s, l in labels.items() if l.get("expect_signal") is True]
    log.check(all(sig[s] for s in pos), "every labelled positive produced a signal", [s for s in pos if not sig[s]])
    log.check(not any(sig[s] for s in ("easy02", "easy03", "easy04")), "everyday episodes produced no signal", [s for s in ("easy02", "easy03", "easy04") if sig[s]])
    s01 = sig["ep01"][0]
    log.check(s01.content.summary and (s01.content.target or s01.content.entities), "signal carries structured content", s01.content.model_dump())

    log.step(3, "Blind rule, then trajectories and baselines before any strength")
    log.check(s01.strength_system is None and s01.trajectory is None, "system view hidden before the analyst rates")
    s01 = m.signals.update(s01.id, actor, strength=0.3)
    log.check(s01.strength_system is not None and s01.trajectory is not None and len(s01.baselines) >= 1, "after rating: strength, risk trajectory and baselines visible")
    log.info(f"ep01 system={s01.strength_system} analyst={s01.strength_analyst} metrics={s01.metrics}")
    log.info(f"risk trajectory: {s01.trajectory.rationale[:140]}")
    log.info("baselines: " + "; ".join(f"{b.ordinary_outcome} (fit {b.fit})" for b in s01.baselines))
    created = [e["action"] for e in reversed(m.audit.events(object_type="trajectory", limit=5000)) if e["object_id"] in {t.id for t in m.trajectories.list(signal_id=s01.id)}]
    reasoned = next((e for e in m.audit.events(object_type="signal", object_id=s01.id) if e["action"] == "signal.reasoned"), None)
    log.check(created and reasoned is not None, "trajectories were proposed and the strength was recorded in the audit")

    log.step(4, "Ordering on the benchmark (reported; asserted only for the fake)")
    def strength(seed):
        return max((m.signals.update(s.id, actor, strength=s.strength_analyst if s.strength_analyst is not None else 0.5).strength_system or 0 for s in sig[seed]), default=0.0)
    chain = {s: strength(s) for s in ("neg01", "ep01", "ep02", "ep03")}
    log.info(f"strengths: {chain}")
    ordered = chain["neg01"] < chain["ep01"] < chain["ep02"] < chain["ep03"]
    if live: log.info(f"ordering neg01 < ep01 < ep02 < ep03: {ordered} (measured, not gated)")
    else: log.check(ordered, "hard negative < weak < repeated < concealed")

    log.step(5, "Lambda: open risk raises a signal no radar anticipated, and nowhere else")
    lam = [s for seed, ss in sig.items() for s in ss if s.via == "lambda"]
    log.info("lambda signals on: " + ", ".join(seed for seed, ss in sig.items() if any(s.via == "lambda" for s in ss)))
    log.check(any(s.via == "lambda" for s in sig["ep16"]), "fly-through episode raised a Lambda signal")
    neg = {seed for seed, l in labels.items() if l.get("expect_signal") is False or l.get("hard_negative")}
    bad = [seed for seed in neg if any(s.via == "lambda" for s in sig[seed])]
    if live: log.info(f"lambda signals on labelled negatives: {bad} (measured, not gated)")
    else: log.check(not bad, "no Lambda signal on a labelled negative", bad)

    log.step(6, "Accumulation: hypotheses gain support on Elm Street in order")
    cov = m.hypotheses.coverage(attack.id)
    log.info("coverage: " + "; ".join(f"{i.hypothesis.claim[:50]} ×{i.support} reach {i.max_reach}" for i in cov.items))
    log.check(any(i.support >= 2 for i in cov.items), "at least one hypothesis is supported by several signals")
    log.check(cov.path_reach >= 0.8, "the supported hypotheses reach close to the risk", cov.path_reach)

    log.step(7, "Confirm, precedent, trace, replay")
    s03 = sig["ep03"][0]
    s03 = m.signals.update(s03.id, actor, strength=0.9)
    t = m.trajectories.update(s03.trajectory.id, actor, status="confirmed")
    log.check(t.status == "confirmed" and m.context.list("precedent"), "confirmed trajectory became precedent")
    tr = m.audit.trace(s03.id)
    log.check({c["task"] for c in tr["llm_calls"]} >= {"trajectory.risk", "trajectory.baseline"} and m.audit.verify(), "trace explains the signal end to end; audit chain verifies")
    rep = m.learn.replay([s01.id, s03.id], actor)
    log.info(f"replay mean |delta| = {rep['mean_abs_delta']}")
    log.check(rep["mean_abs_delta"] is not None and (live or rep["mean_abs_delta"] == 0.0), "replay reproduces stored strengths" if not live else "replay ran")
    st = m.stats(); log.info(f"stats: {st}")
    if live: log.check(st["llm"]["errors"] == 0, "live model returned parseable output on every call", st["llm"])


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--live", action="store_true"); ap.add_argument("--data-dir"); ap.add_argument("--keep", action="store_true")
    ap.add_argument("--testdata", default=str(Path(__file__).resolve().parents[2] / "testdata"))
    a = ap.parse_args(argv)
    if not a.live: os.environ["MISTABRA_FORCE_FAKE"] = "1"
    from mistabra import Mistabra, Settings
    data_dir = Path(a.data_dir) if a.data_dir else Path(tempfile.mkdtemp(prefix="mistabra-acc-"))
    settings = Settings.from_env(data_dir)
    if a.live and not settings.uses_real_llm:
        print("ERROR: --live but no provider configured (MISTABRA_PROVIDER, MISTABRA_MODEL, key or base URL)"); return 2
    m = Mistabra(data_dir, settings); log = Log()
    print(f"acceptance live={a.live} provider={m.provider.name} model={settings.model} fast={settings.fast_model} data={data_dir}", flush=True)
    try:
        run(m, Path(a.testdata), log, a.live)
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc(); log.check(False, f"aborted: {type(e).__name__}: {e}")
    finally:
        m.close()
    rc = log.summary()
    if not a.keep and not a.data_dir: shutil.rmtree(data_dir, ignore_errors=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
