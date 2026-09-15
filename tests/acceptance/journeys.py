"""End-to-end journeys driven ONLY through the public API (`mistabra.Mistabra`, `mistabra.Settings`).
No product internals are imported, so the same journeys run against the fake provider (default)
or a live model (--live). Every step logs what it does, its inputs, timing, and [PASS]/[FAIL]."""
from __future__ import annotations
import json, time
from pathlib import Path


class Log:
    def __init__(self, quiet=False): self.quiet, self.fails, self.passes, self.t0 = quiet, [], 0, time.time()
    def step(self, n, title): self._p(f"\n[{n}] {title}")
    def info(self, msg): self._p(f"    {msg}")
    def check(self, cond, label, actual=None):
        if cond: self.passes += 1; self._p(f"    [PASS] {label}")
        else: self.fails.append(label); self._p(f"    [FAIL] {label}" + (f"  actual={actual!r}" if actual is not None else ""))
    def timed(self, label, fn):
        t = time.perf_counter(); r = fn(); self._p(f"    → {label}  ({(time.perf_counter()-t)*1000:.0f} ms)"); return r
    def progress(self, i, n, label, every=5):
        if i % every == 0 or i == n: self._p(f"    {label} {i}/{n} ({100*i//n}%)  [{time.time()-self.t0:.1f}s elapsed]")
    def _p(self, s):
        if not self.quiet: print(s, flush=True)
    def summary(self):
        print(f"\n=== acceptance summary: {self.passes} passed, {len(self.fails)} failed, {time.time()-self.t0:.1f}s ===", flush=True)
        for f in self.fails: print(f"  FAILED: {f}", flush=True)
        return 0 if not self.fails else 1


def run_journeys(m, testdata: Path, log: Log, live: bool) -> None:
    td = Path(testdata)
    # ---- J1: ontology from documents ----
    log.step(1, "Build ontology draft from documents, review, publish, export")
    docs = [(p.name, p.read_text(encoding="utf-8")) for p in sorted((td / "docs").glob("*.md"))]
    log.info(f"docs: {[d[0] for d in docs]}")
    did = log.timed("ontology.build_draft", lambda: m.ontology.build_draft(m.ontology_id, docs, actor="acc:expert"))
    els = m.ontology.draft_elements(did)
    kinds = {}
    for e in els: kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    log.info(f"draft {did[:8]} elements by kind: {kinds}")
    log.check(kinds.get("objectives", 0) >= 3, "at least 3 objectives extracted", kinds)
    log.check(all(e["body"].get("provenance") or e["body"].get("status") == "inferred" for e in els), "every element has provenance or inferred status")
    log.check(kinds.get("signal_types", 0) >= 3 and kinds.get("observables", 0) >= 3, "signal types and observables inferred", kinds)
    sample = next(e for e in els if e["kind"] == "objectives")
    m.ontology.review(did, "objectives", sample["element_id"], "edit", "acc:expert", patch={"description": (sample["body"].get("description") or "") + " [reviewed]"})
    accepted = log.timed("ontology.accept_all", lambda: m.ontology.accept_all(did, "acc:expert"))
    log.info(f"accepted {accepted} pending elements")
    vid = log.timed("ontology.publish", lambda: m.ontology.publish(did, "acc:expert"))
    onto = m.ontology.load(vid)
    log.check(onto.version == 1 and onto.element_count() == accepted + 1, "published v1 with all accepted elements", (onto.version, onto.element_count(), accepted))
    exp = json.loads(m.ontology.export(vid))
    log.check("functions" not in exp and exp["ontology_id"] == m.ontology_id, "export is the custom JSON schema without baseline functions")
    log.check(any("reviewed" in (o.description or "") for o in onto.objectives), "edited description survived publish")

    # ---- J2: ingest and process ----
    log.step(2, "Register sources, ingest episodes (idempotent), extract observations and mentions")
    recs = [json.loads(l) for l in (td / "episodes.jsonl").read_text().splitlines() if l.strip()]
    src = {}
    for r in recs:
        if r["source"] not in src:
            src[r["source"]] = m.sources.register(r["source"], "testdata", 0.7, "acc:admin")
    created = dup = 0
    for i, r in enumerate(recs, 1):
        out = m.episodes.ingest(src[r["source"]], r["text"], "acc:host", r.get("occurrence_time"), {"seed_id": r["seed_id"]})
        created += out["created"]; dup += not out["created"]
        log.progress(i, len(recs), "ingesting", every=10)
    again = m.episodes.ingest(src[recs[0]["source"]], recs[0]["text"], "acc:host")
    log.check(created == len(recs) and dup == 0 and not again["created"], "all episodes created once; re-ingest is a no-op", (created, dup, again))
    res = log.timed("process (extract + resolve jobs)", lambda: m.process())
    log.info(f"jobs: {res}")
    log.check(res["failed"] == 0, "no failed jobs", res)
    st = m.stats()
    log.info(f"stats: {st}")
    log.check(st["observations"] >= len(recs), "at least one observation per episode", st["observations"])
    log.check(st["mentions"] > 0 and st["entities"] > 0, "mentions extracted and entities created")
    ep_by_seed = {m.episodes.get(e["id"])["metadata_json"] and json.loads(e["metadata_json"])["seed_id"]: e["id"] for e in m.episodes.list(limit=500)}
    e3 = m.episodes.observations(ep_by_seed["ep03"])
    log.check(any(o["modifiers"] for o in e3), "modifiers captured on the concealed-filming episode", [o["modifiers"] for o in e3])
    hits = m.vectors.search("observation", "photographed the front of a house", 3)
    log.check(hits and any(h["meta"]["episode_id"] in (ep_by_seed["ep01"], ep_by_seed["ep02"], ep_by_seed["ep_n1"]) for h in hits), "vector search finds the house-photo observations", [h["text"][:50] for h in hits])

    # ---- J3: entity resolution and corrections ----
    log.step(3, "Entity resolution across episodes; merge / split / relink are audited and reversible")
    hofer = m.entities.find("Jens Hofer", "person")
    log.check(len(hofer) == 1, "Jens Hofer resolved to exactly one entity", [h["canonical_name"] for h in m.entities.list("person")])
    klein, brandt = m.entities.find("Maria Klein", "person"), m.entities.find("Karl Brandt", "person")
    if brandt:
        ms = m.entities.mentions_of(brandt[0]["id"])
        log.info(f"Karl Brandt mentions: {[x['surface'] for x in ms]}")
        log.check(len(ms) >= 4 and any(x["surface"] == "Brandt" for x in ms), "surname-only mention linked to the same person", [x["surface"] for x in ms])
    log.check(klein and brandt and klein[0]["id"] != brandt[0]["id"], "distinct people stay distinct")
    unres = m.entities.unresolved()
    log.info(f"unresolved mentions: {[u['surface'] for u in unres]}")
    if klein and brandt:
        m.entities.merge(klein[0]["id"], brandt[0]["id"], "acc:analyst")
        merged = m.entities.get(brandt[0]["id"])
        log.check(merged["status"] == "merged" and merged["merged_into"] == klein[0]["id"], "merge recorded")
        ev = m.audit.events(object_type="entity", object_id=klein[0]["id"])[0]
        log.check(ev["action"] == "entity.merged" and ev["before"]["drop"]["canonical_name"] == "Karl Brandt", "merge audit carries before-state for reversal")
        bm = [x for x in m.entities.mentions_of(klein[0]["id"]) if "Brandt" in x["surface"]]
        newid = m.entities.split(klein[0]["id"], [x["id"] for x in bm], "Karl Brandt", "acc:analyst")
        log.check(len(m.entities.mentions_of(newid)) == len(bm) and m.entities.get(newid)["canonical_name"] == "Karl Brandt", "split restores a separate entity")

    # ---- J4: audit and traceability ----
    log.step(4, "Audit: chain integrity, trace an observation to its model call, actor attribution")
    log.check(m.audit.verify_chain(), "audit hash chain verifies")
    obs = m.episodes.observations(ep_by_seed["ep01"])[0]
    call = next((c for c in m.audit.llm_calls(limit=2000) if c["id"] == obs["llm_call_id"]), None)
    log.check(call is not None and call["task"] == "extract.episode" and call["prompt_version"], "observation traces to a logged extraction call with prompt version", call and (call["task"], call["prompt_version"]))
    actors = {e["actor"] for e in m.audit.events(limit=5000)}
    log.check({"acc:expert", "acc:admin", "acc:host", "acc:analyst", "system"} <= actors, "all actors attributed in audit", actors)
    llm = m.stats()["llm"]
    log.info(f"model calls: {llm} provider={m.provider.name}")
    if live:
        log.check(llm["errors"] == 0, "live model produced parseable output for every call", llm)

    # ---- J5: reopen ----
    log.step(5, "Data directory is self-contained: close and reopen")
    st1 = m.stats(); m.close()
    from mistabra import Mistabra
    m2 = Mistabra(m.settings.data_dir, settings=m.settings, provider=m.provider)
    st2 = m2.stats()
    log.check(st1["episodes"] == st2["episodes"] and st1["entities"] == st2["entities"] and m2.current_ontology().version == 1, "state survives reopen", (st1["episodes"], st2["episodes"]))
    m2.close()
