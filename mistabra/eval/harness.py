"""Evaluation harness — measurements reported over time, not build gates.

ontology_metrics: precision/recall of extracted elements against a reviewed reference.
resolution_metrics: pairwise precision/recall of mention->entity assignment against labelled clusters.
"""
from __future__ import annotations
import json, time
from itertools import combinations
from pathlib import Path
from rapidfuzz import fuzz
from ..entities import normalize_name


def load_testdata_docs(folder: Path | str) -> list[tuple[str, str]]:
    return [(p.name, p.read_text(encoding="utf-8")) for p in sorted(Path(folder).glob("*.md"))]


def _match(a: str, b: str) -> bool:
    return fuzz.token_set_ratio(normalize_name(a), normalize_name(b)) >= 90


def ontology_metrics(elements: dict[str, list[str]], reference: dict[str, list[str]]) -> dict:
    """elements/reference: kind -> list of labels."""
    out = {}
    for kind, ref in reference.items():
        got = elements.get(kind, [])
        tp = sum(1 for g in got if any(_match(g, r) for r in ref))
        found = sum(1 for r in ref if any(_match(g, r) for g in got))
        out[kind] = {"precision": tp / len(got) if got else 0.0, "recall": found / len(ref) if ref else 0.0, "extracted": len(got), "reference": len(ref)}
    return out


def resolution_metrics(mentions: list[dict], labelled: dict) -> dict:
    """mentions: [{surface, entity_type, entity_id}] with entity_id None for unresolved."""
    truth = {}
    for c in labelled["clusters"]:
        for s in c["surfaces"]:
            truth[(c["type"], normalize_name(s))] = c["entity"]
    items = [(m["entity_id"], truth.get((m["entity_type"], normalize_name(m["surface"])))) for m in mentions]
    items = [(e, t) for e, t in items if t is not None]
    tp = fp = fn = 0
    for (e1, t1), (e2, t2) in combinations(items, 2):
        same_truth, same_pred = t1 == t2, (e1 is not None and e1 == e2)
        if same_pred and same_truth: tp += 1
        elif same_pred and not same_truth: fp += 1
        elif same_truth and not same_pred: fn += 1
    unresolved = sum(1 for e, _ in items if e is None)
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    forbidden = 0
    by_truth = {}
    for e, t in items:
        if e: by_truth.setdefault(t, set()).add(e)
    for a, b in labelled.get("must_not_merge", []):
        if by_truth.get(a, set()) & by_truth.get(b, set()): forbidden += 1
    return {"pairwise_precision": p, "pairwise_recall": r, "labelled_mentions": len(items), "unresolved": unresolved, "forbidden_merges": forbidden}


def run_eval(m, testdata: Path | str, log=print) -> dict:
    """End-to-end measurement on the shipped test data using whatever provider `m` is configured with."""
    td, t0 = Path(testdata), time.time()
    log(f"[eval] provider={m.provider.name} data_dir={m.settings.data_dir}", flush=True) if log is print else log(f"[eval] provider={m.provider.name}")
    docs = load_testdata_docs(td / "docs")
    did = m.ontology.build_draft(m.ontology_id, docs, actor="eval")
    els = m.ontology.draft_elements(did)
    by_kind = {}
    for e in els:
        if e["origin"] != "inferred": by_kind.setdefault(e["kind"], []).append(e["body"]["label"])
    om = ontology_metrics(by_kind, json.loads((td / "reference_ontology.json").read_text()))
    for k, v in om.items(): log(f"[eval] ontology {k:13s} P={v['precision']:.2f} R={v['recall']:.2f} ({v['extracted']} extracted / {v['reference']} ref)")
    m.ontology.accept_all(did, "eval"); vid = m.ontology.publish(did, "eval")
    onto = m.ontology.load(vid)
    log(f"[eval] published v{onto.version}: {len(onto.objectives)} objectives, {len(onto.signal_types)} signal types, {len(onto.observables)} observables")
    src_ids = {}
    from ..ingest import read_jsonl
    recs = list(read_jsonl(td / "episodes.jsonl"))
    for r in recs:
        if r["source"] not in src_ids:
            s = m.sources.by_name(r["source"])
            cred = next((x.credibility for x in onto.sources if x.label == r["source"]), 0.5)
            src_ids[r["source"]] = s["id"] if s else m.sources.register(r["source"], "testdata", cred, "eval")
        m.episodes.ingest(src_ids[r["source"]], r["text"], "eval", r.get("occurrence_time"), {"seed_id": r["seed_id"]})
    n = len(recs)
    log(f"[eval] ingested {n} episodes; processing")
    res = m.jobs.run_pending()
    log(f"[eval] jobs done={res['done']} failed={res['failed']}")
    mentions = [dict(r) for r in m.store.all("SELECT surface, entity_type, entity_id FROM entity_mention")]
    rm = resolution_metrics(mentions, json.loads((td / "labelled_entities.json").read_text()))
    log(f"[eval] resolution pairwise P={rm['pairwise_precision']:.2f} R={rm['pairwise_recall']:.2f} unresolved={rm['unresolved']} forbidden_merges={rm['forbidden_merges']} (labelled mentions {rm['labelled_mentions']})")
    st = m.stats()
    log(f"[eval] stats: {st}")
    log(f"[eval] elapsed {time.time() - t0:.1f}s")
    return {"ontology": om, "resolution": rm, "stats": st, "ontology_version_id": vid}
