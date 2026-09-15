"""Thin CLI over the Python API. Example:
  mistabra ontology-build ./data --docs ./testdata/docs --accept-all --publish
  mistabra ingest ./data --source "Field reports" --jsonl ./testdata/episodes.jsonl
  mistabra process ./data ; mistabra stats ./data ; mistabra eval ./data --testdata ./testdata
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path


def _m(args):
    from .api import Mistabra
    return Mistabra(args.data_dir, log=(lambda s: print(s, flush=True)) if args.verbose else None)


def main(argv=None):
    p = argparse.ArgumentParser(prog="mistabra")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init"); s.add_argument("data_dir")
    s = sub.add_parser("ontology-build"); s.add_argument("data_dir"); s.add_argument("--docs", required=True); s.add_argument("--accept-all", action="store_true"); s.add_argument("--publish", action="store_true"); s.add_argument("--actor", default="cli")
    s = sub.add_parser("ontology-export"); s.add_argument("data_dir"); s.add_argument("--out")
    s = sub.add_parser("ingest"); s.add_argument("data_dir"); s.add_argument("--source", required=True); s.add_argument("--credibility", type=float, default=0.7); s.add_argument("--jsonl", required=True); s.add_argument("--actor", default="cli")
    s = sub.add_parser("process"); s.add_argument("data_dir")
    s = sub.add_parser("stats"); s.add_argument("data_dir")
    s = sub.add_parser("audit"); s.add_argument("data_dir"); s.add_argument("--limit", type=int, default=50)
    s = sub.add_parser("entities"); s.add_argument("data_dir")
    s = sub.add_parser("eval"); s.add_argument("data_dir"); s.add_argument("--testdata", required=True)
    args = p.parse_args(argv)
    m = _m(args)
    if args.cmd == "init":
        print(json.dumps(m.stats(), indent=1))
    elif args.cmd == "ontology-build":
        from .eval import load_testdata_docs
        did = m.ontology.build_draft(m.ontology_id, load_testdata_docs(args.docs), args.actor)
        print(f"draft {did}: {len(m.ontology.draft_elements(did))} elements")
        if args.accept_all: print(f"accepted {m.ontology.accept_all(did, args.actor)}")
        if args.publish: print(f"published version id {m.ontology.publish(did, args.actor)}")
    elif args.cmd == "ontology-export":
        vid = m.ontology.latest_version_id(m.ontology_id)
        text = m.ontology.export(vid) if vid else "{}"
        Path(args.out).write_text(text) if args.out else print(text)
    elif args.cmd == "ingest":
        from .ingest import read_jsonl
        src = m.sources.by_name(args.source) or {"id": m.sources.register(args.source, "jsonl", args.credibility, args.actor)}
        r = m.episodes.ingest_many(src["id"], read_jsonl(args.jsonl), args.actor)
        print(f"created {r['created']} duplicates {r['duplicates']}")
    elif args.cmd == "process":
        print(m.process())
    elif args.cmd == "stats":
        print(json.dumps(m.stats(), indent=1))
    elif args.cmd == "audit":
        for e in m.audit.events(limit=args.limit): print(e["ts"], e["actor"], e["action"], e["object_type"], (e["object_id"] or "")[:8])
    elif args.cmd == "entities":
        for e in m.entities.list(): print(e["entity_type"], "|", e["canonical_name"], "|", ", ".join(e["aliases"]))
    elif args.cmd == "eval":
        from .eval import run_eval
        run_eval(m, args.testdata, log=lambda s: print(s, flush=True))
    m.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
