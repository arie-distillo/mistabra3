"""CLI: mistabra bench DATA_DIR --testdata ./testdata | stats DATA_DIR | signals DATA_DIR [--unblind]"""
from __future__ import annotations
import argparse, json, shutil, sys, tempfile
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(prog="mistabra")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench"); b.add_argument("data_dir", nargs="?"); b.add_argument("--testdata", default=str(Path(__file__).resolve().parents[1] / "testdata")); b.add_argument("--json", action="store_true"); b.add_argument("--decision", choices=["pairwise", "setwise"])
    s = sub.add_parser("stats"); s.add_argument("data_dir")
    g = sub.add_parser("signals"); g.add_argument("data_dir"); g.add_argument("--unblind", action="store_true")
    a = p.parse_args(argv)
    from .api import Mistabra
    from .settings import Settings
    if a.cmd == "bench":
        tmp = a.data_dir is None
        data_dir = a.data_dir or tempfile.mkdtemp(prefix="mistabra-bench-")
        m = Mistabra(data_dir, Settings.from_env(data_dir, **({"decision": a.decision} if a.decision else {})))
        print(f"provider={m.provider.name} model={m.settings.model} fast_model={m.settings.fast_model} decision={m.settings.decision} data_dir={data_dir}", flush=True)
        from .eval import run_benchmark
        out = run_benchmark(m, a.testdata, log=lambda x: print(x, flush=True))
        if a.json: print(json.dumps(out["summary"], indent=1))
        m.close()
        if tmp: shutil.rmtree(data_dir, ignore_errors=True)
    elif a.cmd == "stats":
        m = Mistabra(a.data_dir); print(json.dumps(m.stats(), indent=1)); m.close()
    elif a.cmd == "signals":
        m = Mistabra(a.data_dir, Settings.from_env(a.data_dir, blind=not a.unblind))
        for sg in m.signals.list():
            print(f"{sg.id} via={sg.via:7s} strength={sg.strength_system} analyst={sg.strength_analyst} status={sg.status} target={sg.content.target}")
        m.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
