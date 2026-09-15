"""Standalone acceptance runner.

  python -m tests.acceptance.run                 # fake provider, temp data dir (free, offline)
  python -m tests.acceptance.run --live          # real provider from env (MISTABRA_PROVIDER, key, models)
  python -m tests.acceptance.run --data-dir ./acc-data --keep

Exit code 0 = all checks passed, 1 = failures.
"""
from __future__ import annotations
import argparse, os, shutil, sys, tempfile
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="use the real provider configured in the environment")
    ap.add_argument("--data-dir", default=None); ap.add_argument("--keep", action="store_true")
    ap.add_argument("--testdata", default=str(Path(__file__).resolve().parents[2] / "testdata"))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if not a.live:
        os.environ["MISTABRA_FORCE_MOCK"] = "1"
    from mistabra import Mistabra, Settings
    from tests.acceptance.journeys import Log, run_journeys
    data_dir = Path(a.data_dir) if a.data_dir else Path(tempfile.mkdtemp(prefix="mistabra-acc-"))
    settings = Settings.from_env(data_dir)
    log = Log(quiet=a.quiet)
    print(f"acceptance: live={a.live} provider={'real:' + settings.provider if settings.uses_real_model else 'fake'} data_dir={data_dir}", flush=True)
    if a.live and not settings.uses_real_model:
        print("ERROR: --live requested but no real provider configured (set MISTABRA_PROVIDER and an API key)", flush=True); return 2
    m = Mistabra(data_dir, settings=settings)
    try:
        run_journeys(m, Path(a.testdata), log, live=a.live)
    except Exception as e:  # noqa: BLE001
        log.check(False, f"journey aborted: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
    finally:
        try: m.close()
        except Exception: pass
    rc = log.summary()
    if not a.keep and not a.data_dir: shutil.rmtree(data_dir, ignore_errors=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
