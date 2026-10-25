from mistabra.eval import run_benchmark
from tests.unit.conftest import TESTDATA


def test_benchmark_runs_offline_and_reports(m):
    out = run_benchmark(m, TESTDATA, log=lambda s: None)
    sm = out["summary"]
    assert sm["signal_recall"] == 1.0 and sm["lambda_detected"] is True and sm["orderings_satisfied"] == 1.0
    assert sm["provider"] == "fake" and set(out["per_episode"]) >= {"ep01", "neg01", "easy01", "miss01"}
