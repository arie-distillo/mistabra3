from mistabra.llm import TransportError


def test_jobs_run_in_order_and_record_status(m):
    seen = []
    m.jobs.register("t", lambda p: seen.append(p["n"]))
    for i in range(3): m.jobs.enqueue("t", {"n": i})
    assert m.jobs.run_pending() == {"done": 3, "failed": 0} and seen == [0, 1, 2]
    assert {r["status"] for r in m.store.all("SELECT status FROM job")} == {"done"}


def test_transient_failure_requeues_once(m):
    calls = []
    def h(p):
        calls.append(1)
        if len(calls) == 1: raise TransportError("net")
    m.jobs.register("t", h); m.jobs.enqueue("t", {})
    r = m.jobs.run_pending()
    assert r == {"done": 1, "failed": 1} and len(calls) == 2 and m.store.one("SELECT status FROM job")["status"] == "done"


def test_non_transient_failure_is_final(m):
    m.jobs.register("t", lambda p: 1 / 0); m.jobs.enqueue("t", {})
    r = m.jobs.run_pending()
    j = m.store.one("SELECT status, error, attempts FROM job")
    assert r["failed"] == 1 and j["status"] == "failed" and j["attempts"] == 1 and "ZeroDivisionError" in j["error"]


def test_unknown_kind_fails_cleanly(m):
    m.jobs.enqueue("nope", {})
    assert m.jobs.run_pending()["failed"] == 1
