import pytest
from mistabra import Settings
from mistabra.store import Store
from mistabra.llm import Gateway, FakeProvider, LLMOutputError, LLMTransportError


def _gw(tmp_path, provider):
    return Gateway(Store(tmp_path), Settings(data_dir=tmp_path), provider)

V = {"risk_text": "r", "context": []}


def test_cache_and_log(tmp_path):
    p = FakeProvider(); gw = _gw(tmp_path, p)
    gw.call("radar.derive", V); gw.call("radar.derive", V)
    assert gw.stats == {"calls": 1, "cached": 1, "errors": 0} and p.calls == ["radar.derive"]
    rows = gw.store.all("SELECT cached, ok, prompt_version FROM llm_call ORDER BY ts")
    assert [r["cached"] for r in rows] == [0, 1] and all(r["ok"] for r in rows) and rows[0]["prompt_version"] == "1"


@pytest.mark.parametrize("bad", ["", "no json here", "{\"radars\": \"wrong\"}", "```json\n{\"radars\": [\n"])
def test_malformed_output_not_retried(tmp_path, bad):
    p = FakeProvider(faults={"radar.derive": bad}); gw = _gw(tmp_path, p)
    with pytest.raises(LLMOutputError):
        gw.call("radar.derive", V)
    assert len(p.calls) == 1 and gw.store.one("SELECT ok, error FROM llm_call")["ok"] == 0


def test_fenced_json_accepted(tmp_path):
    gw = _gw(tmp_path, FakeProvider(faults={"radar.derive": "Sure:\n```json\n{\"radars\": [{\"query\": \"q\"}]}\n```"}))
    out, _ = gw.call("radar.derive", V)
    assert out.radars[0].query == "q"


def test_transport_error_retried(tmp_path, monkeypatch):
    class Down:
        name = "down"
        def __init__(self): self.n = 0
        def complete(self, *a): self.n += 1; raise LLMTransportError("net")
    monkeypatch.setattr("mistabra.llm.gateway.time.sleep", lambda s: None)
    p = Down(); gw = _gw(tmp_path, p)
    with pytest.raises(LLMTransportError):
        gw.call("radar.derive", V)
    assert p.n == 3
