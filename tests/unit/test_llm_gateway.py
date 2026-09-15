"""Gateway contract: caching, logging, malformed output is an error and is NOT retried."""
import pytest
from pydantic import BaseModel
from mistabra import Settings
from mistabra.store import Store
from mistabra.llm import Gateway, FakeProvider, LLMOutputError, TransportError
from mistabra.llm.prompts import PROMPTS, Prompt


class Out(BaseModel):
    signal_types: list = []


def _gw(tmp_path, provider):
    return Gateway(Store(tmp_path), Settings(data_dir=tmp_path), provider)


def test_cache_hit_second_time(tmp_path):
    p = FakeProvider(); gw = _gw(tmp_path, p)
    v = {"objective_id": "o", "objective_label": "x", "polarity": "negative", "description": None}
    gw.call("ontology.infer_signals", "reasoning", v, Out); gw.call("ontology.infer_signals", "reasoning", v, Out)
    assert gw.stats == {"calls": 1, "cached": 1, "errors": 0} and len(p.calls) == 1
    rows = gw.store.all("SELECT cached, parsed_ok, model, tier FROM llm_call ORDER BY ts")
    assert [r["cached"] for r in rows] == [0, 1] and all(r["parsed_ok"] for r in rows) and rows[0]["tier"] == "reasoning"


def test_tiers_select_models(tmp_path):
    gw = _gw(tmp_path, FakeProvider())
    assert gw.model_for("reasoning") != gw.model_for("operational")


@pytest.mark.parametrize("bad", ["", "not json at all", "{\"signal_types\": \"wrong type\"}", "```json\n{\"signal_types\": [1, 2\n"])
def test_malformed_output_raises_and_is_not_retried(tmp_path, bad):
    p = FakeProvider(faults={"ontology.infer_signals": bad}); gw = _gw(tmp_path, p)
    with pytest.raises(LLMOutputError):
        gw.call("ontology.infer_signals", "reasoning", {"objective_id": "o", "objective_label": "x", "polarity": "negative"}, Out)
    assert len(p.calls) == 1  # no retry on non-transient failure
    r = gw.store.one("SELECT parsed_ok, error FROM llm_call")
    assert r["parsed_ok"] == 0 and "LLMOutputError" in r["error"]


def test_fenced_json_is_accepted(tmp_path):
    p = FakeProvider(faults={"ontology.infer_signals": "Sure!\n```json\n{\"signal_types\": []}\n```"}); gw = _gw(tmp_path, p)
    out, _ = gw.call("ontology.infer_signals", "reasoning", {"objective_id": "o", "objective_label": "x", "polarity": "negative"}, Out)
    assert out.signal_types == []


def test_transport_error_is_retried_then_raised(tmp_path, monkeypatch):
    class Flaky:
        name = "flaky"
        def __init__(self): self.n = 0
        def complete(self, *a):
            self.n += 1; raise TransportError("down")
    monkeypatch.setattr("mistabra.llm.gateway.time.sleep", lambda s: None)
    p = Flaky(); gw = _gw(tmp_path, p)
    with pytest.raises(TransportError):
        gw.call("ontology.infer_signals", "reasoning", {"objective_id": "o", "objective_label": "x", "polarity": "negative"}, Out)
    assert p.n == 3


def test_prompts_are_versioned():
    for pid, p in PROMPTS.items():
        assert isinstance(p, Prompt) and p.id == pid and p.version
