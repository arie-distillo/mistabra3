"""Gateway contract (PydanticAI output tools, validation retries, error mapping, cache, records, failure log),
settings, and the testbed's own plumbing (scenario loading, other-layer stub, metrics)."""
import pytest

from mistabra_core import Settings
from mistabra_core.llm import OUTPUT_RETRIES, FakeProvider, Gateway, LLMAuthError, LLMOutputError, LLMTransportError, Recorder

V = {"risk_text": "r", "context": [], "hypotheses": []}


def _gw(provider, **kw):
    return Gateway(Settings(force_fake=True, **kw), provider)


def test_cache_and_record():
    p = FakeProvider()
    gw, rec = _gw(p), Recorder()
    gw.call("radar.derive", V, recorder=rec)
    gw.call("radar.derive", V, recorder=rec)
    assert gw.stats == {"calls": 1, "cached": 1, "errors": 0, "retries": 0} and p.calls == ["radar.derive"]
    assert [c.cached for c in rec.calls] == [False, True] and rec.calls[0].prompt_version == "1" and rec.calls[0].retries == 0


def test_disk_cache_survives_a_new_gateway(tmp_path):
    path = tmp_path / "c.sqlite"
    _gw(FakeProvider(), cache_path=path).call("radar.derive", V)
    p = FakeProvider()
    _gw(p, cache_path=path).call("radar.derive", V)
    assert p.calls == []


def test_validation_error_is_sent_back_and_fixed():
    answers = iter(['{"radars": "wrong"}', '{"radars": [{"query": "q"}]}'])
    p, rec = FakeProvider({"radar.derive": lambda d: next(answers)}), Recorder()
    gw = _gw(p)
    out = gw.call("radar.derive", V, recorder=rec)
    assert [r.query for r in out.radars] == ["q"] and p.calls == ["radar.derive"] * 2
    assert rec.calls[0].retries == 1 and gw.stats["retries"] == 1


@pytest.mark.parametrize("bad", ['{"radars": [42]}', "no json here", '{"radars": "wrong"}', '{"radars": ['])
def test_persistently_malformed_output_fails_after_retries(bad, tmp_path):
    import json
    log = tmp_path / "f.jsonl"
    p, rec = FakeProvider({"radar.derive": bad}), Recorder()
    with pytest.raises(LLMOutputError, match="radar.derive"):
        _gw(p, failure_log=log).call("radar.derive", V, recorder=rec)
    assert len(p.calls) == 1 + OUTPUT_RETRIES and rec.calls[0].ok is False
    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert entry["task"] == "radar.derive" and "TASK: radar.derive" in entry["prompt"] and len(entry["messages"]) >= 2


def test_bare_string_radars_accepted():
    out = _gw(FakeProvider({"radar.derive": '{"radars": ["Does the episode report a blocked culvert?", {"query": "q2"}]}'})).call("radar.derive", V)
    assert [r.query for r in out.radars] == ["Does the episode report a blocked culvert?", "q2"]


def _raising(exc):
    n = {"calls": 0}

    def boom(d):
        n["calls"] += 1
        raise exc
    return FakeProvider({"radar.derive": boom}), n


def test_auth_error_not_retried():
    from pydantic_ai.exceptions import ModelHTTPError
    p, n = _raising(ModelHTTPError(401, "m", {"error": {"message": "User not found."}}))
    with pytest.raises(LLMAuthError, match="401"):
        _gw(p).call("radar.derive", V)
    assert n["calls"] == 1


def test_server_error_retried(monkeypatch):
    from pydantic_ai.exceptions import ModelHTTPError
    monkeypatch.setattr("mistabra_core.llm.gateway.time.sleep", lambda s: None)
    p, n = _raising(ModelHTTPError(503, "m", "busy"))
    with pytest.raises(LLMTransportError):
        _gw(p).call("radar.derive", V)
    assert n["calls"] == 3


@pytest.mark.parametrize("provider,cls", [("openrouter", "OpenAIChatModel"), ("openai", "OpenAIChatModel"), ("anthropic", "AnthropicModel")])
def test_backend_builds_pydantic_ai_models(provider, cls):
    from mistabra_core.llm import Backend
    b = Backend(Settings(provider=provider, api_key="k", base_url="http://localhost:11434/v1" if provider == "openai" else None,
                         reasoning="none"))
    m = b.model("reasoning")
    assert type(m).__name__ == cls and b.model("reasoning") is m
    ms = b.model_settings()
    assert ms["temperature"] == 0 and ms["max_tokens"] == 4000
    if provider == "openrouter":
        assert ms["extra_body"] == {"reasoning": {"enabled": False}} and "openrouter.ai" in str(m.client.base_url)


def test_settings_force_fake_and_validation(monkeypatch):
    monkeypatch.setenv("MISTABRA_PROVIDER", "anthropic")
    monkeypatch.setenv("MISTABRA_API_KEY", "k")
    s = Settings.from_env()
    assert not s.uses_real_llm and s.model == "claude-sonnet-4-5"          # MISTABRA_FORCE_FAKE is set by conftest
    with pytest.raises(ValueError):
        Settings(standing="vote")


def test_scenarios_load():
    from testbed.scenario import discover
    sc = {s.id: s for s in discover(include_private=False)}
    assert {"community_safety", "flood_conjunction", "flood_control", "analyst_hunches", "noise_drift", "stability_elm",
            "burnt_bread_ach_classic", "conj_01", "conj_ctrl_01", "miss_01", "miss_ctrl_02"} <= set(sc)
    assert len(sc["community_safety"].steps) == 31 and sc["burnt_bread_ach_classic"].per_hypothesis
    assert [h.id for h in sc["miss_01"].operational.hypotheses] == ["G1", "G2"] and sc["conj_01"].steps[0].hypothesis == "H1"


def test_harness_keeps_and_reuses(tmp_path):
    from mistabra_core import Core
    from testbed.harness import run_scenario
    from testbed.scenario import discover
    sc = next(s for s in discover(include_private=False) if s.id == "flood_conjunction")
    res = run_scenario(Core(Settings(force_fake=True)), sc, log=lambda m: None)
    assert len(res.steps) == 2 and all(s.groups for s in res.steps)
    first = res.steps[0].groups[0]
    assert first.source == "articulated" and res.hypotheses[first.hypothesis_id]["lambda"] and res.hypotheses[first.hypothesis_id]["status"] == "proposed"
    second = res.steps[1].groups[0]
    assert second.hypothesis_id == first.hypothesis_id and second.without > 0      # the stored proposal was matched and weighed


def test_analyst_actions_applied():
    from mistabra_core import Core, Episode, Hypothesis, MissionContext, OperationalContext, Risk
    from testbed.harness import run_scenario
    from testbed.scenario import Scenario, Step
    sc = Scenario(id="t", kind="sequence", mission=MissionContext(version="1", risks=(Risk(id="attack", text="Hateful radicals attack homes on Elm Street."),)),
                  operational=OperationalContext(hypotheses=(Hypothesis(id="H", risk_id="attack", claim="Homes on Elm Street are a target"),)),
                  steps=[Step(episode=Episode(id="a", text="A passer-by photographed a house on Elm Street."), hypothesis="H", after={"confirm": True}, label="a"),
                         Step(episode=Episode(id="b", text="Someone photographed a house on Elm Street again."), hypothesis="H",
                              after={"reject_hypothesis": "H"}, label="b")])
    res = run_scenario(Core(Settings(force_fake=True, standing="noisy_or")), sc, log=lambda m: None)
    assert res.steps[1].groups[0].without > 0 and res.hypotheses["H"]["status"] == "rejected"


def test_metrics_on_known_values():
    from testbed.metrics import auroc, jaccard, spearman
    assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0 and auroc([0.1], [0.9]) == 0.0 and auroc([], [1]) is None
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0 and spearman([1, 2, 3], [3, 2, 1]) == -1.0
    assert jaccard("ventilation intake blocked", "the ventilation intake was blocked") == 1.0


def test_full_offline_run_produces_every_metric():
    from testbed.run import run
    rep = run({"force_fake": True}, ["flood", "miss_ctrl_01", "conj_ctrl_01", "noise", "burnt"], log=lambda m: None)
    m = rep["metrics"]
    assert set(m) == {"A1_score", "A2_lambda", "A3_combination", "A4_matching", "A5_stability", "A6_self_reinforcement", "R1_ach", "E_expectations"}
    assert all(not s["error"] for s in rep["scenarios"]) and rep["llm"]["errors"] == 0


def test_dotenv_loaded_from_parent_without_overriding(tmp_path, monkeypatch):
    from mistabra_core.settings import load_dotenv
    (tmp_path / ".env").write_text("# comment\nMISTABRA_T_A=one   # trailing\nexport MISTABRA_T_B='two words'\nMISTABRA_T_C=x\n", encoding="utf-8")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    monkeypatch.setenv("MISTABRA_T_C", "kept")
    for k in ("MISTABRA_T_A", "MISTABRA_T_B"):
        monkeypatch.delenv(k, raising=False)
    assert load_dotenv(sub) == tmp_path / ".env"
    import os
    assert (os.environ["MISTABRA_T_A"], os.environ["MISTABRA_T_B"], os.environ["MISTABRA_T_C"]) == ("one", "two words", "kept")
    for k in ("MISTABRA_T_A", "MISTABRA_T_B"):
        monkeypatch.delenv(k)


def test_key_belongs_to_the_provider(monkeypatch):
    from mistabra_core.settings import _key_for
    env = {"OPENAI_API_KEY": "sk-openai", "ANTHROPIC_API_KEY": "sk-ant", "OPENROUTER_API_KEY": "sk-or"}
    assert _key_for("openrouter", env) == ("OPENROUTER_API_KEY", "sk-or")
    assert _key_for("anthropic", env) == ("ANTHROPIC_API_KEY", "sk-ant")
    assert _key_for("openrouter", {"OPENAI_API_KEY": "sk-openai"}) == (None, None)       # never another service's key
    assert _key_for("openai", {**env, "MISTABRA_API_KEY": "explicit"}) == ("MISTABRA_API_KEY", "explicit")


def test_nested_json_sent_as_strings_is_decoded():
    """Qwen 3.5 via OpenRouter: nested objects and lists arrive as JSON-encoded strings (from a real failure log)."""
    import json
    args = json.dumps({"relevant": True,
                       "signal": json.dumps({"actor": "weather service", "behaviour": "forecasting heavy rain", "entities": ["Northfield"]}),
                       "trajectories": "\n" + json.dumps([{"hypothesis_id": None, "claim": "Heavy rain will overwhelm the drains.",
                                                            "match_confidence": 0, "rationale": "r", "reach": 0.4, "fit": 0.3}]) + "\n"})
    p = FakeProvider({"trajectory.abduce": args})
    out = _gw(p).call("trajectory.abduce", {"x": 1})
    assert out.signal.actor == "weather service" and out.trajectories[0].claim.startswith("Heavy rain") and len(p.calls) == 1
