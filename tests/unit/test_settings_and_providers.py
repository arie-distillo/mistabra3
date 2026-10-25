from mistabra import Settings
from mistabra.llm import make_provider, FakeProvider
from mistabra.llm.base import AnthropicProvider, OpenAICompatibleProvider


def test_force_fake_wins_over_key(monkeypatch, tmp_path):
    monkeypatch.setenv("MISTABRA_PROVIDER", "anthropic"); monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x"); monkeypatch.setenv("MISTABRA_FORCE_FAKE", "1")
    s = Settings.from_env(tmp_path)
    assert not s.uses_real_llm and isinstance(make_provider(s), FakeProvider)


def test_provider_selection_is_uniform(monkeypatch, tmp_path):
    monkeypatch.delenv("MISTABRA_FORCE_FAKE", raising=False)
    a = Settings(data_dir=tmp_path, provider="anthropic", api_key="k")
    o = Settings(data_dir=tmp_path, provider="openai", api_key="k", base_url="https://openrouter.ai/api/v1", model="x/y", fast_model="x/z")
    assert isinstance(make_provider(a), AnthropicProvider) and isinstance(make_provider(o), OpenAICompatibleProvider)
    assert a.model == "claude-sonnet-4-5" and a.fast_model == a.model and o.fast_model == "x/z"


def test_env_setup(monkeypatch, tmp_path):
    monkeypatch.setenv("MISTABRA_PROVIDER", "openai"); monkeypatch.setenv("MISTABRA_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("MISTABRA_MODEL", "llama3"); monkeypatch.delenv("MISTABRA_FORCE_FAKE", raising=False); monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    s = Settings.from_env(tmp_path)
    assert s.uses_real_llm and s.model == "llama3"   # local endpoint, no key needed


def test_unknown_provider_rejected(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        make_provider(Settings(data_dir=tmp_path, provider="nope", api_key="k"))
