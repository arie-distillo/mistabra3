"""A key in the environment or .env must never reach a real provider when force_mock is set."""
import os
from mistabra import Settings
from mistabra.llm import make_provider, FakeProvider


def test_force_mock_wins_over_key(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-key")
    monkeypatch.setenv("MISTABRA_PROVIDER", "anthropic")
    monkeypatch.setenv("MISTABRA_FORCE_MOCK", "1")
    s = Settings.from_env(tmp_path)
    assert s.api_key == "sk-fake-key" and not s.uses_real_model
    assert isinstance(make_provider(s), FakeProvider)


def test_real_provider_only_when_key_and_no_force(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-key")
    monkeypatch.setenv("MISTABRA_PROVIDER", "anthropic")
    monkeypatch.delenv("MISTABRA_FORCE_MOCK", raising=False)
    s = Settings.from_env(tmp_path)
    assert s.uses_real_model  # construction of the real client is allowed; no network call happens here
