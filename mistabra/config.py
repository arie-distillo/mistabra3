from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() not in ("", "0", "false", "no")


@dataclass
class Settings:
    """Runtime configuration. `force_mock` wins over any key, including one loaded
    from .env, so a default run can never reach a paid model by accident."""
    data_dir: Path
    provider: str = "fake"                      # fake | anthropic | openai_compatible
    reasoning_model: str = "claude-sonnet-4-5"
    operational_model: str = "claude-haiku-4-5"
    api_key: str | None = None
    base_url: str | None = None                 # for openai_compatible (e.g. OpenRouter)
    embed_dim: int = 256
    vector_backend: str = "numpy"               # numpy (default, persistent brute-force) | lancedb (optional extra)
    force_mock: bool = False
    chunk_chars: int = 1400
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls, data_dir: str | Path, **overrides) -> "Settings":
        env = os.environ
        s = cls(
            data_dir=Path(data_dir),
            provider=env.get("MISTABRA_PROVIDER", "fake"),
            reasoning_model=env.get("MISTABRA_REASONING_MODEL", "claude-sonnet-4-5"),
            operational_model=env.get("MISTABRA_OPERATIONAL_MODEL", "claude-haiku-4-5"),
            api_key=env.get("MISTABRA_API_KEY") or env.get("ANTHROPIC_API_KEY") or env.get("OPENROUTER_API_KEY"),
            base_url=env.get("MISTABRA_BASE_URL"),
            vector_backend=env.get("MISTABRA_VECTOR_BACKEND", "numpy"),
            force_mock=_truthy(env.get("MISTABRA_FORCE_MOCK")),
        )
        for k, v in overrides.items():
            setattr(s, k, v)
        return s

    @property
    def uses_real_model(self) -> bool:
        return self.provider != "fake" and bool(self.api_key) and not self.force_mock
