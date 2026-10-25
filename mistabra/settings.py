from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() not in ("", "0", "false", "no")


@dataclass
class Settings:
    """Everything configurable, in one place.

    LLM setup is uniform across providers: `provider` picks the adapter,
    `model` (reasoning) and `fast_model` (operational) pick the models,
    `api_key` and `base_url` are passed through. `force_fake` wins over any key.
    """
    data_dir: Path
    provider: str = "fake"                  # fake | anthropic | openai   (openai = any OpenAI-compatible endpoint)
    model: str = ""                         # reasoning tier: radar derivation, trajectories
    fast_model: str = ""                    # operational tier: radar reads, Lambda-signal check (defaults to model)
    api_key: str | None = None
    base_url: str | None = None             # e.g. https://openrouter.ai/api/v1, http://localhost:11434/v1
    force_fake: bool = False
    blind: bool = True                      # hide system strength/trajectory until the analyst rates
    timeout_s: float = 120.0
    max_baselines: int = 3
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls, data_dir: str | Path, **overrides) -> "Settings":
        e = os.environ
        s = cls(
            data_dir=Path(data_dir),
            provider=e.get("MISTABRA_PROVIDER", "fake"),
            model=e.get("MISTABRA_MODEL", ""),
            fast_model=e.get("MISTABRA_FAST_MODEL", ""),
            api_key=e.get("MISTABRA_API_KEY") or e.get("ANTHROPIC_API_KEY") or e.get("OPENAI_API_KEY") or e.get("OPENROUTER_API_KEY"),
            base_url=e.get("MISTABRA_BASE_URL"),
            force_fake=_truthy(e.get("MISTABRA_FORCE_FAKE")),
            blind=not _truthy(e.get("MISTABRA_NO_BLIND")),
        )
        for k, v in overrides.items():
            setattr(s, k, v)
        return s

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        if not self.model:
            self.model = {"anthropic": "claude-sonnet-4-5", "openai": "gpt-4o"}.get(self.provider, "fake-model")
        if not self.fast_model:
            self.fast_model = self.model

    @property
    def uses_real_llm(self) -> bool:
        return self.provider != "fake" and bool(self.api_key or self.base_url) and not self.force_fake
