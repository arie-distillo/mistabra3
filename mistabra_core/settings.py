from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() not in ("", "0", "false", "no")


@dataclass
class Settings:
    """Everything configurable about the reference implementation, in one place.

    LLM setup is uniform across providers: `provider` picks the adapter, `model` (reasoning tier) and
    `fast_model` (operational tier) pick the models, `api_key` and `base_url` pass through.
    `force_fake` wins over any key, so tests can never reach a paid model.
    """
    provider: str = "fake"                  # fake | anthropic | openai (any OpenAI-compatible endpoint)
    model: str = ""                         # reasoning tier: abduction, baselines, standing
    fast_model: str = ""                    # operational tier: radar reads, Lambda-signal check (defaults to model)
    api_key: str | None = None
    base_url: str | None = None
    force_fake: bool = False
    timeout_s: float = 120.0
    cache_path: Path | None = None          # optional on-disk cache of model outputs (SQLite); in-memory otherwise

    standing: str = "judged"                # judged (reasoning model weighs the attached trajectories) | noisy_or (baseline)
    proposed_weight: float = 0.5            # weight of a `proposed` trajectory in standing; confirmed = 1, rejected = 0
    relevance_threshold: float = 0.5        # radar read / Lambda-signal check confidence needed to proceed
    match_threshold: float = 0.6            # below this the abducer's match to a stored hypothesis is treated as new
    max_baselines: int = 3
    max_trajectories: int = 3               # per risk, when the layer completes the hypothesis
    max_attached: int = 30                  # attached trajectories weighed in standing (most recent kept)
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        e = os.environ
        s = cls(
            provider=e.get("MISTABRA_PROVIDER", "fake"),
            model=e.get("MISTABRA_MODEL", ""),
            fast_model=e.get("MISTABRA_FAST_MODEL", ""),
            api_key=e.get("MISTABRA_API_KEY") or e.get("ANTHROPIC_API_KEY") or e.get("OPENAI_API_KEY") or e.get("OPENROUTER_API_KEY"),
            base_url=e.get("MISTABRA_BASE_URL"),
            force_fake=_truthy(e.get("MISTABRA_FORCE_FAKE")),
            cache_path=Path(e["MISTABRA_CACHE"]) if e.get("MISTABRA_CACHE") else None,
            standing=e.get("MISTABRA_STANDING", "judged"),
        )
        for k, v in overrides.items():
            setattr(s, k, v)
        s.__post_init__()
        return s

    def __post_init__(self):
        if not self.model:
            self.model = {"anthropic": "claude-sonnet-4-5", "openai": "gpt-4o"}.get(self.provider, "fake-model")
        if not self.fast_model:
            self.fast_model = self.model
        if self.standing not in ("judged", "noisy_or"):
            raise ValueError(f"standing must be 'judged' or 'noisy_or', not {self.standing!r}")

    @property
    def uses_real_llm(self) -> bool:
        return self.provider != "fake" and bool(self.api_key or self.base_url) and not self.force_fake
