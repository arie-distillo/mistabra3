from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(start: Path | None = None) -> Path | None:
    """Load MISTABRA_* style variables from the first `.env` found in `start` (default: the current directory) or its
    parents. Variables already set in the environment win. Returns the file used, if any. No dependency needed."""
    here = (start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        f = d / ".env"
        if f.is_file():
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip().removeprefix("export ").strip()
                v = v.strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                    v = v[1:-1]
                elif " #" in v:
                    v = v.split(" #", 1)[0].strip()
                os.environ.setdefault(k, v)
            return f
    return None


KEY_VARS = {"anthropic": ["ANTHROPIC_API_KEY"], "openai": ["OPENAI_API_KEY"], "openrouter": ["OPENROUTER_API_KEY"]}


def _key_for(provider: str, env) -> tuple[str | None, str | None]:
    """The API key for this provider: MISTABRA_API_KEY first, then only the provider's own variable — never another
    service's key. Returns (variable name, value)."""
    for var in ["MISTABRA_API_KEY", *KEY_VARS.get(provider, [])]:
        if env.get(var):
            return var, env[var]
    return None, None


def describe_key(provider: str) -> str:
    var, val = _key_for(provider, os.environ)
    return f"{var}={val[:6]}…{val[-4:]}" if val and len(val) > 12 else (f"{var}=(short)" if val else "no key")


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() not in ("", "0", "false", "no")


@dataclass
class Settings:
    """Everything configurable about the reference implementation, in one place.

    LLM setup is uniform across providers: `provider` picks the adapter, `model` (reasoning tier) and
    `fast_model` (operational tier) pick the models, `api_key` and `base_url` pass through.
    `force_fake` wins over any key, so tests can never reach a paid model.
    """
    provider: str = "fake"                  # fake | anthropic | openai (any OpenAI-compatible endpoint) | openrouter
    model: str = ""                         # reasoning tier: abduction, baselines, standing
    fast_model: str = ""                    # operational tier: radar reads, Lambda-signal check (defaults to model)
    api_key: str | None = None
    base_url: str | None = None
    force_fake: bool = False
    timeout_s: float = 120.0
    cache_path: Path | None = None          # optional on-disk cache of model outputs (SQLite); in-memory otherwise
    max_tokens: int = 4000                  # per call; thinking models count their reasoning against it
    reasoning: str | None = None            # None = model default | none | low | medium | high (openai / openrouter)
    structured: bool = True                 # send each prompt's JSON schema as response_format (openai / openrouter)

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
        """Settings from MISTABRA_* environment variables, after loading a `.env` file if one is found."""
        load_dotenv()
        e = os.environ
        s = cls(
            provider=e.get("MISTABRA_PROVIDER", "fake"),
            model=e.get("MISTABRA_MODEL", ""),
            fast_model=e.get("MISTABRA_FAST_MODEL", ""),
            api_key=_key_for(e.get("MISTABRA_PROVIDER", "fake"), e)[1],
            base_url=e.get("MISTABRA_BASE_URL"),
            force_fake=_truthy(e.get("MISTABRA_FORCE_FAKE")),
            cache_path=Path(e["MISTABRA_CACHE"]) if e.get("MISTABRA_CACHE") else None,
            standing=e.get("MISTABRA_STANDING", "judged"),
            max_tokens=int(e.get("MISTABRA_MAX_TOKENS") or 4000),
            reasoning=e.get("MISTABRA_REASONING") or None,
            structured=not (e.get("MISTABRA_STRUCTURED", "1").strip().lower() in ("0", "false", "no")),
        )
        for k, v in overrides.items():
            setattr(s, k, v)
        s.__post_init__()
        return s

    def __post_init__(self):
        if not self.model:
            self.model = {"anthropic": "claude-sonnet-4-5", "openai": "gpt-4o", "openrouter": "anthropic/claude-sonnet-4.5"}.get(self.provider, "fake-model")
        if not self.fast_model:
            self.fast_model = self.model
        if self.reasoning not in (None, "none", "low", "medium", "high"):
            raise ValueError(f"reasoning must be none | low | medium | high, not {self.reasoning!r}")
        if self.standing not in ("judged", "noisy_or"):
            raise ValueError(f"standing must be 'judged' or 'noisy_or', not {self.standing!r}")

    @property
    def uses_real_llm(self) -> bool:
        return self.provider != "fake" and bool(self.api_key or self.base_url) and not self.force_fake
