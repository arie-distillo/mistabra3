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

    LLM setup is uniform across providers (all calls go through PydanticAI): `provider` picks the model class, `model` (reasoning tier) and
    `fast_model` (operational tier) pick the models, `api_key` and `base_url` pass through.
    `force_fake` wins over any key, so tests can never reach a paid model.
    """
    provider: str = "fake"                  # fake | anthropic | openai (any OpenAI-compatible endpoint) | openrouter
    model: str = ""                         # reasoning tier: assessment (and standing, when judged)
    fast_model: str = ""                    # operational tier: episode triage (defaults to model)
    api_key: str | None = None
    base_url: str | None = None
    force_fake: bool = False
    timeout_s: float = 120.0
    cache_path: Path | None = None          # optional on-disk cache of model outputs (SQLite); in-memory otherwise
    max_tokens: int = 4000                  # per call; thinking models count their reasoning against it
    reasoning: str | None = None            # None = model default | none | low | medium | high (openai / openrouter)
    failure_log: Path | None = None         # append failed model calls (prompt, raw output, error) as JSON lines
    call_log: Path | None = None            # append every model call (prompt, validated output; cached ones too) as JSON lines
    temperature: float = 0.0                # sampling temperature for every call
    cache_salt: str = ""                    # part of the cache key: a different salt gives a fresh, independently cached run
    concurrency: int = 8                    # model requests in flight at once (independent calls run in parallel)

    standing: str = "noisy_or"              # noisy_or (computed; default since 2026-11-06) | judged (the reasoning model weighs them)
    max_held: int = 6                       # held hypotheses shown to the assessment: those triage flagged, then the closest
    max_calls: int | None = None            # stop with an error after this many model calls (cache hits are free)
    sharpen_after: int = 3                  # supporting facts (this episode included) before a held hypothesis is sharpened
    max_evidence: int = 6                   # facts already attached to a hypothesis shown when assessing a new episode
    new_hypothesis_score: str = "fit"        # fit (one formula for all) | margin (option B: a hypothesis written from this
                                            # episode is scored by its margin over the ordinary explanations and reach only)
    unconfirmed_cap: float = 0.8            # standing ceiling while nothing attached is confirmed by the analyst
    proposed_weight: float = 0.5            # weight of a `proposed` trajectory in standing; confirmed = 1, rejected = 0
    relevance_threshold: float = 0.5        # triage confidence needed to proceed
    match_threshold: float = 0.6            # a 'supports' verdict on a held hypothesis needs this confidence
    max_baselines: int = 3
    max_trajectories: int = 2               # new hypotheses per episode and risk (held ones get a verdict each)
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
            standing=e.get("MISTABRA_STANDING", "noisy_or"),
            max_tokens=int(e.get("MISTABRA_MAX_TOKENS") or 4000),
            reasoning=e.get("MISTABRA_REASONING") or None,
            failure_log=Path(e["MISTABRA_FAILURE_LOG"]) if e.get("MISTABRA_FAILURE_LOG") else None,
            concurrency=int(e.get("MISTABRA_CONCURRENCY") or 8),
            temperature=float(e.get("MISTABRA_TEMPERATURE") or 0.0),
            new_hypothesis_score=e.get("MISTABRA_NEW_HYPOTHESIS_SCORE", "fit"),
            max_calls=int(e["MISTABRA_MAX_CALLS"]) if e.get("MISTABRA_MAX_CALLS") else None,
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
        if self.new_hypothesis_score not in ("fit", "margin"):
            raise ValueError(f"new_hypothesis_score must be 'fit' or 'margin', not {self.new_hypothesis_score!r}")
        for name in ("relevance_threshold", "match_threshold", "proposed_weight", "unconfirmed_cap"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, not {v}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be between 0 and 2, not {self.temperature}")
        for name in ("max_trajectories", "max_baselines", "max_attached", "max_evidence", "sharpen_after", "concurrency", "max_tokens", "max_held"):
            v = getattr(self, name)
            if not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be a whole number of at least 1, not {v!r}")

    @property
    def uses_real_llm(self) -> bool:
        return self.provider != "fake" and bool(self.api_key or self.base_url) and not self.force_fake
