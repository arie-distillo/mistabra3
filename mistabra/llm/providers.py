"""Providers: a deterministic in-process fake (unit tests, offline demos) and two real ones."""
from __future__ import annotations
import json, re
from typing import Protocol
import httpx
from ..config import Settings


class Provider(Protocol):
    name: str
    def complete(self, model: str, system: str, user: str, max_tokens: int) -> tuple[str, tuple[int | None, int | None]]: ...


def _input(user: str) -> dict:
    i = user.find("INPUT_JSON:")
    body = user[i + len("INPUT_JSON:"):]
    body = body[: body.rfind("}") + 1]
    return json.loads(body)


def _task(user: str) -> str:
    return user.split("\n", 1)[0].replace("TASK:", "").strip()


class FakeProvider:
    """Structurally valid, plausible outputs from cheap heuristics. Proves plumbing, not quality.
    `faults` maps task -> raw text to return instead (fault injection for contract tests)."""
    name = "fake"

    def __init__(self, faults: dict[str, str] | None = None):
        self.faults = faults or {}
        self.calls: list[str] = []

    def complete(self, model, system, user, max_tokens):
        task, data = _task(user), _input(user)
        self.calls.append(task)
        if task in self.faults:
            return self.faults[task], (None, None)
        fn = getattr(self, "_" + task.replace(".", "_"))
        return json.dumps(fn(data)), (len(user) // 4, 200)

    # --- ontology ---
    def _ontology_extract(self, d):
        text = d["chunk_text"]
        out = {"objectives": [], "concepts": [], "entity_types": [], "sources": []}
        for line in text.splitlines():
            line = line.strip().lstrip("-*• ")
            m = re.match(r"(Risk|Goal)\s*[A-Z]?\d*\s*[:\-]\s*(.+)", line)
            if m:
                label = m.group(2).split(".")[0].strip()
                out["objectives"].append({"label": label, "polarity": "negative" if m.group(1) == "Risk" else "positive",
                                          "description": m.group(2).strip(), "concerns_entity_types": [], "evidence": line})
                continue
            m = re.match(r"Term\s*[:\-]\s*([^(:]+)(?:\(([^)]*)\))?\s*[:\-]?\s*(.*)", line)
            if m:
                aliases = [a.strip() for a in (m.group(2) or "").split(",") if a.strip()]
                parent = None
                pm = re.search(r"\bkind of ([A-Za-z ]+)", m.group(3) or "")
                if pm:
                    parent = pm.group(1).strip().rstrip(".")
                out["concepts"].append({"label": m.group(1).strip(), "aliases": aliases, "parent_label": parent, "evidence": line})
                continue
            m = re.match(r"Entity type\s*[:\-]\s*([A-Za-z ]+)(?:\s*\(([^)]*)\))?", line)
            if m:
                out["entity_types"].append({"label": m.group(1).strip().lower(), "attributes": [a.strip() for a in (m.group(2) or "").split(",") if a.strip()], "evidence": line})
                continue
            m = re.match(r"Source\s*[:\-]\s*([^(]+)(?:\(credibility\s*([0-9.]+)\))?", line)
            if m:
                out["sources"].append({"label": m.group(1).strip(), "credibility": float(m.group(2)) if m.group(2) else 0.5, "evidence": line})
        return out

    def _ontology_infer_signals(self, d):
        neg = d.get("polarity") == "negative"
        chain = [("target selection", 0.2), ("reconnaissance", 0.4), ("capability building", 0.6), ("rehearsal", 0.8)] if neg else \
                [("interest", 0.2), ("engagement", 0.5), ("commitment", 0.8)]
        sts = []
        for i, (label, prox) in enumerate(chain):
            nxt = [chain[i + 1][0]] if i + 1 < len(chain) else []
            obs = [{"label": f"{label} observable A", "description": f"a visible form of {label}", "prior": 0.2},
                   {"label": f"{label} observable B", "description": f"another form of {label}", "prior": 0.1}]
            if label == "reconnaissance":
                obs = [{"label": "image capture of place", "description": "photographing or filming a location", "prior": 0.15},
                       {"label": "sustained watching", "description": "observing a location or person over time", "prior": 0.3}]
            sts.append({"label": label, "description": f"{label} preceding: {d['objective_label']}", "proximity_to_harm": prox,
                        "precursor_to": nxt, "observables": obs})
        return {"signal_types": sts}

    # --- extraction ---
    def _extract_episode(self, d):
        text = d["episode_text"]
        types = [t["id"] for t in d.get("entity_types", [])] or ["person", "location", "organization"]
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        mentions, seen = [], set()
        for m in re.finditer(r"\b([A-Z][a-zA-Z']+(?:\s+(?:of\s+|the\s+)?[A-Z][a-zA-Z']+)*)\b", text):
            s = m.group(1)
            if s in seen or s.split()[0] in ("A", "An", "The", "On", "At", "From", "In", "After", "Two", "Three", "Same", "Someone", "Over", "During"):
                continue
            seen.add(s)
            low = s.lower()
            if any(k in low for k in ("street", "road", "avenue", "lane", "square", "park", "beach")):
                et = "location"
            elif any(k in low for k in ("centre", "center", "school", "synagogue", "church", "mosque", "hall", "clinic")):
                et = "institution" if "institution" in types else "organization"
            elif any(k in low for k in ("ltd", "inc", "group", "trust", "agency", "company")):
                et = "organization"
            elif re.search(r"\b[A-Z]{2,3}[- ]?\d{2,4}\b", s):
                et = "vehicle"
            else:
                et = "person"
            mentions.append({"surface": s, "entity_type": et if et in types else types[0], "attributes": {}})
        obs_ids = [o["id"] for o in d.get("observables", [])]
        observations = []
        for i, s in enumerate(sents):
            low = s.lower()
            ids = [o for o in obs_ids if any(k in low for k in ("photograph", "filmed", "photo")) and "image" in o] + \
                  [o for o in obs_ids if any(k in low for k in ("watch", "observ", "parked")) and "watch" in o]
            mods = {}
            if re.search(r"\b(night|3am|midnight|2am|late evening)\b", low): mods["time_of_day"] = "night"
            if re.search(r"\b(again|second day|third|three evenings|two days|repeated)\b", low): mods["repetition"] = True
            if re.search(r"\b(covered|hidden|obscured|masked|concealed)\b", low): mods["concealment"] = True
            if re.search(r"\b(telephoto|camera|lens|binoculars)\b", low): mods["tooling"] = True
            observations.append({"text": s, "observable_ids": ids, "modifiers": mods,
                                 "mention_surfaces": [m["surface"] for m in mentions if m["surface"] in s]})
        return {"observations": observations or [{"text": text, "observable_ids": [], "modifiers": {}, "mention_surfaces": []}], "mentions": mentions}

    def _entities_adjudicate(self, d):
        surf = d["mention"]["surface"].lower().replace("'s", "").split()
        for c in d["candidates"]:
            ct = c["canonical_name"].lower().split()
            if surf and ct and surf[-1] == ct[-1]:
                return {"decision": "link", "entity_id": c["id"], "confidence": 0.8, "rationale": "same final name token (fake)"}
        return {"decision": "unresolved", "entity_id": None, "confidence": 0.4, "rationale": "insufficient evidence (fake)"}


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, base_url: str | None = None, timeout: float = 120):
        self.client = httpx.Client(base_url=base_url or "https://api.anthropic.com", timeout=timeout,
                                   headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"})

    def complete(self, model, system, user, max_tokens):
        try:
            r = self.client.post("/v1/messages", json={"model": model, "max_tokens": max_tokens, "temperature": 0,
                                                       "system": system, "messages": [{"role": "user", "content": user}]})
        except httpx.HTTPError as e:
            from .gateway import TransportError
            raise TransportError(str(e)) from None
        if r.status_code >= 500 or r.status_code == 429:
            from .gateway import TransportError
            raise TransportError(f"HTTP {r.status_code}: {r.text[:200]}")
        r.raise_for_status()
        j = r.json()
        text = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
        u = j.get("usage", {})
        return text, (u.get("input_tokens"), u.get("output_tokens"))


class OpenAICompatibleProvider:
    """Chat-completions API (OpenRouter, vLLM, Ollama, ...)."""
    name = "openai_compatible"

    def __init__(self, api_key: str, base_url: str, timeout: float = 120):
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, headers={"Authorization": f"Bearer {api_key}"})

    def complete(self, model, system, user, max_tokens):
        try:
            r = self.client.post("/chat/completions", json={"model": model, "max_tokens": max_tokens, "temperature": 0,
                                                            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        except httpx.HTTPError as e:
            from .gateway import TransportError
            raise TransportError(str(e)) from None
        if r.status_code >= 500 or r.status_code == 429:
            from .gateway import TransportError
            raise TransportError(f"HTTP {r.status_code}: {r.text[:200]}")
        r.raise_for_status()
        j = r.json()
        u = j.get("usage", {})
        return j["choices"][0]["message"]["content"], (u.get("prompt_tokens"), u.get("completion_tokens"))


def make_provider(settings: Settings):
    if not settings.uses_real_model:
        return FakeProvider()
    if settings.provider == "anthropic":
        return AnthropicProvider(settings.api_key, settings.base_url)
    if settings.provider == "openai_compatible":
        if not settings.base_url:
            raise ValueError("openai_compatible provider requires base_url")
        return OpenAICompatibleProvider(settings.api_key, settings.base_url)
    raise ValueError(f"unknown provider {settings.provider}")
