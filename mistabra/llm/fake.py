"""Deterministic fake provider. Structurally valid, plausibly ordered outputs from cheap
heuristics — proves plumbing, not quality. `faults` injects raw text per task for contract tests."""
from __future__ import annotations
import json, re

_INPUT = "INPUT_JSON:"

BENIGN_ROLES = ["estate agent", "lanyard", "tour group", "guide", "journalist", "bylined", "student", "assignment",
                "delivery driver", "marked", "licensed hunter", "organizer", "organiser", "invitations", "commentator", "planning decision"]
SUSPECT_MODS = {"night": ["night", "3am", "midnight", "late evening"], "repetition": ["again", "second day", "three evenings", "several days", "repeated"],
                "concealment": ["covered plates", "hidden", "hood", "masked", "concealed", "anonymous"], "tooling": ["telephoto", "camera", "lens", "binoculars"],
                "explicit_intent": ["hurt", "pay", "attack", "threat", "do something about"]}
REACH_HINTS = [(0.9, ["hurt people", "attack", "accelerant", "circled", "timed how long", "security cameras"]),
               (0.7, ["shotgun", "weapon", "like-minded", "made to pay"]),
               (0.5, ["address", "list of names", "workplaces", "watching the house", "3d", "fly-through", "layout"]),
               (0.3, ["photograph", "filmed", "sketch", "map", "mocking", "photo"])]
ORDINARY = [("photograph|photo|filmed|sketch", [("property listing", "the person has a property interest"), ("hobby album", "the person photographs for a hobby")]),
            ("address|names|workplaces|list", [("event invitation", "a community organiser is collecting contacts"), ("journalism", "a journalist is researching a story")]),
            ("shotgun|accelerant|weapon", [("lawful hunting", "the buyer is a licensed sportsperson")]),
            ("parked|watching|van|car", [("delivery round", "a delivery driver visits repeatedly")]),
            ("pay|hurt|criticised|mocking|like-minded", [("political speech", "the speaker is venting or campaigning")]),
            (".*", [("everyday activity", "nothing beyond ordinary life is happening")])]


def _task(user: str) -> str:
    return user.split("\n", 1)[0].replace("TASK:", "").strip()


def _input(user: str) -> dict:
    body = user[user.find(_INPUT) + len(_INPUT):]
    return json.loads(body[: body.rfind("}") + 1])


def _mods(text: str) -> list[str]:
    low = text.lower()
    return [m for m, kws in SUSPECT_MODS.items() if any(k in low for k in kws)]


def _entities(text: str) -> list[str]:
    stop = {"A", "An", "The", "On", "At", "From", "In", "After", "Two", "Three", "Same", "Someone", "Over", "During", "Karl", "Jens", "Maria"}
    out = []
    for m in re.finditer(r"\b([A-Z][a-zA-Z']+(?:\s+(?:of\s+|the\s+)?[A-Z][a-zA-Z']+)*)\b", text):
        s = m.group(1)
        if s.split()[0] not in stop or len(s.split()) > 1:
            if s not in out: out.append(s)
    return out[:6]


class FakeProvider:
    name = "fake"

    def __init__(self, faults: dict[str, str] | None = None):
        self.faults, self.calls = faults or {}, []

    def complete(self, model, system, user, max_tokens):
        task, d = _task(user), _input(user)
        self.calls.append(task)
        if task in self.faults:
            return self.faults[task], (None, None)
        out = getattr(self, "_" + task.replace(".", "_"))(d)
        return json.dumps(out), (len(user) // 4, 150)

    def _radar_derive(self, d):
        return {"radars": [
            {"query": "Someone photographs, films, sketches, watches or observes a member, a member's home, or a community institution."},
            {"query": "Someone seeks or compiles the address, names, workplaces, schedules or locations of members or institutions."},
            {"query": "Someone expresses hostility, threat, or intent to harm members or the community, or recruits others to act against it."},
            {"query": "Someone acquires weapons, accelerants, or other means, or rehearses access, timing or security at a community site."}]}

    @staticmethod
    def _matches(query: str, text: str) -> bool:
        q, t = query.lower(), text.lower()
        groups = [["photograph", "filmed", "film", "sketch", "watch", "observ"], ["address", "names", "workplaces", "schedule", "map marking", "list of"],
                  ["pay", "hurt", "hostile", "hostility", "like-minded", "mocking", "criticised", "threat"], ["shotgun", "accelerant", "weapon", "timed how long", "security cameras"]]
        community = any(k in t for k in ("street", "centre", "worship", "school", "member", "community", "maple road", "church lane", "klein", "hofer"))
        for g in groups:
            if any(k in q for k in g) and any(k in t for k in g):
                return community or g is groups[2]
        return False

    def _content(self, text: str, risk: str) -> dict:
        low = text.lower()
        target = next((e for e in _entities(text) if any(k in e.lower() for k in ("street", "centre", "worship", "school", "house", "road", "lane"))), None)
        return {"actor": "passer-by" if "passer-by" in low else (next((e for e in _entities(text) if e.split()[0] in ("Karl", "Jens", "Maria", "Thomas", "Anna", "Ruth")), None) or "unknown"),
                "behaviour": " ".join(low.split()[:6]), "target": target, "means": "shotgun" if "shotgun" in low else ("accelerant" if "accelerant" in low else None),
                "time": "night" if "night" in _mods(text) else None, "modifiers": _mods(text), "entities": _entities(text), "summary": text[:120]}

    def _radar_read(self, d):
        rel = self._matches(d["radar_query"], d["episode_text"])
        if not rel:
            return {"relevant": False, "confidence": 0.8, "content": None, "missing": []}
        missing = [] if any(k in d["episode_text"].lower() for k in ("photograph", "filmed", "bought", "posted", "asked", "sketched", "timed", "shared", "wrote", "compiled", "purchased", "parked")) else ["what exactly happened"]
        return {"relevant": True, "confidence": 0.8, "content": self._content(d["episode_text"], d["risk_text"]), "missing": missing}

    def _signal_lambda_check(self, d):
        low, risk = d["episode_text"].lower(), d["risk_text"].lower()
        rel = ("community" in risk and any(k in low for k in ("community centre", "place of worship", "community school", "members of community", "community x")) and len(low) > 40)
        return {"relevant": rel, "confidence": 0.6 if rel else 0.9, "why": "mentions a community institution in an unusual way" if rel else "",
                "content": self._content(d["episode_text"], d["risk_text"]) if rel else None}

    def _trajectory_risk(self, d):
        text, low = d["episode_text"], d["episode_text"].lower()
        reach = next((r for r, kws in REACH_HINTS if any(k in low for k in kws)), 0.3)
        mods = _mods(text)
        fit = min(0.95, 0.4 + 0.15 * len(mods))
        if any(b in low for b in BENIGN_ROLES): fit = max(0.15, fit - 0.25)
        content = d.get("content") or {}
        target = content.get("target") or "the community"
        actor = content.get("actor") or "the actor"
        claim = f"{target} is a target of hostile preparation" if reach < 0.7 else f"{actor} intends to harm the community"
        existing = {h["claim"].lower(): h["id"] for h in d.get("existing_hypotheses", [])}
        hid = existing.get(claim.lower())
        return {"hypothesis_id": hid, "claim": None if hid else claim, "reach": reach, "fit": fit,
                "rationale": f"The episode shows {content.get('behaviour') or 'behaviour'} toward {target}; if {claim.lower()}, this is a step toward: {d['risk_text']}.",
                "entities": content.get("entities", []), "missing": ["occurrence time"] if d.get("occurred_kind") == "unknown" else []}

    def _trajectory_baseline(self, d):
        low = d["episode_text"].lower()
        mods, benign = _mods(d["episode_text"]), any(b in low for b in BENIGN_ROLES)
        base = 0.9 if benign else 0.5
        fit = max(0.1, base - 0.2 * len(mods))
        outs = next(o for pat, o in ORDINARY if re.search(pat, low))
        return {"baselines": [{"ordinary_outcome": o, "claim": c, "rationale": f"If {c}, the episode is ordinary and ends in {o}.", "fit": fit} for o, c in outs][: d.get("max", 3)]}
