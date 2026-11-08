"""Deterministic fake model: structurally valid, plausibly ordered outputs from cheap lexical heuristics.

It proves plumbing, not quality: every prompt is answered, the same input always gives the same output, and the
answers move in the right direction (concealment raises fit, a stated benign role lowers it, overlapping words
match a stored hypothesis). Numbers it produces in the testbed are not measurements of the method.

`overrides` maps a task to a function (variables -> dict) or a raw string, for contract and functional tests.
Adapted from the PoC fake; the lexicon is broadened so the testbed's scenarios in other domains get through.
"""
from __future__ import annotations

import json
import re

_INPUT = "INPUT_JSON:"
_STOP = set("""a an the and or of to in on at for from by with without into onto over under after before during while
this that these those it its is are was were be been being has have had do does did not no nor but so than then
there their them they he she his her him we our you your who whom which what when where why how all any each
some such same other another one two three four five several many more most much very just only also again
about across against along among around between within near off out up down since until upon via per
member members someone somebody person people man woman passer-by""".split())

# synonym groups: tokens mapped to a shared key so that paraphrases overlap
_GROUPS = [
    "photograph photographed photographs photo photos filmed film filming sketch sketched watch watched watching observe observed observing camera telephoto lens binoculars recorded recording",
    "house home homes houses residence flat apartment",
    "address addresses names workplaces schedule schedules list compiled compiling",
    "hurt harm attack attacked threat threatened threats pay violence kill hostile hostility hate hateful",
    "shotgun weapon weapons accelerant firearm knife explosive",
    "rain rainfall storm downpour flood flooding floods water runoff drainage drain drains sewer",
    "budget funding cut cuts underfunded maintenance",
    "ventilation intake airflow pressure pascal dust particulate fibres fibers cellulose contamination contaminated airborne",
    "catalyst expired expiration",
    "mixing mixer rpm speed",
    "temperature heat thermostat sensor overheating",
    "timer bake baking unattended oven ovens",
    "flour batch supplier",
    "sabotage deliberate grievance",
    "centre center",
]
_KEY = {w: g.split()[0] for g in _GROUPS for w in g.split()}

BENIGN_ROLES = ["estate agent", "lanyard", "tour group", "guide", "journalist", "bylined", "student", "assignment",
                "delivery driver", "marked", "licensed hunter", "organizer", "organiser", "invitations", "commentator",
                "planning decision", "council", "parent", "routine", "scheduled", "within normal", "within specification"]
SUSPECT_MODS = {"night": ["night", "3am", "midnight", "late evening"],
                "repetition": ["again", "second day", "three evenings", "several days", "repeated", "consecutive"],
                "concealment": ["covered plates", "hidden", "hood", "masked", "concealed", "anonymous"],
                "tooling": ["telephoto", "binoculars", "drone", "3d", "fly-through"],
                "explicit_intent": ["hurt", "made to pay", "attack", "threat", "do something about"],
                "anomaly": ["spike", "blocked", "drop below", "failed", "cut", "shortfall", "exceeded", "unresponsive"]}
NOVEL = ["3d", "fly-through", "model", "reconstructed", "drone", "unlocked", "doors"]
REACH_HINTS = [(0.9, ["hurt people", "attack", "accelerant", "circled", "timed how long", "security cameras", "flood warning"]),
               (0.7, ["shotgun", "weapon", "like-minded", "made to pay", "overflow"]),
               (0.5, ["address", "list of names", "workplaces", "watching the house", "3d", "fly-through", "layout", "budget", "blocked"]),
               (0.3, ["photograph", "filmed", "sketch", "map", "mocking", "photo", "forecast"])]
ORDINARY = [("photograph|photo|filmed|sketch", [("property listing", "the person has a property interest"), ("hobby album", "the person photographs for a hobby")]),
            ("address|names|workplaces|list", [("event invitation", "a community organiser is collecting contacts"), ("journalism", "a journalist is researching a story")]),
            ("shotgun|accelerant|weapon", [("lawful hunting", "the buyer is a licensed sportsperson")]),
            ("parked|watching|van|car", [("delivery round", "a delivery driver visits repeatedly")]),
            ("pay|hurt|criticised|mocking|like-minded", [("political speech", "the speaker is venting or campaigning")]),
            ("rain|storm|budget|drain", [("routine variation", "this is within the normal seasonal or budget range")]),
            (".*", [("everyday activity", "nothing beyond ordinary operations is happening")])]


def _task(user: str) -> str:
    return user.split("\n", 1)[0].replace("TASK:", "").strip()


def _input(user: str) -> dict:
    body = user[user.find(_INPUT) + len(_INPUT):]
    return json.loads(body[: body.rfind("}") + 1])


def keys(text: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z0-9][a-z0-9'-]+", (text or "").lower()):
        w = w.strip("'-")
        if len(w) < 3 or w in _STOP:
            continue
        out.add(_KEY.get(w, _KEY.get(w.rstrip("s"), w[:6])))
    return out


def overlap(a: str, b: str) -> int:
    return len(keys(a) & keys(b))


def mods(text: str) -> list[str]:
    low = text.lower()
    return [m for m, kws in SUSPECT_MODS.items() if any(k in low for k in kws)]


def benign(text: str) -> bool:
    low = text.lower()
    return any(b in low for b in BENIGN_ROLES)


def entities(text: str) -> list[str]:
    stop = {"A", "An", "The", "On", "At", "From", "In", "After", "Two", "Three", "Same", "Someone", "Over", "During", "If", "Batch", "Standard"}
    out = []
    for m in re.finditer(r"\b([A-Z][a-zA-Z']+(?:\s+(?:of\s+|the\s+)?[A-Z][a-zA-Z']+)*)\b", text):
        s = m.group(1)
        if (s.split()[0] not in stop or len(s.split()) > 1) and s not in out:
            out.append(s)
    return out[:6]


def _target(text: str) -> str | None:
    places = ("street", "centre", "center", "worship", "school", "house", "road", "lane", "district", "districts", "chamber", "room", "plant")
    for e in entities(text):
        if any(k in e.lower() for k in places):
            return e
    m = re.search(r"\b(?:the|a)\s+([a-z]+(?:\s+[a-z]+)?\s+(?:district|districts|room|valve|intake|sensor|oven|batch))\b", text.lower())
    return m.group(1) if m else None


def content(text: str) -> dict:
    low = text.lower()
    names = [e for e in entities(text) if e.split()[0] in ("Karl", "Jens", "Maria", "Thomas", "Anna", "Ruth", "Hofer", "Vance", "Martinez")]
    return {"actor": "passer-by" if "passer-by" in low else (names[0] if names else "unknown"),
            "behaviour": " ".join(low.split()[:7]), "target": _target(text),
            "means": next((m for m in ("shotgun", "accelerant", "telephoto lens", "drone") if m in low), None),
            "time": "night" if "night" in mods(text) else None, "modifiers": mods(text), "entities": entities(text), "summary": text[:140]}


def reach_of(text: str) -> float:
    low = text.lower()
    return next((r for r, kws in REACH_HINTS if any(k in low for k in kws)), 0.3)


def fit_of(text: str, claim: str) -> float:
    f = 0.3 + 0.12 * len(mods(text)) + 0.04 * min(overlap(text, claim), 5)
    if benign(text):
        f -= 0.25
    return round(max(0.05, min(0.95, f)), 3)


def claim_for(text: str, risk: str) -> str:
    t = _target(text) or next(iter(entities(text)), None)
    low_risk = risk.lower()
    if any(w in low_risk for w in ("attack", "harm", "hostile", "violence")):
        return f"{t or 'the community'} is a target of hostile preparation"
    subject = t or " ".join(sorted(keys(text))[:3])
    return f"{subject} is contributing to: {risk.rstrip('.')}"


class FakeProvider:
    """A PydanticAI FunctionModel: answers each prompt through the output tool, like a real model would, so the
    gateway's whole path (output tool, validation, retries, cache, records) runs offline."""
    name = "fake"

    def __init__(self, overrides: dict | None = None):
        self.overrides, self.calls = overrides or {}, []

    def model(self, tier: str):
        from pydantic_ai.models.function import FunctionModel
        return FunctionModel(self._respond, model_name="fake-model")

    def model_settings(self) -> dict:
        return {}

    def _respond(self, messages, info):
        from pydantic_ai.messages import ModelResponse, ToolCallPart, UserPromptPart
        user = next(part.content for m in messages for part in getattr(m, "parts", []) if isinstance(part, UserPromptPart))
        task, d = _task(user), _input(user)
        self.calls.append(task)
        if task in self.overrides:
            o = self.overrides[task]
            out = o(d) if callable(o) else o
        else:
            out = getattr(self, "_" + task.replace(".", "_"))(d)
        args = out if isinstance(out, str) else json.dumps(out)
        return ModelResponse(parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=args)])

    # ---- radars
    def _radar_derive(self, d):
        qs = [{"query": f"Episodes showing: {d['risk_text']}"}]
        qs += [{"query": f"Episodes bearing on: {h}"} for h in d.get("hypotheses", [])[:4]]
        return {"radars": qs}

    # ---- triage
    def _episode_triage(self, d):
        text = d["episode"]["text"]
        touches = [h["id"] for h in d.get("held_hypotheses", []) if overlap(text, h["claim"]) >= 2]
        attention = bool(keys(text) & {"photograph", "address", "hurt", "shotgun", "rain", "budget", "ventilation"})
        radar = next((r["index"] for r in d.get("radars", []) if (ov := overlap(text, r["query"])) >= 2 or (ov >= 1 and (bool(mods(text)) or attention))), None)
        low = text.lower()
        novel = overlap(text, d["risk_text"]) >= 1 and any(k in low for k in NOVEL) and len(low) > 40
        rel = radar is not None or bool(touches) or novel
        return {"relevant": rel, "confidence": 0.8 if rel else 0.85, "radar": radar, "touches": touches,
                "why": "fake triage" if rel else "", "content": content(text) if rel else None}

    # ---- assessment
    def _baselines(self, ep, stored, max_new):
        low = ep.lower()
        f = round(max(0.05, (0.85 if benign(ep) else 0.5) - 0.15 * len(mods(ep))), 3)
        assessed = [{"baseline_id": b["id"], "fit": f, "ordinary_outcome": b["claim"], "rationale": f"If {b['claim']}, the episode is ordinary."}
                    for b in stored]
        new = []
        if not assessed or benign(ep):
            outs = next(o for pat, o in ORDINARY if re.search(pat, low))
            have = {b["claim"].lower() for b in stored}
            new = [{"claim": c, "ordinary_outcome": o, "rationale": f"If {c}, the episode is ordinary and ends in {o}.", "fit": f}
                   for o, c in outs if c.lower() not in have][: max_new if not stored else 1]
        return {"assessed": assessed, "new": new}

    def _episode_assess(self, d):
        ep = d["episode"]["text"]
        sig = d.get("signal")
        sig_text = ep + " " + ((sig or {}).get("note") or (sig or {}).get("summary") or "")
        out_signal = None if (sig or d.get("detected_signal")) else content(ep)
        reach, mb = reach_of(ep), d.get("max_new_baselines", 3)
        missing = ["occurrence time"] if not d["episode"].get("occurred") else []
        g = d.get("given_hypothesis")
        if g:
            return {"relevant": True, "signal": out_signal, "given": {
                "stance": "supports", "fit": fit_of(sig_text, g["claim"]), "reach": reach, "entities": entities(ep), "missing": missing,
                "rationale": f"If {g['claim'].lower()}, the episode is a step toward: {d['risk_text']}",
                "baselines": self._baselines(ep, g.get("baselines", []), mb)}, "stored": [], "new": []}
        negating = any(w in ep.lower() for w in ("cleared", "completed", "upgraded", "restored", "within normal", "ruled out"))
        stored = []
        for h in d.get("held_hypotheses", []):
            ov = overlap(sig_text, h["claim"])
            stance = "unrelated" if ov < 2 else ("weakens" if negating else "supports")
            v = {"hypothesis_id": h["id"], "stance": stance, "confidence": round(min(0.95, 0.45 + 0.1 * ov), 2),
                 "fit": 0.1 if stance == "weakens" else fit_of(sig_text, h["claim"]), "reach": reach, "missing": [],
                 "rationale": f"The episode bears on the held hypothesis '{h['claim']}' ({stance})."}
            if stance == "supports":
                v["baselines"] = self._baselines(ep, h.get("baselines", []), mb)
            stored.append(v)
        relevant = any(v["stance"] != "unrelated" for v in stored) or overlap(sig_text, d["risk_text"]) >= 1 or bool(sig) or bool(mods(ep))
        new = []
        if relevant and not any(v["stance"] != "unrelated" for v in stored) and d.get("max_new", 2) > 0:
            claim = claim_for(sig_text, d["risk_text"])
            new.append({"claim": claim, "fit": fit_of(sig_text, claim), "reach": reach, "entities": entities(ep), "missing": missing,
                        "rationale": f"If {claim}, the episode is a step toward: {d['risk_text']}", "baselines": self._baselines(ep, [], mb)})
        return {"relevant": relevant, "signal": out_signal if relevant else None, "stored": stored, "new": new}

    def _lookback_select(self, d):
        return {"indices": [e["index"] for e in d.get("episodes", [])][: d.get("max", 3)], "why": "fake: the most recent ones"}

    def _hypothesis_sharpen(self, d):
        return {"sharper_claim": None, "why": "fake: never sharpens"}

    # ---- standing
    def _standing_judge(self, d):
        def nor(ts):
            p, damp, n = 1.0, 1.0, 0
            for t in ts:
                if t.get("stance") == "weakens":
                    damp *= 1 - 0.5 * t["weight"]
                    continue
                p *= 1 - t["weight"] * t["score"]
                n += t["weight"] > 0 and t["score"] < 0.5
            return round(min(1.0, (1 - p + (0.08 if n >= 2 else 0.0)) * damp), 3)
        return {"standings": [{"key": h["key"], "before": nor(h["trajectories"]), "value": nor(h["trajectories"] + [h["this"]]),
                               "rationale": "fake: weighted noisy-OR with a bonus for several weak trajectories"} for h in d["hypotheses"]]}
