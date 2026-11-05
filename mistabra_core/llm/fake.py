"""Deterministic fake provider: structurally valid, plausibly ordered outputs from cheap lexical heuristics.

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
    name = "fake"

    def __init__(self, overrides: dict | None = None):
        self.overrides, self.calls = overrides or {}, []

    def complete(self, model, system, user, max_tokens, schema=None):
        task, d = _task(user), _input(user)
        self.calls.append(task)
        if task in self.overrides:
            o = self.overrides[task]
            out = o(d) if callable(o) else o
            return (out if isinstance(out, str) else json.dumps(out)), (None, None)
        out = getattr(self, "_" + task.replace(".", "_"))(d)
        return json.dumps(out), (len(user) // 4, 150)

    # ---- radars
    def _radar_derive(self, d):
        qs = [{"query": f"Episodes showing: {d['risk_text']}"}]
        qs += [{"query": f"Episodes bearing on: {h}"} for h in d.get("hypotheses", [])[:4]]
        return {"radars": qs}

    def _radar_read(self, d):
        text, q = d["episode_text"], d["radar_query"]
        ov = overlap(text, q)
        attention = bool(keys(text) & {"photograph", "address", "hurt", "shotgun", "rain", "budget", "ventilation"})
        rel = ov >= 2 or (ov >= 1 and (bool(mods(text)) or attention))
        return {"relevant": rel, "confidence": 0.8 if rel else 0.85, "content": content(text) if rel else None}

    def _signal_lambda_check(self, d):
        text, low = d["episode_text"], d["episode_text"].lower()
        rel = overlap(text, d["risk_text"]) >= 1 and any(k in low for k in NOVEL) and len(low) > 40
        return {"relevant": rel, "confidence": 0.6 if rel else 0.9, "why": "an unanticipated means bearing on the risk" if rel else "",
                "content": content(text) if rel else None}

    # ---- abduction
    def _trajectory_abduce(self, d):
        ep = d["episode"]["text"]
        sig = d.get("signal")
        sig_text = ep + " " + ((sig or {}).get("note") or (sig or {}).get("summary") or "")
        out_signal = None if sig else content(ep)
        reach = reach_of(ep)
        if d.get("given_hypothesis"):
            claim = d["given_hypothesis"]
            return {"relevant": True, "signal": out_signal, "trajectories": [{
                "hypothesis_id": "given", "claim": None, "match_confidence": 1.0, "reach": reach, "fit": fit_of(sig_text, claim),
                "rationale": f"If {claim.lower()}, the episode is a step toward: {d['risk_text']}", "entities": entities(ep),
                "missing": ["occurrence time"] if not d["episode"].get("occurred") else []}]}
        stored = d.get("stored_hypotheses", []) + d.get("rejected_hypotheses", [])
        scored = sorted(((overlap(sig_text, h["claim"]), h) for h in stored), key=lambda x: -x[0])
        trs = []
        for ov, h in scored:
            if ov >= 2 and len(trs) < d.get("max", 3):
                trs.append({"hypothesis_id": h["id"], "claim": None, "match_confidence": round(min(0.95, 0.45 + 0.1 * ov), 2),
                            "reach": reach, "fit": fit_of(sig_text, h["claim"]),
                            "rationale": f"The episode bears on the stored hypothesis '{h['claim']}'.", "entities": entities(ep), "missing": []})
        relevant = bool(trs) or overlap(sig_text, d["risk_text"]) >= 1 or bool(sig) or bool(mods(ep))
        if not trs and relevant:
            claim = claim_for(sig_text, d["risk_text"])
            trs.append({"hypothesis_id": None, "claim": claim, "match_confidence": 0.0, "reach": reach, "fit": fit_of(sig_text, claim),
                        "rationale": f"If {claim}, the episode is a step toward: {d['risk_text']}", "entities": entities(ep),
                        "missing": ["occurrence time"] if not d["episode"].get("occurred") else []})
        return {"relevant": relevant, "signal": out_signal if relevant else None, "trajectories": trs}

    def _trajectory_baselines(self, d):
        ep = d["episode"]["text"]
        low = ep.lower()
        base = 0.85 if benign(ep) else 0.5
        f = round(max(0.05, base - 0.15 * len(mods(ep))), 3)
        assessed = [{"baseline_id": b["id"], "fit": f, "ordinary_outcome": b["claim"], "rationale": f"If {b['claim']}, the episode is ordinary."}
                    for b in d.get("stored_baselines", [])]
        new = []
        if not assessed or benign(ep):
            outs = next(o for pat, o in ORDINARY if re.search(pat, low))
            have = {b["claim"].lower() for b in d.get("stored_baselines", [])}
            new = [{"claim": c, "ordinary_outcome": o, "rationale": f"If {c}, the episode is ordinary and ends in {o}.", "fit": f}
                   for o, c in outs if c.lower() not in have][: d.get("max_new", 3)]
        return {"assessed": assessed, "new": new}

    # ---- standing
    def _standing_judge(self, d):
        p, n = 1.0, 0
        for t in d["trajectories"]:
            p *= 1 - t["weight"] * t["score"]
            n += t["weight"] > 0 and t["score"] < 0.5
        v = 1 - p + (0.08 if n >= 2 else 0.0)
        return {"value": round(min(1.0, v), 3), "rationale": "fake: weighted noisy-OR with a bonus for several weak trajectories"}
