"""Quickstart: a warehouse keeps losing stock to spoilage, and the analyst's two explanations are wrong.

    python examples/cold_store.py            # with the model configured in .env (see .env.example)
    python examples/cold_store.py --offline  # the offline fake: shows the flow, not the reasoning

What happens: eight short reports arrive one by one. Two of them contradict the analyst's hypotheses (the compressors
are fine; the doors are closed). Three others mean little alone: a new night cleaning contractor, gaps in one cold
room's temperature log at night, and the cleaners charging their floor scrubbers from the socket next to that room's
controller. Together they point to a cause nobody stated: the controller is being unplugged at night. Mistabra should
weaken the analyst's hypotheses, articulate the new one (a Lambda hypothesis), and build its standing as the clues
combine. The contractor report arrives before any hypothesis it supports exists; the Casebook's look-back re-reads it
once one appears.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mistabra_core import Casebook, Core, Document, Episode, MissionContext, Risk, Settings  # noqa: E402

MISSION = MissionContext(
    version="cold-store-1",
    documents=(Document(id="site", title="Site notes", text=(
        "Northgate distribution centre has three cold rooms kept at 2-4 °C. Each room has its own refrigeration controller, "
        "mounted on the wall by the room's door and powered from a standard wall socket. Spoilage is checked at every "
        "morning stock count.")),),
    risks=(Risk(id="spoilage", text="Chilled stock at the Northgate distribution centre spoils again."),),
)

ANALYST = [
    "Stock spoils because one of the refrigeration compressors is failing.",
    "Stock spoils because staff leave the cold-room doors open during busy shifts.",
]

EPISODES = [
    ("e1", "maintenance contractor", "The quarterly compressor service found both compressors within specification and refrigerant pressure normal."),
    ("e2", "door alarm log", "Door-open alarms in the last month were all during day shifts, and every door was closed again within two minutes."),
    ("e3", "facilities", "Facilities signed a new night cleaning contractor in March; the crew works from midnight to 4 a.m."),
    ("e4", "transport desk", "The delivery van to the Eastside store now takes the ring road because of roadworks."),
    ("e5", "temperature logger", "The temperature logger in Cold Room 2 shows gaps of 40 to 90 minutes between 1 and 3 a.m. on four nights in the last two weeks."),
    ("e6", "night supervisor", "The cleaners charge their floor scrubbers from the socket by Cold Room 2's door, the only free socket on that wall."),
    ("e7", "stock count", "Every spoiled batch in the last two weeks came from Cold Room 2."),
    ("e8", "canteen", "The canteen added a vegetarian option to the lunch menu."),
]


def main(offline: bool) -> None:
    settings = Settings(force_fake=True) if offline else Settings.from_env()
    core = Core(settings)
    if core.provider.name == "fake":
        print("Offline fake model: this shows the flow of a case, not real reasoning. Configure a model in .env for that.\n")
    case = Casebook(core, MISSION)
    for claim in ANALYST:
        case.add_hypothesis(claim, risk="spoilage")

    for eid, source, text in EPISODES:
        got = case.ingest(Episode(id=eid, text=text, source=source, credibility=0.8))
        print(f"[{eid}] {text}")
        if not got.assessment.groups:
            print("      bears on nothing (kept for the look-back)")
        for g, hid in zip(got.assessment.groups, got.hypothesis_ids):
            t = g.target
            new = "  NEW" if hid in got.created else ""
            print(f"      {t.stance:9} {hid:4} score {t.score:.2f}  standing {g.standing.without:.2f} -> {g.standing.value:.2f}"
                  f"{new}  {t.hypothesis.claim}")
        for lb in got.lookback:
            print(f"      look-back: earlier [{lb.episode_id}] now {lb.group.target.stance} {lb.hypothesis_id}")

    print("\nWhere the case stands:")
    for h in sorted(case.hypotheses("spoilage"), key=lambda h: -(h.standing or 0)):
        tag = "analyst" if h.author == "analyst" else ("new, refines " + h.refines if h.refines else "new")
        print(f"  {h.standing or 0:.2f}  {h.id:4} ({tag}) {h.claim}")


if __name__ == "__main__":
    main(offline="--offline" in sys.argv)
