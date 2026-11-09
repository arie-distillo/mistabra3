"""The eight input patterns (spec v3 §4.2): what is given stays given, what is missing is completed."""
import pytest

from mistabra.core import HypothesisDraft, SignalContent, SignalDraft, TrajectoryInput
from tests.conftest import DOORS, GROCERIES, HOODED, HOUSE, ep, make_core

NO_RADAR = {"episode.triage": {"relevant": False, "confidence": 0.9, "radar": None, "touches": [], "content": None}}


@pytest.mark.parametrize("kw,n", [({}, 1), ({"signal": SignalDraft(note="x")}, 2), ({"hypothesis": "H"}, 3), ({"risk": "r"}, 4),
                                  ({"signal": SignalDraft(note="x"), "hypothesis": "H"}, 5), ({"signal": SignalDraft(note="x"), "risk": "r"}, 6),
                                  ({"hypothesis": "H", "risk": "r"}, 7), ({"signal": SignalDraft(note="x"), "hypothesis": "H", "risk": "r"}, 8)])
def test_pattern_numbers(kw, n):
    assert TrajectoryInput(episode=ep("t"), **kw).pattern == n


def test_signal_draft_needs_something():
    with pytest.raises(ValueError):
        SignalDraft()


def test_p1_stream_episode_completes_everything(core, mission, op):
    a = core.assess(mission, op, TrajectoryInput(episode=ep(HOUSE)))
    assert a.pattern == 1 and a.groups
    g = next(g for g in a.groups if g.target.risk.risk_id == "attack")
    assert g.target.risk.origin == "predefined" and g.target.signal.origin == "predefined" and g.target.signal.radar
    assert g.target.hypothesis.origin == "predefined" and g.target.hypothesis.id == "H_elm"   # matched, not re-articulated
    assert g.baselines and 0 <= g.target.score <= 1
    assert all(gr.target.risk.risk_id != "old" for gr in a.groups)                          # retired risk is never a target


def test_p1_irrelevant_episode_returns_no_groups(core, mission, op):
    assert core.assess(mission, op, TrajectoryInput(episode=ep(GROCERIES))).groups == ()


def test_p1_triage_sees_held_hypotheses_and_irrelevant_stops_there(mission, op):
    seen = []
    core = make_core({"episode.triage": lambda d: (seen.append(d), {"relevant": False, "confidence": 0.9})[1]})
    a = core.assess(mission, op, TrajectoryInput(episode=ep(GROCERIES)))
    assert a.groups == () and {c.task for c in a.provenance.calls} <= {"radar.derive", "episode.triage"}
    attack = next(d for d in seen if d["risk_text"].startswith("Hateful"))
    assert {h["id"] for h in attack["held_hypotheses"]} == {"H_elm", "H_centre", "H_worship"} and attack["radars"]


def test_p1_lambda_signal_when_no_radar_matches(mission, op):
    core = make_core({"episode.triage": {"relevant": True, "confidence": 0.8, "radar": None, "why": "unanticipated", "content": {"summary": "s"}}})
    a = core.assess(mission, op, TrajectoryInput(episode=ep(HOUSE)))
    assert a.groups and all(g.target.signal.origin == "articulated" and g.target.lambda_signal for g in a.groups)


def test_p1_lambda_hypothesis_when_nothing_stored_fits(core, mission):
    from mistabra.core import OperationalContext
    a = core.assess(mission, OperationalContext(), TrajectoryInput(episode=ep(HOODED)))
    g = next(g for g in a.groups if g.target.risk.risk_id == "attack")
    assert g.target.hypothesis.origin == "articulated" and g.target.lambda_hypothesis and g.target.hypothesis.id is None
    assert all(b.hypothesis.origin == "articulated" and not b.lambda_hypothesis for b in g.baselines)


def test_p2_analyst_signal_given_hypothesis_completed(core, mission, op):
    a = core.assess(mission, op, TrajectoryInput(episode=ep(HOUSE), signal=SignalDraft(note="photography of a member's home")))
    assert a.pattern == 2 and a.groups
    t = a.groups[0].target
    assert t.signal.origin == "given" and t.signal.content.summary == "photography of a member's home"
    assert t.hypothesis.origin in ("predefined", "articulated")


def test_p2_matched_hypothesis_is_not_lambda(core, mission, op):
    a = core.assess(mission, op, TrajectoryInput(episode=ep(HOUSE), signal=SignalDraft(note="photographed Elm Street house"), risk="attack"))
    t = a.groups[0].target
    assert t.hypothesis.origin == "predefined" and not t.lambda_hypothesis      # spec v3 open question 2, as decided


def test_p3_analyst_senses_link_lambda_signal(mission, op):
    core = make_core(NO_RADAR)
    a = core.assess(mission, op, TrajectoryInput(episode=ep(DOORS), hypothesis="H_centre"))
    assert len(a.groups) == 1
    t = a.groups[0].target
    assert t.hypothesis.origin == "given" and t.hypothesis.id == "H_centre"
    assert t.risk.risk_id == "attack" and t.risk.origin == "predefined"
    assert t.signal.origin == "articulated" and t.lambda_signal and not t.lambda_hypothesis


def test_p3_given_hypothesis_always_one_group_even_if_irrelevant(core, mission, op):
    a = core.assess(mission, op, TrajectoryInput(episode=ep(GROCERIES), hypothesis="H_elm"))
    assert len(a.groups) == 1 and a.groups[0].target.hypothesis.id == "H_elm"


def test_p3_draft_hypothesis_without_risk_lands_under_one_risk(core, mission, op):
    a = core.assess(mission, op, TrajectoryInput(episode=ep(HOODED), hypothesis=HypothesisDraft(claim="the Community Centre is being scouted")))
    assert len(a.groups) == 1
    t = a.groups[0].target
    assert t.hypothesis.origin == "given" and t.hypothesis.id is None and t.risk.risk_id in ("attack", "burglary")


def test_p4_risk_given_limits_groups(core, mission, op):
    a = core.assess(mission, op, TrajectoryInput(episode=ep(HOUSE), risk="burglary"))
    assert a.groups and all(g.target.risk.risk_id == "burglary" and g.target.risk.origin == "given" for g in a.groups)
    assert core.assess(mission, op, TrajectoryInput(episode=ep(GROCERIES), risk="attack")).groups == ()


def test_p5_p6_p7_given_elements_kept(core, mission, op):
    s = SignalDraft(content=SignalContent(actor="unknown", behaviour="filming entrances", target="Community Centre"))
    a5 = core.assess(mission, op, TrajectoryInput(episode=ep(HOODED), signal=s, hypothesis="H_centre"))
    a6 = core.assess(mission, op, TrajectoryInput(episode=ep(HOODED), signal=s, risk="attack"))
    a7 = core.assess(mission, op, TrajectoryInput(episode=ep(HOODED), hypothesis="H_centre", risk="attack"))
    t5, t6, t7 = a5.groups[0].target, a6.groups[0].target, a7.groups[0].target
    assert t5.signal.content == s.content and t5.hypothesis.origin == "given" and t5.risk.origin == "predefined"
    assert t6.signal.origin == "given" and t6.risk.origin == "given" and t6.hypothesis.origin in ("predefined", "articulated")
    assert t7.hypothesis.origin == "given" and t7.risk.origin == "given" and t7.signal.origin in ("predefined", "articulated")


def test_p8_full_trajectory_nothing_completed(core, mission, op):
    s = SignalDraft(content=SignalContent(actor="unknown", behaviour="filming entrances at night", target="Community Centre"))
    a = core.assess(mission, op, TrajectoryInput(episode=ep(HOODED), signal=s, hypothesis="H_centre", risk="attack"))
    t = a.groups[0].target
    assert (t.signal.origin, t.hypothesis.origin, t.risk.origin) == ("given", "given", "given")
    assert not t.lambda_signal and not t.lambda_hypothesis and a.groups[0].inconsistency is None


def test_inconsistency_noted_not_corrected(core, mission, op):
    a = core.assess(mission, op, TrajectoryInput(episode=ep(HOUSE), hypothesis="H_elm", risk="burglary"))
    g = a.groups[0]
    assert g.inconsistency and "H_elm" in g.inconsistency
    assert g.target.hypothesis.id == "H_elm" and g.target.risk.risk_id == "burglary"


def test_unknown_references_raise(core, mission, op):
    with pytest.raises(KeyError):
        core.assess(mission, op, TrajectoryInput(episode=ep(HOUSE), hypothesis="nope"))
    with pytest.raises(KeyError):
        core.assess(mission, op, TrajectoryInput(episode=ep(HOUSE), risk="nope"))
    with pytest.raises(ValueError):
        core.assess(mission, op, TrajectoryInput(episode=ep(HOUSE)), scope="wide")
