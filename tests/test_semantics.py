"""Semantics (spec v3 §5): scope, detail, reuse and rejection, score in isolation, standing, provenance, statelessness."""
import pytest

from mistabra_core import OperationalContext, TrajectoryInput
from mistabra_core.scoring import noisy_or, score_from
from tests.conftest import HOODED, HOUSE, ep, make_core, traj


# ---------------------------------------------------------------- scope and detail
def test_scope_open_adds_other_groups_given_first(core, mission, op):
    given = core.assess(mission, op, TrajectoryInput(episode=ep(HOUSE), hypothesis="H_centre"))
    opened = core.assess(mission, op, TrajectoryInput(episode=ep(HOUSE), hypothesis="H_centre"), scope="open")
    assert len(given.groups) == 1 and len(opened.groups) > 1
    assert opened.groups[0].target.hypothesis.id == "H_centre" and opened.groups[0].target.hypothesis.source == "given"
    assert any(g.target.hypothesis.id == "H_elm" for g in opened.groups[1:])


def test_scope_has_no_effect_with_nothing_pointed(core, mission, op):
    a = core.assess(mission, op, TrajectoryInput(episode=ep(HOUSE)))
    b = core.assess(mission, op, TrajectoryInput(episode=ep(HOUSE)), scope="open")
    assert [g.target.hypothesis.claim for g in a.groups] == [g.target.hypothesis.claim for g in b.groups]


def test_detail_scores_same_values_no_rationale(core, mission, op):
    full = core.assess(mission, op, TrajectoryInput(episode=ep(HOUSE)))
    short = core.assess(mission, op, TrajectoryInput(episode=ep(HOUSE)), detail="scores")
    assert [(g.target.score, g.standing.value) for g in full.groups] == [(g.target.score, g.standing.value) for g in short.groups]
    assert all(g.target.rationale is None and g.standing.rationale is None and all(b.rationale is None for b in g.baselines) for g in short.groups)
    assert any(g.target.rationale for g in full.groups)


# ---------------------------------------------------------------- reuse, matching, rejection
def test_stored_baselines_reused_and_new_only_when_uncovered(core, mission, op):
    g = core.assess(mission, op, TrajectoryInput(episode=ep(HOUSE), hypothesis="H_elm")).groups[0]
    stored = [b for b in g.baselines if b.hypothesis.source == "predefined"]
    assert [b.hypothesis.id for b in stored] == ["B_listing"]
    assert not any(b.hypothesis.source == "articulated" for b in g.baselines)   # the stored one covers it: nothing new


def test_low_match_confidence_is_treated_as_new(mission, op):
    over = {"trajectory.abduce": {"relevant": True, "signal": None, "trajectories": [
        {"hypothesis_id": "H_elm", "claim": "Elm Street homes are being watched by someone", "match_confidence": 0.3, "reach": 0.3, "fit": 0.5}]}}
    t = make_core(over).assess(mission, op, TrajectoryInput(episode=ep(HOUSE), risk="attack")).groups[0].target
    assert t.hypothesis.source == "articulated" and t.hypothesis.claim.startswith("Elm Street homes")


def test_rejected_claim_never_rearticulated(mission, op):
    over = {"trajectory.abduce": {"relevant": True, "signal": None, "trajectories": [
        {"hypothesis_id": None, "claim": "The Place of Worship is a target of hostile preparation", "reach": 0.3, "fit": 0.6}]},
        "radar.read": {"relevant": True, "confidence": 0.9, "content": {"summary": "night photography of entrances"}}}
    core = make_core(over)
    ti = TrajectoryInput(episode=ep("Someone photographed the Place of Worship entrances at night."), risk="attack")
    assert core.assess(mission, op, ti).groups == ()                          # scope given: rejected hypothesis not reported
    opened = core.assess(mission, op, ti, scope="open")                            # scope open: reported, marked rejected
    g = next(g for g in opened.groups if g.target.hypothesis.id == "H_worship")
    assert g.target.hypothesis.status == "rejected" and g.target.hypothesis.source == "predefined"


def test_rejected_hypothesis_given_is_reported_with_status(core, mission, op):
    g = core.assess(mission, op, TrajectoryInput(episode=ep(HOUSE), hypothesis="H_worship")).groups[0]
    assert g.target.hypothesis.status == "rejected" and g.target.hypothesis.source == "given"


# ---------------------------------------------------------------- score
def test_score_formula_documented_and_bounded():
    base = dict(reach=0.3, fit=0.4, gap=0.0, specificity=1.0, credibility=0.8)
    s0 = score_from(**base)
    assert 0 <= s0 <= 1
    assert score_from(**{**base, "fit": 0.9}) > s0 and score_from(**{**base, "gap": 0.5}) > s0 and score_from(**{**base, "reach": 0.9}) > s0
    assert score_from(**{**base, "gap": -0.5}) < s0 and score_from(**{**base, "credibility": 0.4}) < s0
    assert score_from(reach=1, fit=1, gap=1, specificity=1, credibility=1) == 1.0


def test_score_is_independent_of_other_trajectories(core, mission, op):
    ti = TrajectoryInput(episode=ep(HOUSE, eid="new"), hypothesis="H_elm")
    alone = core.assess(mission, op, ti).groups[0]
    crowded = op.model_copy(update={"trajectories": tuple(traj(f"t{i}", "H_elm", 0.9, "confirmed") for i in range(4))})
    busy = core.assess(mission, crowded, ti).groups[0]
    assert alone.target.score == busy.target.score
    assert busy.standing.value > alone.standing.value and busy.standing.without > 0 == alone.standing.without


# ---------------------------------------------------------------- standing
@pytest.mark.parametrize("strategy", ["judged", "noisy_or"])
def test_standing_weights_by_status_and_delta(mission, op, strategy):
    core = make_core(standing=strategy, proposed_weight=0.5)
    ctx = op.model_copy(update={"trajectories": (traj("c", "H_elm", 0.4, "confirmed"), traj("p", "H_elm", 0.4, "proposed"),
                                                 traj("r", "H_elm", 0.9, "rejected"))})
    s = core.assess(mission, ctx, TrajectoryInput(episode=ep(HOUSE, eid="new"), hypothesis="H_elm")).groups[0].standing
    w = {u.trajectory_id: (u.status, u.weight) for u in s.used}
    assert w["c"] == ("confirmed", 1.0) and w["p"] == ("proposed", 0.5) and w["r"] == ("rejected", 0.0)
    assert w["this:new"][0] == "this" and w["this:new"][1] == 0.5
    assert s.delta == round(s.value - s.without, 3) and s.value >= s.without > 0
    if strategy == "noisy_or":
        assert s.without == noisy_or([{"weight": 1.0, "score": 0.4}, {"weight": 0.5, "score": 0.4}, {"weight": 0.0, "score": 0.9}])


def test_rejected_trajectories_alone_give_no_standing(mission, op):
    core = make_core(standing="noisy_or")
    ctx = op.model_copy(update={"trajectories": (traj("r1", "H_elm", 0.9, "rejected"), traj("r2", "H_elm", 0.9, "rejected"))})
    s = core.assess(mission, ctx, TrajectoryInput(episode=ep(HOUSE, eid="new"), hypothesis="H_elm")).groups[0].standing
    assert s.without == 0.0


def test_analysts_own_trajectory_weighs_fully(core, mission, op):
    from mistabra_core import SignalDraft
    s = core.assess(mission, op, TrajectoryInput(episode=ep(HOUSE, eid="own"), signal=SignalDraft(note="n"), hypothesis="H_elm")).groups[0].standing
    assert {u.trajectory_id: u.weight for u in s.used}["this:own"] == 1.0


def test_reassessing_an_episode_does_not_count_it_twice(core, mission, op):
    ctx = op.model_copy(update={"trajectories": (traj("same", "H_elm", 0.9, "confirmed", eid="e1"),)})
    s = core.assess(mission, ctx, TrajectoryInput(episode=ep(HOUSE, eid="e1"), hypothesis="H_elm")).groups[0].standing
    assert "same" not in {u.trajectory_id for u in s.used} and s.without == 0.0


def test_articulated_hypothesis_has_standing_from_this_episode_only(core, mission):
    g = next(g for g in core.assess(mission, OperationalContext(), TrajectoryInput(episode=ep(HOODED))).groups if g.target.risk.risk_id == "attack")
    assert g.standing.without == 0.0 and g.standing.value > 0 and [u.status for u in g.standing.used] == ["this"]


# ---------------------------------------------------------------- provenance, determinism, statelessness
def test_provenance_lists_what_was_read_and_called(core, mission, op):
    ctx = op.model_copy(update={"trajectories": (traj("c", "H_elm", 0.4, "confirmed"),)})
    a = core.assess(mission, ctx, TrajectoryInput(episode=ep(HOUSE, eid="new"), hypothesis="H_elm"))
    p = a.provenance
    assert p.implementation == "reference" and p.mission_version == "m1" and p.deterministic
    assert {"H_elm", "B_listing", "c"} <= set(p.operational_items) and p.operational_items["H_elm"] == 1
    assert {"trajectory.abduce", "trajectory.baselines", "standing.judge"} <= set(p.prompts) and p.calls
    assert len(p.input_hash) == 16


def test_identical_inputs_identical_results(core, mission, op):
    ti = TrajectoryInput(episode=ep(HOUSE))
    a, b = core.assess(mission, op, ti), core.assess(mission, op, ti)
    assert a.model_dump(exclude={"provenance"}) == b.model_dump(exclude={"provenance"}) and a.provenance.input_hash == b.provenance.input_hash
    fresh = make_core().assess(mission, op, ti)
    assert fresh.model_dump(exclude={"provenance"}) == a.model_dump(exclude={"provenance"})


def test_inputs_are_not_mutated(core, mission, op):
    before = (mission.model_dump(), op.model_dump())
    core.assess(mission, op, TrajectoryInput(episode=ep(HOUSE)), scope="open")
    assert (mission.model_dump(), op.model_dump()) == before


def test_radars_follow_the_stored_hypotheses(core, mission, op):
    core.assess(mission, op, TrajectoryInput(episode=ep(HOUSE)))
    n = len(core._radars)
    core.assess(mission, op, TrajectoryInput(episode=ep(HOODED)))
    assert len(core._radars) == n                              # same contexts: radars reused
    more = op.model_copy(update={"hypotheses": op.hypotheses[:1]})
    core.assess(mission, more, TrajectoryInput(episode=ep(HOUSE)))
    assert len(core._radars) > n                               # stored hypotheses changed: radars re-derived


# ---------------------------------------------------------------- counter-evidence (stance)
WEAKENS = {"trajectory.abduce": {"relevant": True, "signal": None, "trajectories": [],
                                 "stored": [{"hypothesis_id": "H_elm", "stance": "weakens", "fit": 0.1, "rationale": "the visitor was the new tenant"},
                                            {"hypothesis_id": "H_centre", "stance": "unrelated"}]}}


def test_counter_evidence_attaches_as_weakens(mission, op):
    a = make_core(WEAKENS).assess(mission, op, TrajectoryInput(episode=ep(HOUSE), risk="attack"))
    assert len(a.groups) == 1
    t = a.groups[0].target
    assert t.hypothesis.id == "H_elm" and t.hypothesis.source == "predefined" and t.stance == "weakens" and t.score == 0.0
    assert "tenant" in t.rationale


@pytest.mark.parametrize("strategy", ["judged", "noisy_or"])
def test_counter_evidence_lowers_standing(mission, op, strategy):
    ctx = op.model_copy(update={"trajectories": (traj("c1", "H_elm", 0.6, "confirmed"), traj("c2", "H_elm", 0.5, "confirmed"))})
    s = make_core(WEAKENS, standing=strategy).assess(mission, ctx, TrajectoryInput(episode=ep(HOUSE, eid="new"), risk="attack")).groups[0].standing
    if strategy == "noisy_or":
        assert s.value < s.without and s.delta < 0
    assert [u.status for u in s.used][-1] == "this"


def test_supporting_trajectories_listed_before_counter_evidence(mission, op):
    over = {"trajectory.abduce": {"relevant": True, "signal": None,
                                  "trajectories": [{"hypothesis_id": "H_centre", "match_confidence": 0.9, "fit": 0.6, "reach": 0.3}],
                                  "stored": [{"hypothesis_id": "H_elm", "stance": "weakens", "fit": 0.1}]}}
    a = make_core(over).assess(mission, op, TrajectoryInput(episode=ep(HOODED), risk="attack"))
    assert [(g.target.hypothesis.id, g.target.stance) for g in a.groups] == [("H_centre", "supports"), ("H_elm", "weakens")]
