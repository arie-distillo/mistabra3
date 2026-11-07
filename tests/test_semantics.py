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
    assert opened.groups[0].target.hypothesis.id == "H_centre" and opened.groups[0].target.hypothesis.origin == "given"
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
    stored = [b for b in g.baselines if b.hypothesis.origin == "predefined"]
    assert [b.hypothesis.id for b in stored] == ["B_listing"]
    assert not any(b.hypothesis.origin == "articulated" for b in g.baselines)   # the stored one covers it: nothing new


def test_low_confidence_verdict_is_not_a_match(mission, op):
    over = {"episode.assess": {"stored": [{"hypothesis_id": "H_elm", "stance": "supports", "confidence": 0.3, "fit": 0.5, "reach": 0.3}],
                               "new": [{"claim": "Elm Street homes are being watched by someone", "fit": 0.5, "reach": 0.3}]}}
    a = make_core(over).assess(mission, op, TrajectoryInput(episode=ep(HOUSE), risk="attack"))
    assert [(g.target.hypothesis.origin, g.target.hypothesis.claim[:15]) for g in a.groups] == [("articulated", "Elm Street home")]


def test_new_claim_restating_a_held_one_is_that_one(mission, op):
    over = {"episode.assess": {"new": [{"claim": "homes on Elm Street are a target of hostile preparation.", "fit": 0.5, "reach": 0.3}]}}
    t = make_core(over).assess(mission, op, TrajectoryInput(episode=ep(HOUSE), risk="attack")).groups[0].target
    assert t.hypothesis.origin == "predefined" and t.hypothesis.id == "H_elm" and not t.lambda_hypothesis


def test_rejected_claim_never_rearticulated(mission, op):
    over = {"episode.assess": {"new": [{"claim": "The Place of Worship is a target of hostile preparation", "reach": 0.3, "fit": 0.6}]},
            "episode.triage": {"relevant": True, "confidence": 0.9, "radar": 0, "content": {"summary": "night photography of entrances"}}}
    core = make_core(over)
    ti = TrajectoryInput(episode=ep("Someone photographed the Place of Worship entrances at night."), risk="attack")
    assert core.assess(mission, op, ti).groups == ()                          # scope given: rejected hypothesis not reported
    opened = core.assess(mission, op, ti, scope="open")                            # scope open: reported, marked rejected
    g = next(g for g in opened.groups if g.target.hypothesis.id == "H_worship")
    assert g.target.hypothesis.status == "rejected" and g.target.hypothesis.origin == "predefined"


def test_rejected_hypothesis_given_is_reported_with_status(core, mission, op):
    g = core.assess(mission, op, TrajectoryInput(episode=ep(HOUSE), hypothesis="H_worship")).groups[0]
    assert g.target.hypothesis.status == "rejected" and g.target.hypothesis.origin == "given"


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
    assert {"episode.assess", "standing.judge"} <= set(p.prompts) and p.calls
    assert len([c for c in p.calls if c.task == "standing.judge"]) == 1          # one standing call for all touched hypotheses
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


def test_radars_come_from_the_risk_not_from_hypotheses(mission, op):
    """Spec §5.2c: radars shaped by the held hypotheses would only look where the analyst already looks."""
    seen = []
    core = make_core({"radar.derive": lambda d: (seen.append(d), {"radars": [{"query": "someone watches a community building"}]})[1]})
    core.assess(mission, op, TrajectoryInput(episode=ep(HOUSE)))
    n = len(seen)
    core.assess(mission, op.model_copy(update={"hypotheses": op.hypotheses[:1]}), TrajectoryInput(episode=ep(HOODED)))
    assert len(seen) == n                                     # other hypotheses: same radars, not re-derived
    assert all("hypotheses" not in d for d in seen)           # and the hypotheses never reach the derivation


# ---------------------------------------------------------------- counter-evidence (stance)
WEAKENS = {"episode.assess": {"relevant": True, "new": [],
                              "stored": [{"hypothesis_id": "H_elm", "stance": "weakens", "fit": 0.1, "rationale": "the visitor was the new tenant"},
                                         {"hypothesis_id": "H_centre", "stance": "unrelated"}]}}


def _judge(fn):
    """A standing.judge override from a per-hypothesis function (hypothesis dict -> (before, value))."""
    return lambda d: {"standings": [dict(zip(("before", "value"), fn(h)), key=h["key"], rationale="x") for h in d["hypotheses"]]}


def test_counter_evidence_attaches_as_weakens(mission, op):
    a = make_core(WEAKENS).assess(mission, op, TrajectoryInput(episode=ep(HOUSE), risk="attack"))
    assert len(a.groups) == 1
    g = a.groups[0]
    t = g.target
    assert t.hypothesis.id == "H_elm" and t.hypothesis.origin == "predefined" and t.stance == "weakens" and t.score == 0.0
    assert "tenant" in t.rationale and g.baselines == ()


@pytest.mark.parametrize("strategy", ["judged", "noisy_or"])
def test_counter_evidence_lowers_standing(mission, op, strategy):
    ctx = op.model_copy(update={"trajectories": (traj("c1", "H_elm", 0.6, "confirmed"), traj("c2", "H_elm", 0.5, "confirmed"))})
    s = make_core(WEAKENS, standing=strategy).assess(mission, ctx, TrajectoryInput(episode=ep(HOUSE, eid="new"), risk="attack")).groups[0].standing
    assert s.value < s.without and s.delta < 0            # both strategies: counter-evidence lowers standing
    assert [u.status for u in s.used][-1] == "this"


def test_judge_that_ignores_counter_evidence_is_damped(mission, op):
    ctx = op.model_copy(update={"trajectories": (traj("c1", "H_elm", 0.6, "confirmed"),)})
    over = {**WEAKENS, "standing.judge": _judge(lambda h: (0.6, 0.6))}   # the judge does not lower it
    s = make_core(over).assess(mission, ctx, TrajectoryInput(episode=ep(HOUSE, eid="new"), risk="attack")).groups[0].standing
    assert s.without == 0.6 and s.value == 0.45            # damped by half its weight (proposed: 0.5)


def test_supporting_episode_never_lowers_standing(mission, op):
    ctx = op.model_copy(update={"trajectories": (traj("c1", "H_elm", 0.6, "confirmed"),)})
    over = {"episode.assess": {"stored": [{"hypothesis_id": "H_elm", "stance": "supports", "confidence": 0.9, "fit": 0.6, "reach": 0.3}]},
            "standing.judge": _judge(lambda h: (0.7, 0.3))}
    s = make_core(over).assess(mission, ctx, TrajectoryInput(episode=ep(HOODED, eid="new"), risk="attack")).groups[0].standing
    assert s.without == 0.7 and s.value == 0.7 and s.delta == 0.0


def test_before_is_zero_without_weighted_evidence(mission, op):
    over = {**WEAKENS, "standing.judge": _judge(lambda h: (0.4, 0.2))}
    s = make_core(over).assess(mission, op, TrajectoryInput(episode=ep(HOUSE, eid="new"), risk="attack")).groups[0].standing
    assert s.without == 0.0 and s.value == 0.0             # nothing attached: nothing to lower


def test_one_standing_call_judges_all_touched_hypotheses_together(mission, op):
    seen = []
    over = {"episode.assess": {"stored": [{"hypothesis_id": "H_centre", "stance": "supports", "confidence": 0.9, "fit": 0.6, "reach": 0.3},
                                          {"hypothesis_id": "H_elm", "stance": "weakens", "fit": 0.1}]},
            "standing.judge": lambda d: (seen.append(d), _judge(lambda h: (0.2, 0.3 if h["this"]["stance"] == "supports" else 0.1))(d))[1]}
    ctx = op.model_copy(update={"trajectories": (traj("c1", "H_elm", 0.6, "confirmed"),)})
    a = make_core(over).assess(mission, ctx, TrajectoryInput(episode=ep(HOODED, eid="new"), risk="attack"))
    assert len(seen) == 1 and [h["claim"][:7] for h in seen[0]["hypotheses"]] == ["The Com", "Homes o"]
    assert [(g.target.hypothesis.id, g.target.stance, g.standing.without, g.standing.value) for g in a.groups] == \
        [("H_centre", "supports", 0.0, 0.3), ("H_elm", "weakens", 0.2, 0.1)]


def test_held_hypotheses_carry_their_evidence_into_the_assessment(mission, op):
    seen = []
    over = {"episode.assess": lambda d: (seen.append(d), {"relevant": False})[1]}
    ctx = op.model_copy(update={"trajectories": (traj("c1", "H_elm", 0.6, "confirmed"),)})
    make_core(over).assess(mission, ctx, TrajectoryInput(episode=ep(HOUSE, eid="new"), risk="attack"))
    elm = next(h for h in seen[0]["held_hypotheses"] if h["id"] == "H_elm")
    assert elm["evidence"] == [""] or len(elm["evidence"]) == 1
    assert elm["baselines"] == [{"id": "B_listing", "claim": "the person has a property interest"}]


# ---------------------------------------------------------------- v3.2: anchored standing, refinement
def test_kept_standing_anchors_before(mission, op):
    seen = []
    hyps = tuple(h.model_copy(update={"standing": 0.6}) if h.id == "H_elm" else h for h in op.hypotheses)
    ctx = op.model_copy(update={"hypotheses": hyps, "trajectories": (traj("c1", "H_elm", 0.6, "confirmed"),)})
    over = {"episode.assess": {"stored": [{"hypothesis_id": "H_elm", "stance": "supports", "confidence": 0.9, "fit": 0.6, "reach": 0.3}]},
            "standing.judge": lambda d: (seen.append(d), _judge(lambda h: (0.3, 0.7))(d))[1]}
    s = make_core(over).assess(mission, ctx, TrajectoryInput(episode=ep(HOODED, eid="new"), risk="attack")).groups[0].standing
    assert seen[0]["hypotheses"][0]["before"] == 0.6                 # the judge is told
    assert s.without == 0.6 and s.value == 0.7 and s.delta == 0.1     # and its own "before" is overridden


def test_kept_standing_ignored_without_weighted_evidence(mission, op):
    hyps = tuple(h.model_copy(update={"standing": 0.6}) if h.id == "H_elm" else h for h in op.hypotheses)
    over = {**WEAKENS, "standing.judge": _judge(lambda h: (0.0, 0.0))}
    s = make_core(over).assess(mission, op.model_copy(update={"hypotheses": hyps}),
                               TrajectoryInput(episode=ep(HOUSE, eid="new"), risk="attack")).groups[0].standing
    assert s.without == 0.0


def test_refinement_is_a_new_hypothesis_linked_to_the_held_one(mission, op):
    over = {"episode.assess": {"stored": [{"hypothesis_id": "H_centre", "stance": "supports", "confidence": 0.9, "fit": 0.5, "reach": 0.3}],
                               "new": [{"claim": "The Community Centre's side doors are being mapped for a night entry", "refines": "H_centre",
                                        "fit": 0.6, "reach": 0.5},
                                       {"claim": "Something else entirely", "refines": "nope", "fit": 0.4, "reach": 0.2}]}}
    a = make_core(over).assess(mission, op, TrajectoryInput(episode=ep(HOODED), risk="attack"))
    held = next(g.target.hypothesis for g in a.groups if g.target.hypothesis.origin == "predefined")
    assert held.id == "H_centre" and held.refines is None
    new = [g.target.hypothesis for g in a.groups if g.target.hypothesis.origin == "articulated"]
    assert [(h.refines) for h in new] == ["H_centre", None]           # an unknown id is dropped, not invented


def test_revised_claim_becomes_a_refinement(mission, op):
    over = {"episode.assess": {"stored": [{"hypothesis_id": "H_centre", "stance": "supports", "confidence": 0.9, "fit": 0.5, "reach": 0.3,
                                           "revised_claim": "The Community Centre's side doors are being mapped for a night entry"}]}}
    a = make_core(over).assess(mission, op, TrajectoryInput(episode=ep(HOODED), risk="attack"))
    hs = [(g.target.hypothesis.origin, g.target.hypothesis.id, g.target.hypothesis.refines) for g in a.groups]
    assert ("predefined", "H_centre", None) in hs and ("articulated", None, "H_centre") in hs


def test_unconfirmed_standing_is_capped(mission, op):
    over = {"episode.assess": {"stored": [{"hypothesis_id": "H_elm", "stance": "supports", "confidence": 0.9, "fit": 0.6, "reach": 0.3}]},
            "standing.judge": _judge(lambda h: (0.85, 0.95))}
    proposed = op.model_copy(update={"trajectories": (traj("p1", "H_elm", 0.6, "proposed"),)})
    s = make_core(over).assess(mission, proposed, TrajectoryInput(episode=ep(HOODED, eid="new"), risk="attack")).groups[0].standing
    assert s.without == 0.8 and s.value == 0.8
    confirmed = op.model_copy(update={"trajectories": (traj("c1", "H_elm", 0.6, "confirmed"),)})
    s = make_core(over).assess(mission, confirmed, TrajectoryInput(episode=ep(HOODED, eid="new"), risk="attack")).groups[0].standing
    assert s.value == 0.95


@pytest.mark.parametrize("strategy", ["judged", "noisy_or"])
def test_given_hypothesis_episode_does_not_bear_on_is_neutral(mission, op, strategy):
    ctx = op.model_copy(update={"trajectories": (traj("c1", "H_elm", 0.6, "confirmed"),)})
    over = {"episode.assess": {"given": {"stance": "unrelated", "fit": 0.05, "rationale": "a routine detail"}},
            "standing.judge": _judge(lambda h: (0.6, 0.3))}
    g = make_core(over, standing=strategy).assess(mission, ctx, TrajectoryInput(episode=ep("The council repainted the bus shelter on Main Road.", eid="new"), hypothesis="H_elm")).groups[0]
    assert g.target.stance == "neutral" and g.target.score == 0.0 and g.baselines == ()
    assert g.standing.value == g.standing.without and g.standing.delta == 0.0


def test_neutral_facts_stay_in_view_for_conjunctions(mission, op):
    """A fact neutral alone (a recorded payload) must reach both the assessment and the standing judgment of a later
    episode (the stated limit) that completes the conjunction."""
    from mistabra_core.types import SignalContent
    payload = traj("n1", "H_elm", 0.0).model_copy(update={"stance": "neutral", "signal": SignalContent(summary="payload was 620 kg")})
    ctx = op.model_copy(update={"trajectories": (payload,)})
    seen = {}
    over = {"episode.assess": lambda d: (seen.setdefault("assess", d), {"given": {"stance": "supports", "fit": 0.3, "reach": 0.3}})[1],
            "standing.judge": lambda d: (seen.setdefault("judge", d), _judge(lambda h: (0.0, 0.5))(d))[1]}
    make_core(over).assess(mission, ctx, TrajectoryInput(episode=ep("The rated capacity is 500 kg.", eid="new"), hypothesis="H_elm"))
    assert seen["assess"]["given_hypothesis"]["evidence"] == ["(neutral) payload was 620 kg"]
    assert [t["stance"] for t in seen["judge"]["hypotheses"][0]["trajectories"]] == ["neutral"]


def test_default_standing_is_noisy_or_and_makes_no_standing_call(mission, op):
    from mistabra_core import Core, Settings
    from mistabra_core.llm.fake import FakeProvider
    core = Core(Settings(force_fake=True), provider=FakeProvider())
    assert core.settings.standing == "noisy_or"
    a = core.assess(mission, op, TrajectoryInput(episode=ep(HOUSE), risk="attack"))
    assert a.groups and "standing.judge" not in {c.task for c in a.provenance.calls}


# ---------------------------------------------------------------- sharpening
SUPPORTS_ELM = {"episode.assess": {"stored": [{"hypothesis_id": "H_elm", "stance": "supports", "confidence": 0.9, "fit": 0.6, "reach": 0.3}]}}


def _with_facts(op, n, extra_hyps=()):
    return op.model_copy(update={"hypotheses": op.hypotheses + tuple(extra_hyps),
                                 "trajectories": tuple(traj(f"p{i}", "H_elm", 0.4, "proposed") for i in range(n))})


def test_held_hypothesis_with_enough_facts_is_sharpened_and_the_refinement_inherits_them(mission, op):
    seen = []
    over = {**SUPPORTS_ELM, "hypothesis.sharpen": lambda d: (seen.append(d), {"sharper_claim": "A named group is mapping Elm Street homes "
                                                                                                 "from a parked van at night", "why": "x"})[1]}
    a = make_core(over, standing="noisy_or").assess(mission, _with_facts(op, 2), TrajectoryInput(episode=ep(HOODED, eid="new"), risk="attack"))
    assert len(seen) == 1 and len(seen[0]["facts"]) == 3 and seen[0]["facts"][-1]["new"]
    ref = next(g for g in a.groups if g.target.hypothesis.refines == "H_elm")
    held = next(g for g in a.groups if g.target.hypothesis.id == "H_elm")
    assert ref.target.hypothesis.origin == "articulated" and ref.target.lambda_hypothesis
    assert ref.standing.without == held.standing.without > 0          # same inherited evidence


def test_too_few_facts_or_an_existing_refinement_means_no_sharpening(mission, op):
    from mistabra_core.types import Hypothesis
    calls = []
    over = {**SUPPORTS_ELM, "hypothesis.sharpen": lambda d: (calls.append(d), {"sharper_claim": "x y z"})[1]}
    make_core(over).assess(mission, _with_facts(op, 1), TrajectoryInput(episode=ep(HOODED, eid="new"), risk="attack"))
    child = Hypothesis(id="H_elm2", risk_id="attack", claim="A sharper Elm Street claim", author="system", refines="H_elm")
    make_core(over).assess(mission, _with_facts(op, 3, [child]), TrajectoryInput(episode=ep(HOODED, eid="new"), risk="attack"))
    assert calls == []


def test_a_stored_refinement_counts_its_parents_evidence(mission, op):
    from mistabra_core.types import Hypothesis
    child = Hypothesis(id="H_elm2", risk_id="attack", claim="A sharper Elm Street claim", author="system", status="proposed", refines="H_elm")
    over = {"episode.assess": {"stored": [{"hypothesis_id": "H_elm2", "stance": "supports", "confidence": 0.9, "fit": 0.6, "reach": 0.3}]}}
    g = make_core(over, standing="noisy_or").assess(mission, _with_facts(op, 2, [child]),
                                                    TrajectoryInput(episode=ep(HOODED, eid="new"), risk="attack")).groups[0]
    assert g.target.hypothesis.id == "H_elm2" and g.standing.without > 0 and len(g.standing.used) == 3
