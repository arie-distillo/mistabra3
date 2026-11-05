"""The decision model behind the interface: pairwise reproduces the PoC; setwise scores the set with state."""
import pytest
from mistabra import Mistabra, Settings
from mistabra.llm import FakeProvider
from mistabra.decision import PairwiseDecision, SetwiseDecision, HypothesisSet, HypothesisRef, State, Posterior, Abduction, kl_bits, normalise, make_decision
from tests.unit.conftest import HOUSE, HOODED, FLYTHROUGH, AGENT


def _m(tmp_path, **kw):
    m = Mistabra(tmp_path, Settings(data_dir=tmp_path, blind=False, **kw), provider=FakeProvider())
    m.context.create("Community X in Northfield; institutions: Community Centre on Maple Road.", "a", title="about")
    r = m.risks.create("Hateful radicals attack members of Community X.", "a"); m.radars.derive(r.id, "a"); m.sources.create("Field reports", "a", credibility=0.8)
    return m, r


def test_arithmetic():
    d = normalise({"a": 2, "b": 1, "none": 1}); assert abs(sum(d.values()) - 1) < 1e-3 and d["a"] == 0.5
    assert kl_bits({"a": 0.5, "b": 0.5}, None) == 0.0 and kl_bits({"a": 0.99, "b": 0.01}, None) > 0.5
    with pytest.raises(ValueError): make_decision("nope", None, None)


def test_pairwise_posterior_shape_and_strength_formula(tmp_path):
    hs = HypothesisSet("r", (HypothesisRef("new", "X is a target", "risk"),), (HypothesisRef("baseline:0", "property listing", "baseline"),))
    ab = Abduction("new", "X is a target", "why", 0.5, 0.7, (), (), (), ({"claim": "property listing", "ordinary_outcome": "listing", "rationale": "", "fit": 0.4},), 1.0, 0.8)
    sc = PairwiseDecision(None, Settings(data_dir=tmp_path)).score([], State({}, None), {}, hs, ab)
    assert set(sc.posterior.distribution) == {"new", "baseline:0", "none"} and sc.posterior.distribution["new"] > sc.posterior.distribution["baseline:0"]
    from mistabra.decision import strength_from
    assert sc.metrics["gap"] == 0.3 and sc.strength == strength_from(0.5, 0.7, 0.3, 1.0, 0.8) == 0.47 and not sc.unexplained


def test_pairwise_default_writes_posterior_and_keeps_strengths(tmp_path):
    m, r = _m(tmp_path)
    ep = m.episodes.create(HOUSE, "host", source="Field reports")
    s = m.signals.list(episode_id=ep.id)[0]
    assert s.posterior is not None and s.posterior.decision == "pairwise" and abs(sum(s.posterior.distribution.values()) - 1) < 1e-2
    assert set(s.posterior.distribution) >= {t.hypothesis_id for t in m.trajectories.list(signal_id=s.id)} | {"none"}
    assert m.stats()["posteriors"] == 1 and m.stats()["decision"] == "pairwise"
    assert any(e["action"] == "posterior.written" for e in m.audit.events(object_type="posterior"))
    cov = m.hypotheses.coverage(r.id)
    assert cov.items and cov.items[0].belief is not None
    m.close()


def test_blind_rule_hides_posterior(tmp_path):
    m = Mistabra(tmp_path, Settings(data_dir=tmp_path), provider=FakeProvider())
    r = m.risks.create("Hateful radicals attack members of Community X.", "a"); m.radars.derive(r.id, "a"); m.sources.create("Field reports", "a")
    s = m.signals.list(episode_id=m.episodes.create(HOUSE, "host", source="Field reports").id)[0]
    assert s.posterior is None
    assert m.signals.update(s.id, "analyst", strength=0.4).posterior is not None
    m.close()


def test_setwise_scores_the_set_with_state(tmp_path):
    m, r = _m(tmp_path, decision="setwise")
    e1 = m.episodes.create(HOUSE, "host", source="Field reports", occurred="2026-08-03T10:00:00Z")
    s1 = m.signals.list(episode_id=e1.id)[0]
    assert s1.posterior.decision == "setwise" and s1.strength_system == round(sum(v for k, v in s1.posterior.distribution.items() if k == s1.trajectory.hypothesis_id), 3)
    assert {c["task"] for c in m.audit.trace(s1.id)["llm_calls"]} >= {"trajectory.risk", "trajectory.baseline", "decision.setwise"}
    e2 = m.episodes.create("The same man photographed Jens Hofer's house on Elm Street again, on a second day.", "host", source="Field reports", occurred="2026-08-05T09:40:00Z")
    s2 = m.signals.list(episode_id=e2.id)[0]
    # state: the second signal sees the first one attached to the same hypothesis, and the prior is the first posterior
    assert s2.trajectory.hypothesis_id == s1.trajectory.hypothesis_id
    assert s2.posterior.distribution[s2.trajectory.hypothesis_id] > s1.posterior.distribution[s1.trajectory.hypothesis_id]
    call = m.store.one("SELECT input_json FROM llm_call WHERE task='decision.setwise' AND subject_id=?", (s2.id,))["input_json"]
    assert '"attached": [{' in call and '"prior": {' in call
    assert s2.posterior.info_gain >= 0
    m.close()


def test_setwise_none_creates_lambda_hypothesis_and_rescores(tmp_path):
    m, r = _m(tmp_path, decision="setwise", none_threshold=0.35)
    m.episodes.create("A passer-by photographed the Community Centre on Maple Road.", "host", source="Field reports")   # existing hypothesis about the Centre
    before = {h.id for h in m.hypotheses.list(risk_id=r.id)}
    assert len(before) == 1
    ep = m.episodes.create(FLYTHROUGH, "host", source="Field reports")        # the abducer reuses that hypothesis; the fake scorer puts mass on none
    s = m.signals.list(episode_id=ep.id)[0]
    new = [h for h in m.hypotheses.list(risk_id=r.id) if h.id not in before]
    assert len(new) == 1 and new[0].is_lambda and s.trajectory.hypothesis_id == new[0].id
    ev = [e for e in m.audit.events(object_type="hypothesis", object_id=new[0].id) if e["action"] == "hypothesis.created"][0]
    assert ev["after"]["reason"] == "none_of_these"
    assert [e["after"]["rescored"] for e in m.audit.events(object_type="posterior") if e["after"]["signal_id"] == s.id] == [True]
    assert m.store.one("SELECT COUNT(*) AS n FROM llm_call WHERE task='decision.setwise' AND subject_id=?", (s.id,))["n"] == 2
    assert s.posterior.distribution.get("none", 0) < 0.35
    m.close()


def test_replay_compares_decision_models_on_the_same_signals(tmp_path):
    m, r = _m(tmp_path)
    ids = [m.signals.list(episode_id=m.episodes.create(t, "host", source="Field reports").id)[0].id for t in (AGENT, HOUSE, HOODED)]
    same = m.learn.replay(ids, "analyst")
    assert same["decision"] == "pairwise" and same["mean_abs_delta"] == 0.0
    other = m.learn.replay(ids, "analyst", decision="setwise")
    assert other["decision"] == "setwise" and all(set(row["posterior"]) >= {"none"} for row in other["rows"])
    assert m.stats()["posteriors"] == 3                                     # replay never persists
    sw = [row["replayed"] for row in other["rows"]]
    assert sw[0] < sw[2]                                                      # hard negative below the concealed observation under setwise too
    m.close()


def test_next_action_is_reserved(tmp_path):
    with pytest.raises(NotImplementedError):
        SetwiseDecision(None, Settings(data_dir=tmp_path)).next_action([], State({}, None), ())
