"""Episodes drive everything; the analyst intervenes through reads and updates."""
import pytest
from tests.unit.conftest import HOUSE, HOODED, AGENT, FLYTHROUGH, GROCERIES


def test_episode_creates_signal_trajectories_and_hypothesis(m, ready):
    ep = m.episodes.create(HOUSE, "host", source="Field reports", occurred="2026-08-03T10:15:00Z")
    sigs = m.signals.list(episode_id=ep.id)
    assert len(sigs) == 1 and sigs[0].via == "radar" and sigs[0].radar_id and sigs[0].status == "reasoned"
    assert sigs[0].content.target == "Elm Street" and "Elm Street" in sigs[0].content.entities
    trs = m.trajectories.list(signal_id=sigs[0].id)
    assert sum(t.outcome == "risk" for t in trs) == 1 and sum(t.outcome == "ordinary" for t in trs) >= 1
    assert all(t.status == "proposed" for t in trs)
    assert len(m.hypotheses.list(risk_id=ready.id)) == 1


def test_idempotent_episode(m, ready):
    a = m.episodes.create(HOUSE, "host", source="Field reports"); b = m.episodes.create(" " + HOUSE + " ", "host", source="Field reports")
    assert a.id == b.id and m.stats()["signals"] == 1


def test_occurred_forms_and_missing(m, ready):
    e = m.episodes.create("A concealed observer with a camera watched Maria Klein's home on Elm Street.", "host", source="Field reports")
    assert e.occurred_kind == "unknown"
    s = m.signals.list(episode_id=e.id)[0]
    assert "occurrence time" in s.missing
    r = m.episodes.create(HOODED, "host", source="Field reports", occurred=("2026-08-06", "2026-08-08"))
    assert r.occurred_kind == "range" and r.occurred_end == "2026-08-08"


def test_no_signal_for_unrelated_episode(m, ready):
    ep = m.episodes.create(GROCERIES, "host", source="Field reports")
    assert m.signals.list(episode_id=ep.id) == []


def test_lambda_signal_on_the_right_risk(m, ready):
    """No radar anticipated the episode; every risk is checked and only the one it could serve gets a Lambda signal."""
    other = m.risks.create("Burglars target homes in Riverside.", "a")
    ep = m.episodes.create(FLYTHROUGH, "host", source="Field reports")
    sigs = m.signals.list(episode_id=ep.id)
    assert [s.via for s in sigs] == ["lambda"] and sigs[0].risk_id == ready.id and sigs[0].note
    assert m.signals.list(risk_id=other.id) == []
    assert {c["task"] for c in m.audit.trace(sigs[0].id)["llm_calls"]} >= {"signal.lambda_check"}


def test_strength_ordering_and_hard_negative(m, ready):
    ids = [m.episodes.create(t, "host", source="Field reports").id for t in (AGENT, HOUSE, HOODED)]
    st = [m.signals._build(m.store.one("SELECT * FROM signal WHERE episode_id=?", (i,)), unblind=True).strength_system for i in ids]
    assert st[0] < st[1] < st[2]


def test_blind_rule(m, ready):
    ep = m.episodes.create(HOUSE, "host", source="Field reports")
    s = m.signals.list(episode_id=ep.id)[0]
    assert s.strength_system is None and s.metrics is None and s.trajectory is None and s.baselines == []
    s = m.signals.update(s.id, "analyst", strength=0.3)
    assert s.strength_analyst == 0.3 and s.strength_system is not None and s.trajectory is not None and s.baselines
    ev = m.audit.events(object_type="signal", object_id=s.id)[0]
    assert ev["action"] == "signal.updated" and ev["after"]["system_was"] == s.strength_system and ev["blind"] == 1


def test_blind_off_setting(tmp_path):
    from mistabra import Mistabra, Settings
    from mistabra.llm import FakeProvider
    m = Mistabra(tmp_path, Settings(data_dir=tmp_path, blind=False), provider=FakeProvider())
    r = m.risks.create("Hateful radicals attack members of Community X.", "a"); m.radars.derive(r.id, "a"); m.sources.create("Field reports", "a")
    ep = m.episodes.create(HOUSE, "host", source="Field reports")
    assert m.signals.list(episode_id=ep.id)[0].strength_system is not None
    m.close()


def test_analyst_raises_hunch(m, ready):
    ep = m.episodes.create(GROCERIES, "host", source="Field reports")
    s = m.signals.create(ep.id, ready.id, "analyst", note="the shopper is the man from Elm Street")
    assert s.via == "analyst" and s.origin == "analyst" and m.trajectories.list(signal_id=s.id)


def test_confirm_trajectory_makes_precedent_and_support(m, ready):
    ep = m.episodes.create(HOODED, "host", source="Field reports", occurred="2026-08-09T22:30:00Z")
    s = m.signals.update(m.signals.list(episode_id=ep.id)[0].id, "analyst", strength=0.8)
    t = m.trajectories.update(s.trajectory.id, "analyst", status="confirmed")
    assert t.status == "confirmed" and m.signals.get(s.id).status == "confirmed"
    assert m.context.list("precedent") and m.hypotheses.get(t.hypothesis_id).status == "supported"
    # precedent is retrieved for the next signal of the same risk
    ep2 = m.episodes.create(HOUSE, "host", source="Field reports")
    call = m.store.one("SELECT input_json FROM llm_call WHERE task='trajectory.risk' AND subject_id=?", (m.signals.list(episode_id=ep2.id)[0].id,))
    assert "precedents" in call["input_json"] and "Confirmed" in call["input_json"] or '"precedents": [{' in call["input_json"]


def test_hypothesis_reuse_and_coverage(m, ready):
    m.episodes.create(HOUSE, "host", source="Field reports", occurred="2026-08-03T10:15:00Z")
    m.episodes.create("The same man photographed Jens Hofer's house on Elm Street again, on a second day.", "host", source="Field reports", occurred="2026-08-05T09:40:00Z")
    hs = m.hypotheses.list(risk_id=ready.id)
    assert len(hs) == 1 and len(m.hypotheses.signals(hs[0].id)) == 2
    cov = m.hypotheses.coverage(ready.id)
    assert cov.items[0].support == 2 and cov.items[0].first_seen == "2026-08-03T10:15:00Z"


def test_dismiss_and_list_filters(m, ready):
    ep = m.episodes.create(HOUSE, "host", source="Field reports")
    s = m.signals.list(episode_id=ep.id)[0]
    m.signals.remove(s.id, "analyst", reason="known neighbour")
    assert m.signals.list(status="dismissed")[0].id == s.id and m.signals.list(status="reasoned") == []


def test_radar_update_disables(m, ready):
    for r in m.radars.list(ready.id):
        m.radars.update(r.id, "a", enabled=False)
    ep = m.episodes.create(HOUSE, "host", source="Field reports")
    assert m.signals.list(episode_id=ep.id) == []
    assert any(e["action"] == "radar.updated" for e in m.audit.events(object_type="radar"))


def test_trace_and_replay(m, ready):
    ep = m.episodes.create(HOUSE, "host", source="Field reports")
    s = m.signals.list(episode_id=ep.id)[0]
    tr = m.audit.trace(s.id)
    assert {c["task"] for c in tr["llm_calls"]} >= {"radar.read", "trajectory.risk", "trajectory.baseline"}
    assert any(e["action"] == "signal.reasoned" for e in tr["events"]) and m.audit.verify()
    rep = m.learn.replay([s.id], "analyst")
    assert rep["mean_abs_delta"] == 0.0 and m.signals._build(m.store.one("SELECT * FROM signal WHERE id=?", (s.id,)), unblind=True).strength_system == rep["rows"][0]["stored"]
    with pytest.raises(ValueError):
        m.learn.replay([s.id], "analyst", prompt_version="99")


def test_reopen_data_dir(tmp_path, m, ready):
    m.episodes.create(HOUSE, "host", source="Field reports"); n = m.stats()["signals"]; m.close()
    from mistabra import Mistabra, Settings
    from mistabra.llm import FakeProvider
    m2 = Mistabra(tmp_path, Settings(data_dir=tmp_path), provider=FakeProvider())
    assert m2.stats()["signals"] == n and m2.risks.list()[0].id == ready.id
    m2.close()
