"""The Casebook: the layer above the core that keeps a case between calls."""
import json

from mistabra_core import Casebook, Core, Episode, Settings
from mistabra_core.llm.fake import FakeProvider

from .conftest import HOUSE, ep


def _case(mission, op=None, path=None, **kw):
    return Casebook(Core(Settings(force_fake=True), provider=FakeProvider(kw.pop("overrides", {}))), mission, op, path=path, **kw)


def test_ingest_keeps_what_the_core_returned(mission):
    case = _case(mission)
    got = case.ingest(ep(HOUSE), risk="attack")
    assert got.assessment.groups and len(got.trajectories) == len(got.assessment.groups)
    assert {t.id for t in case.trajectories()} >= set(got.trajectories)
    assert all(case.hypothesis(h) is not None for h in got.hypothesis_ids) and got.created
    assert case.assessments[-1]["trajectories"] == got.trajectories


def test_the_case_is_the_next_calls_operational_context(mission):
    case = _case(mission)
    first = case.ingest(ep(HOUSE, eid="e1"), risk="attack")
    second = case.ingest(ep(HOUSE, eid="e2"), risk="attack")
    assert second.assessment.groups[0].standing.without > 0            # the first episode's trajectory was weighed
    assert first.hypothesis_ids[0] in {h.id for h in case.hypotheses("attack")}


def test_analyst_decisions_are_recorded_and_change_the_case(mission):
    case = _case(mission)
    hid = case.add_hypothesis("Someone is mapping homes on Elm Street", risk="attack")
    got = case.ingest(ep(HOUSE), risk="attack")
    t = got.trajectories[0]
    case.confirm(t, by="arie", note="seen on camera")
    assert next(x for x in case.trajectories() if x.id == t).status == "confirmed"
    assert case.hypothesis(got.hypothesis_ids[0]).standing is None          # re-judged from the evidence next time
    case.reject_hypothesis(hid, by="arie", note="no link")
    assert case.hypothesis(hid).status == "rejected" and case.hypothesis(hid).version == 2
    d = case.decisions
    assert [(x["kind"], x["status"], x["by"]) for x in d] == [("trajectory", "confirmed", "arie"), ("hypothesis", "rejected", "arie")]
    assert d[0]["assessment"] == case.assessments[-1]["id"] and d[0]["previous"] == "proposed"


def test_export_joins_each_decision_with_its_assessment(mission, tmp_path):
    case = _case(mission)
    got = case.ingest(ep(HOUSE), risk="attack")
    case.confirm(got.trajectories[0], by="arie")
    out = tmp_path / "feedback.jsonl"
    assert case.export(out) == 1
    row = json.loads(out.read_text(encoding="utf-8"))
    assert row["decision"]["status"] == "confirmed" and row["episode"]["text"] == HOUSE
    assert row["assessment"]["groups"] and row["trajectory"]["id"] == got.trajectories[0]


def test_a_casebook_on_disk_survives_a_restart(mission, tmp_path):
    db = tmp_path / "case.sqlite"
    case = _case(mission, path=db)
    hid = case.add_hypothesis("Someone is mapping homes on Elm Street", risk="attack")
    got = case.ingest(ep(HOUSE), risk="attack")
    case.confirm(got.trajectories[0], by="arie")
    again = _case(mission, path=db)
    assert again.hypothesis(hid).claim.startswith("Someone") and len(again.trajectories()) == len(case.trajectories())
    assert again.decisions == case.decisions and again.assessments == case.assessments
    assert again.add_hypothesis("another", risk="attack") != hid                  # counters were restored too


def test_look_back_can_be_turned_off_and_has_a_budget(mission):
    case = _case(mission, lookback=False)
    case.ingest(Episode(id="x1", text="The council repainted the bus shelter on Main Road."))
    got = case.ingest(ep(HOUSE), risk="attack")
    assert got.lookback == [] and case.unattached
    case = _case(mission, lookback_budget=0)
    case.ingest(Episode(id="x1", text="The council repainted the bus shelter on Main Road."))
    assert case.ingest(ep(HOUSE), risk="attack").lookback == []
