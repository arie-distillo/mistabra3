import json, pytest
from tests.unit.conftest import DOC


def test_publish_only_accepted_and_drops_dangling(m):
    did = m.ontology.build_draft("default", [DOC], actor="de")
    m.ontology.accept_all(did, "de")
    m.ontology.review(did, "signal_types", "sig.rehearsal", "reject", "de")
    vid = m.ontology.publish(did, "de")
    o = m.ontology.load(vid)
    assert o.version == 1 and "sig.rehearsal" not in {s.id for s in o.signal_types}
    cap = next(s for s in o.signal_types if s.id == "sig.capability_building")
    assert "sig.rehearsal" not in cap.precursor_to
    assert all(o.status == "confirmed" for o in o.objectives)
    assert (m.settings.data_dir / "ontology" / "default" / "v1.json").exists()


def test_edit_and_merge(m):
    did = m.ontology.build_draft("default", [DOC], actor="de")
    m.ontology.review(did, "concepts", "con.community_centre", "edit", "de", patch={"label": "CX Centre"})
    m.ontology.review(did, "concepts", "con.community_institution", "merge", "de", into="con.community_centre")
    els = {e["element_id"]: e for e in m.ontology.draft_elements(did) if e["kind"] == "concepts"}
    assert els["con.community_centre"]["body"]["label"] == "CX Centre" and els["con.community_centre"]["status"] == "accepted"
    assert els["con.community_institution"]["status"] == "merged" and "Community institution" in els["con.community_centre"]["body"]["aliases"]
    acts = [a["action"] for a in m.audit.events(object_type="ontology_draft_element")]
    assert "ontology.element_edit" in acts and "ontology.element_merge" in acts


def test_unaccepted_pending_elements_are_excluded(m):
    did = m.ontology.build_draft("default", [DOC], actor="de")
    m.ontology.accept_all(did, "de", kinds=("objectives", "entity_types", "sources", "concepts"))
    vid = m.ontology.publish(did, "de")
    o = m.ontology.load(vid)
    assert o.signal_types == [] and o.observables == [] and len(o.objectives) == 2


def test_revision_carries_previous_elements_and_increments_version(m):
    did = m.ontology.build_draft("default", [DOC], actor="de"); m.ontology.accept_all(did, "de"); v1 = m.ontology.publish(did, "de")
    did2 = m.ontology.build_draft("default", [("more.md", "Risk R3: Vandalism of community institutions.")], actor="de")
    els = m.ontology.draft_elements(did2)
    assert any(e["origin"] == "existing" and e["status"] == "accepted" for e in els)
    m.ontology.accept_all(did2, "de"); v2 = m.ontology.publish(did2, "de")
    o1, o2 = m.ontology.load(v1), m.ontology.load(v2)
    assert (o1.version, o2.version) == (1, 2) and len(o2.objectives) == 3 and len(o1.objectives) == 2
    assert [v["version"] for v in m.ontology.versions("default")] == [1, 2]


def test_import_taxonomy(m):
    did = m.ontology.build_draft("default", [DOC], actor="de")
    n = m.ontology.import_taxonomy(did, {"concepts": [{"label": "Reconnaissance", "aliases": ["recce"]}]}, "de")
    e = next(x for x in m.ontology.draft_elements(did) if x["element_id"] == "con.reconnaissance")
    assert n == 1 and e["origin"] == "imported" and e["body"]["status"] == "imported"


def test_export_is_valid_json(m, published):
    d = json.loads(m.ontology.export(published))
    assert d["ontology_id"] == "default" and d["version"] == 1 and "functions" not in d
