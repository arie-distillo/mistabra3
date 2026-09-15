from mistabra.ontology.builder import chunk_text
from tests.unit.conftest import DOC


def test_chunking_respects_paragraphs():
    text = "\n\n".join(f"para {i} " + "x" * 300 for i in range(10))
    chunks = chunk_text(text, 1000)
    assert 3 <= len(chunks) <= 5 and all(len(c) <= 1000 for c in chunks) and "".join(chunks).count("para") == 10


def test_draft_has_provenance_and_inferred_status(m):
    did = m.ontology.build_draft("default", [DOC], actor="de")
    els = m.ontology.draft_elements(did)
    kinds = {e["kind"] for e in els}
    assert {"objectives", "concepts", "entity_types", "sources", "signal_types", "observables"} <= kinds
    for e in els:
        if e["kind"] in ("objectives", "concepts", "entity_types", "sources"):
            assert e["body"]["provenance"] and e["body"]["provenance"][0]["doc"] == "risk_register.md" and e["origin"] == "extracted"
        else:
            assert e["body"]["status"] == "inferred" and e["origin"] == "inferred"
    obj = [e for e in els if e["kind"] == "objectives"]
    assert {o["body"]["polarity"] for o in obj} == {"negative"} and len(obj) == 2
    centre = next(e for e in els if e["element_id"] == "con.community_centre")
    assert centre["body"]["aliases"] == ["the centre"] and centre["body"]["parent_id"] == "con.community_institution"
    assert all(e["status"] == "pending" for e in els)
    assert any(a["action"] == "ontology.draft_built" for a in m.audit.events(object_type="ontology_draft"))


def test_signal_types_link_objectives_and_observables(m):
    did = m.ontology.build_draft("default", [DOC], actor="de")
    sts = {e["element_id"]: e["body"] for e in m.ontology.draft_elements(did) if e["kind"] == "signal_types"}
    assert "sig.reconnaissance" in sts and len(sts["sig.reconnaissance"]["objective_ids"]) == 2  # shared across both risks
    obs = {e["element_id"]: e["body"] for e in m.ontology.draft_elements(did) if e["kind"] == "observables"}
    assert obs["obs.image_capture_of_place"]["can_serve"][0]["signal_type_id"] == "sig.reconnaissance"
