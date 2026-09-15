import pytest
from mistabra.entities import SimpleResolver, normalize_name, canonical_name, name_similarity

C = [{"id": "e1", "entity_type": "person", "canonical_name": "Jens Hofer", "aliases": []},
     {"id": "e2", "entity_type": "person", "canonical_name": "Maria Klein", "aliases": ["M. Klein"]},
     {"id": "e3", "entity_type": "organization", "canonical_name": "Northfield Gazette", "aliases": []}]


def test_normalization():
    assert normalize_name("The Community Centre") == "community centre" and normalize_name("Jens Hofer's") == "jens hofer"
    assert canonical_name("Jens Hofer's") == "Jens Hofer" and canonical_name("the Community Centre") == "Community Centre"


def test_similarity_subset_rules():
    assert name_similarity("hofer", "jens hofer") == 95          # surname
    assert name_similarity("northfield", "northfield gazette") == 80  # head token only -> ambiguous
    assert name_similarity("jens hofer", "jens hofer") == 100


def test_exact_alias_and_surname_link_without_adjudicator():
    r = SimpleResolver(gateway=None)
    assert r.resolve({"surface": "M. Klein", "entity_type": "person"}, C).outcome == "link"
    d = r.resolve({"surface": "Hofer", "entity_type": "person"}, C)
    assert d.outcome == "link" and d.entity_id == "e1"


def test_ambiguous_without_adjudicator_is_unresolved():
    d = SimpleResolver(gateway=None).resolve({"surface": "Northfield", "entity_type": "organization"}, C)
    assert d.outcome == "unresolved" and "no adjudicator" in d.rationale


def test_unrelated_is_new():
    d = SimpleResolver(gateway=None).resolve({"surface": "Thomas Berg", "entity_type": "person"}, C)
    assert d.outcome == "new"


def test_no_candidates_is_new():
    assert SimpleResolver().resolve({"surface": "Anyone", "entity_type": "vehicle"}, []).outcome == "new"


def test_adjudication_path_uses_operational_model(m):
    r = SimpleResolver(gateway=m.gateway)
    d = r.resolve({"surface": "Northfield", "entity_type": "organization"}, C)
    assert d.outcome == "unresolved" and d.rationale.startswith("adjudicated") and m.gateway.provider.calls == ["entities.adjudicate"]


def test_splink_backend_optional():
    pytest.importorskip("splink")
    from mistabra.entities import SplinkResolver
    r = SplinkResolver()
    assert r.resolve({"surface": "Jens Hofer", "entity_type": "person"}, C).outcome == "link"
    assert r.resolve({"surface": "Someone Else", "entity_type": "person"}, C).outcome == "new"
