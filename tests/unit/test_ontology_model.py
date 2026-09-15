import pytest
from mistabra.ontology import Ontology, Objective, SignalType, Observable, Concept, EntityType
from mistabra.ontology.model import CanServe


def _onto():
    return Ontology(ontology_id="t", objectives=[Objective(id="obj.a", label="A", concerns_entity_types=["et.person"])],
                    entity_types=[EntityType(id="et.person", label="person")],
                    signal_types=[SignalType(id="sig.recon", label="recon", objective_ids=["obj.a"], proximity_to_harm=0.4, precursor_to=["sig.reh"]),
                                  SignalType(id="sig.reh", label="rehearsal", objective_ids=["obj.a"], proximity_to_harm=0.8)],
                    observables=[Observable(id="obs.photo", label="photo", can_serve=[CanServe(signal_type_id="sig.recon", prior=0.15)])],
                    concepts=[Concept(id="con.x", label="X", aliases=["ex"]), Concept(id="con.y", label="Y", parent_id="con.x")])


def test_valid_ontology_roundtrips():
    o = _onto(); o2 = Ontology.model_validate_json(o.model_dump_json())
    assert o2.element_count() == 7 and o2.concept_index()["ex"] == "con.x"


def test_induced_progression_is_ordered_by_proximity():
    prog = _onto().induced_progression("obj.a")
    assert [s["id"] for s in prog["signal_types"]] == ["sig.recon", "sig.reh"]
    assert prog["observables"][0]["can_serve"][0]["signal_type_id"] == "sig.recon"


@pytest.mark.parametrize("mutate", [
    lambda d: d["observables"][0]["can_serve"].append({"signal_type_id": "sig.missing", "prior": 0.1}),
    lambda d: d["signal_types"][0]["precursor_to"].append("sig.nope"),
    lambda d: d["concepts"][1].__setitem__("parent_id", "con.nope"),
    lambda d: d["objectives"][0]["concerns_entity_types"].append("et.nope"),
    lambda d: d["observables"][0]["can_serve"].__setitem__(0, {"signal_type_id": "sig.recon", "prior": 1.5}),
])
def test_dangling_or_invalid_refs_rejected(mutate):
    d = _onto().model_dump(); mutate(d)
    with pytest.raises(ValueError):
        Ontology.model_validate(d)


def test_no_baseline_functions_in_schema():
    assert "functions" not in Ontology.model_fields
