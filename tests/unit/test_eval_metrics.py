from mistabra.eval import ontology_metrics, resolution_metrics


def test_ontology_metrics_fuzzy_label_match():
    m = ontology_metrics({"objectives": ["Burglary of homes in Riverside neighbourhood", "Something else"]}, {"objectives": ["Burglary of homes in the Riverside neighbourhood", "Vandalism"]})
    assert m["objectives"]["precision"] == 0.5 and m["objectives"]["recall"] == 0.5


def test_resolution_metrics_pairwise():
    lab = {"clusters": [{"entity": "A", "type": "person", "surfaces": ["A One", "One"]}, {"entity": "B", "type": "person", "surfaces": ["B Two"]}], "must_not_merge": [["A", "B"]]}
    good = [{"surface": "A One", "entity_type": "person", "entity_id": "x"}, {"surface": "One", "entity_type": "person", "entity_id": "x"}, {"surface": "B Two", "entity_type": "person", "entity_id": "y"}]
    r = resolution_metrics(good, lab); assert (r["pairwise_precision"], r["pairwise_recall"], r["forbidden_merges"]) == (1.0, 1.0, 0)
    bad = [dict(g, entity_id="x") for g in good]
    r = resolution_metrics(bad, lab); assert r["pairwise_precision"] < 1 and r["forbidden_merges"] == 1
    unres = [dict(good[0]), dict(good[1], entity_id=None), good[2]]
    r = resolution_metrics(unres, lab); assert r["pairwise_recall"] == 0.0 and r["unresolved"] == 1
