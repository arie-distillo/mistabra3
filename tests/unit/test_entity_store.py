def _ingest_all(m, source, texts):
    ids = [m.episodes.ingest(source, t, "host")["episode_id"] for t in texts]
    m.process(); return ids


def test_mentions_resolve_across_episodes_and_link_objects(m, published, source):
    e1, e2 = _ingest_all(m, source, ["Jens Hofer walked on Elm Street.", "Hofer returned to Elm Street."])
    people = m.entities.list("person")
    assert len(people) == 1 and people[0]["canonical_name"] == "Jens Hofer" and "Hofer" in people[0]["aliases"]
    ent = people[0]["id"]
    assert {l["object_id"] for l in m.entities.links(ent) if l["object_type"] == "episode"} == {e1, e2}
    assert {e["canonical_name"] for e in m.entities.entities_for("episode", e2)} == {"Jens Hofer", "Elm Street"}
    assert len(m.entities.mentions_of(ent)) == 2


def test_merge_relinks_mentions_and_is_audited(m, published, source):
    _ingest_all(m, source, ["Karl Brandt posted online.", "Thomas Berg bought a shotgun."])
    a, b = (m.entities.find("Karl Brandt")[0]["id"], m.entities.find("Thomas Berg")[0]["id"])
    m.entities.merge(a, b, "analyst")
    assert m.entities.get(b)["status"] == "merged" and m.entities.get(b)["merged_into"] == a
    assert "Thomas Berg" in m.entities.get(a)["aliases"] and len(m.entities.mentions_of(a)) == 2 and m.entities.links(b) == []
    ev = m.audit.events(object_type="entity", object_id=a)[0]
    assert ev["action"] == "entity.merged" and ev["before"]["drop"]["canonical_name"] == "Thomas Berg"
    # mentions are untouched text-wise (surface preserved)
    assert {x["surface"] for x in m.entities.mentions_of(a)} == {"Karl Brandt", "Thomas Berg"}


def test_split_and_relink(m, published, source):
    _ingest_all(m, source, ["Karl Brandt posted online.", "Brandt bought accelerant."])
    ent = m.entities.find("Karl Brandt")[0]["id"]
    second = [x for x in m.entities.mentions_of(ent) if x["surface"] == "Brandt"][0]["id"]
    new = m.entities.split(ent, [second], "Another Brandt", "analyst")
    assert m.entities.get(new)["canonical_name"] == "Another Brandt" and len(m.entities.mentions_of(ent)) == 1
    m.entities.relink(second, None, "analyst")
    assert m.entities.unresolved()[0]["id"] == second
    m.entities.relink(second, ent, "analyst")
    assert len(m.entities.mentions_of(ent)) == 2


def test_unresolved_state_persists_and_is_listable(m, published, source):
    _ingest_all(m, source, ["Northfield Gazette published a story.", "Northfield hosted a fair."])
    unres = m.entities.unresolved()
    assert len(unres) == 1 and unres[0]["surface"] == "Northfield" and unres[0]["entity_id"] is None


def test_graph_view(m, published, source):
    _ingest_all(m, source, ["Jens Hofer walked on Elm Street."])
    g = m.entities.graph()
    assert g.number_of_nodes() >= 3 and g.number_of_edges() >= 2
