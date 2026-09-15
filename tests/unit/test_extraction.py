def test_extraction_produces_observations_mentions_and_vectors(m, published, source):
    r = m.episodes.ingest(source, "A passer-by photographed Jens Hofer's house on Elm Street. Later he watched the Community Centre at night.", "host")
    assert m.process() == {"done": 2, "failed": 0}
    obs = m.episodes.observations(r["episode_id"])
    assert len(obs) == 2 and obs[0]["observable_ids"] == ["obs.image_capture_of_place"] and obs[1]["modifiers"].get("time_of_day") == "night"
    assert all(o["pipeline_version"] == "extract-1" and o["llm_call_id"] for o in obs)
    mentions = m.episodes.mentions(r["episode_id"])
    assert {x["surface"] for x in mentions} >= {"Jens Hofer's", "Elm Street", "Community Centre"}
    assert m.vectors.count("observation") == 2
    top = m.vectors.search("observation", "photographed a house", 1)[0]
    assert top["id"] == obs[0]["id"]
    assert m.episodes.get(r["episode_id"])["extraction_status"] == "done"


def test_extraction_without_ontology_uses_default_entity_types(m, source):
    r = m.episodes.ingest(source, "Maria Klein walked to Maple Road.", "host")
    m.process()
    assert {x["entity_type"] for x in m.episodes.mentions(r["episode_id"])} <= {"person", "location", "organization", "institution", "vehicle"}


def test_extraction_failure_is_recorded_not_raised_by_runner(m, published, source):
    from mistabra.llm import FakeProvider
    m.gateway.provider = FakeProvider(faults={"extract.episode": "garbage"})
    r = m.episodes.ingest(source, "Anything.", "host")
    res = m.process()
    assert res["failed"] == 1 and m.episodes.get(r["episode_id"])["extraction_status"] == "failed"
    job = m.store.one("SELECT status, error FROM job")
    assert job["status"] == "failed" and "LLMOutputError" in job["error"]
    assert any(a["action"] == "episode.extraction_failed" for a in m.audit.events(object_type="episode"))
