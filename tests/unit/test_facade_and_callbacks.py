def test_callbacks_fire_and_stats(m, published, source):
    got = []
    m.callbacks.on("episode.ingested", got.append); m.callbacks.on("*", lambda e: got.append(("any", e["event"])))
    m.episodes.ingest(source, "Jens Hofer walked.", "host"); m.process()
    assert any(isinstance(g, dict) and g["event"] == "episode.ingested" for g in got) and ("any", "ontology.published") not in got
    s = m.stats()
    assert s["episodes"] == 1 and s["provider"] == "fake" and s["ontology_versions"] == 1 and s["vector_backend"] == "numpy"
    assert m.audit.verify_chain() and m.audit.llm_calls()


def test_data_directory_is_self_contained_and_reopenable(tmp_path, published, m, source):
    m.episodes.ingest(source, "Maria Klein walked.", "host"); m.process(); m.close()
    from mistabra import Mistabra, Settings
    from mistabra.llm import FakeProvider
    m2 = Mistabra(tmp_path, settings=Settings(data_dir=tmp_path, force_mock=True), provider=FakeProvider())
    assert m2.stats()["episodes"] == 1 and m2.current_ontology().version == 1 and m2.entities.find("Maria Klein")
    m2.close()
