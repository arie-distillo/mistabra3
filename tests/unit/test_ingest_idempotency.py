from mistabra.ingest import OccurrenceTime


def test_duplicate_episode_not_created_twice(m, source):
    a = m.episodes.ingest(source, "Same text.", "host"); b = m.episodes.ingest(source, "  Same text. ", "host")
    assert a["created"] and not b["created"] and a["episode_id"] == b["episode_id"]
    assert len(m.jobs.pending()) == 1


def test_occurrence_time_forms(m, source):
    e1 = m.episodes.get(m.episodes.ingest(source, "a", "host", occurrence_time="2026-08-01T10:00:00Z")["episode_id"])
    e2 = m.episodes.get(m.episodes.ingest(source, "b", "host", occurrence_time={"start": "2026-08-01", "end": "2026-08-03"})["episode_id"])
    e3 = m.episodes.get(m.episodes.ingest(source, "c", "host")["episode_id"])
    assert (e1["occurrence_kind"], e1["occurrence_start"]) == ("exact", "2026-08-01T10:00:00Z")
    assert (e2["occurrence_kind"], e2["occurrence_end"]) == ("range", "2026-08-03")
    assert e3["occurrence_kind"] == "unknown" and e3["occurrence_start"] is None and e3["ingested_at"]


def test_bad_occurrence_time_rejected():
    import pytest
    with pytest.raises(ValueError):
        OccurrenceTime.parse({"nonsense": 1})


def test_source_credibility_change_is_audited(m, source):
    m.sources.set_credibility(source, 0.3, "admin")
    ev = m.audit.events(object_type="source", object_id=source)[0]
    assert ev["action"] == "source.credibility_changed" and ev["before"]["credibility"] == 0.8 and ev["after"]["credibility"] == 0.3
