from mistabra.store import Store


def test_audit_written_in_same_transaction_and_chained(tmp_path):
    s = Store(tmp_path)
    with s.tx("alice") as tx:
        tx.execute("INSERT INTO source(id,name,type,credibility,config_json,created_at) VALUES('s1','x','t',0.5,'{}','now')")
        tx.audit("source.registered", "source", "s1", after={"name": "x"})
    with s.tx("bob") as tx:
        tx.audit("noop", "source", "s1")
    ev = s.all("SELECT * FROM audit_event ORDER BY rowid")
    assert [e["actor"] for e in ev] == ["alice", "bob"]
    assert ev[1]["prev_hash"] == ev[0]["hash"] and ev[0]["prev_hash"] == ""
    assert s.verify_audit_chain()
    s.close()


def test_rollback_removes_change_and_its_audit(tmp_path):
    s = Store(tmp_path)
    try:
        with s.tx("alice") as tx:
            tx.execute("INSERT INTO source(id,name,type,credibility,config_json,created_at) VALUES('s1','x','t',0.5,'{}','now')")
            tx.audit("source.registered", "source", "s1")
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert s.one("SELECT COUNT(*) AS n FROM source")["n"] == 0
    assert s.one("SELECT COUNT(*) AS n FROM audit_event")["n"] == 0
    s.close()


def test_tampering_breaks_chain(tmp_path):
    s = Store(tmp_path)
    with s.tx("a") as tx:
        tx.audit("x", "y", "z", after={"v": 1})
    s.conn.execute("UPDATE audit_event SET after_json='{\"v\": 2}'")
    assert not s.verify_audit_chain()
    s.close()
