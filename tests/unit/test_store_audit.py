from mistabra.store import Store


def test_audit_chain_and_rollback(tmp_path):
    s = Store(tmp_path)
    with s.tx("alice") as tx:
        tx.execute("INSERT INTO source(id,name,type,credibility,created_at) VALUES('s','x','t',0.5,'t')"); tx.audit("source.created", "source", "s", after={"n": 1})
    try:
        with s.tx("bob") as tx:
            tx.audit("x", "y", "z"); raise RuntimeError
    except RuntimeError:
        pass
    assert s.count("audit") == 1 and s.verify_audit()
    s.conn.execute("UPDATE audit SET after_json='{\"n\": 2}'")
    assert not s.verify_audit()
    s.close()
