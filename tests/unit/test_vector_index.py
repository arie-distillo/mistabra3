from mistabra.vectors import VectorIndex
from mistabra.llm import HashingEmbedder


def test_hashing_embedder_deterministic_and_normalized():
    e = HashingEmbedder(64)
    a, b = e.embed(["photographed a house"])[0], e.embed(["photographed a house"])[0]
    assert a == b and abs(sum(x * x for x in a) - 1) < 1e-5 and len(a) == 64


def test_numpy_backend_upsert_search_persist(tmp_path):
    v = VectorIndex(tmp_path, HashingEmbedder(128), "numpy")
    v.upsert("obs", [("a", "photographed a house on Elm Street", {"k": 1}), ("b", "bought groceries at the supermarket", {})])
    assert v.search("obs", "photographed the house", 1)[0]["id"] == "a"
    v.upsert("obs", [("a", "bought bread", {})])
    assert v.count("obs") == 2 and v.search("obs", "bought", 1)[0]["id"] in ("a", "b")
    v2 = VectorIndex(tmp_path, HashingEmbedder(128), "numpy")
    assert v2.count("obs") == 2 and v2.search("obs", "supermarket groceries", 1)[0]["id"] == "b"


def test_empty_table_search(tmp_path):
    assert VectorIndex(tmp_path, HashingEmbedder(16), "numpy").search("none", "x") == []
