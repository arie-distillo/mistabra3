"""Embedders. HashingEmbedder is deterministic and offline (feature hashing over
unigrams+bigrams) — adequate for tests and a functional prefilter; plug a real
embedding endpoint for production quality."""
from __future__ import annotations
import hashlib, re
import numpy as np
import httpx


class HashingEmbedder:
    name = "hashing-v1"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            toks = re.findall(r"\w+", t.lower())
            feats = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
            for f in feats:
                h = int(hashlib.md5(f.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0 if (h >> 40) & 1 else -1.0
            n = np.linalg.norm(v)
            out.append((v / n if n else v).tolist())
        return out


class OpenAICompatibleEmbedder:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.name, self.model = f"api:{model}", model
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=60, headers={"Authorization": f"Bearer {api_key}"})

    def embed(self, texts: list[str]) -> list[list[float]]:
        r = self.client.post("/embeddings", json={"model": self.model, "input": texts})
        r.raise_for_status()
        return [d["embedding"] for d in r.json()["data"]]
