"""Entity resolution backends. All are conservative: `unresolved` is a valid outcome.

SimpleResolver (default): normalization + alias/exact match + rapidfuzz similarity,
with model adjudication in the ambiguous band.
SplinkResolver (optional): probabilistic record linkage (Fellegi-Sunter) on DuckDB,
with hand-set m/u parameters until confirmed links exist to train on.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from pydantic import BaseModel
from rapidfuzz import fuzz

_ARTICLES = {"the", "a", "an", "mr", "mrs", "ms", "dr"}


def normalize_name(s: str) -> str:
    s = re.sub(r"'s\b", "", s)
    toks = [t for t in re.findall(r"\w+", s.lower()) if t not in _ARTICLES]
    return " ".join(toks)


def canonical_name(surface: str) -> str:
    """Canonical form for a new entity: strip possessive and leading articles, keep case."""
    s = re.sub(r"'s\b", "", surface.strip())
    return re.sub(r"^(?:the|a|an)\s+", "", s, flags=re.I).strip() or surface


def name_similarity(a: str, b: str) -> float:
    """0-100. Subset matches are only strong when the shorter side is multi-token or is the
    longer side's final token (surname / head noun): 'Hofer' ~ 'Jens Hofer' yes,
    'Northfield' ~ 'Northfield Gazette' only ambiguous."""
    ta, tb = a.split(), b.split()
    if not ta or not tb: return 0.0
    if ta == tb: return 100.0
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if set(short) <= set(long_):
        return 95.0 if (len(short) >= 2 or short[-1] == long_[-1]) else 80.0
    return float(fuzz.token_sort_ratio(a, b))


@dataclass
class Decision:
    outcome: str                  # link | new | unresolved
    entity_id: str | None
    score: float
    rationale: str


class _Adjudication(BaseModel):
    decision: str; entity_id: str | None = None; confidence: float = 0.0; rationale: str = ""


class SimpleResolver:
    name = "simple"

    def __init__(self, gateway=None, link_threshold: float = 92.0, ambiguous_threshold: float = 74.0):
        self.gw, self.link_t, self.amb_t = gateway, link_threshold, ambiguous_threshold

    def resolve(self, mention: dict, candidates: list[dict]) -> Decision:
        surf = normalize_name(mention["surface"])
        if not surf:
            return Decision("unresolved", None, 0.0, "empty surface")
        best, best_score = None, 0.0
        for c in candidates:
            names = [c["canonical_name"], *c.get("aliases", [])]
            for n in names:
                nn = normalize_name(n)
                if nn == surf:
                    return Decision("link", c["id"], 1.0, f"exact match on '{n}'")
            score = max(name_similarity(surf, normalize_name(n)) for n in names)
            if score > best_score:
                best, best_score = c, score
        if best is None:
            return Decision("new", None, 0.0, "no candidates of this type")
        if best_score >= self.link_t:
            return Decision("link", best["id"], best_score / 100, f"fuzzy match {best_score:.0f} to '{best['canonical_name']}'")
        if best_score >= self.amb_t:
            if self.gw is None:
                return Decision("unresolved", None, best_score / 100, f"ambiguous ({best_score:.0f}) and no adjudicator")
            top = sorted(candidates, key=lambda c: -max(name_similarity(surf, normalize_name(n)) for n in [c["canonical_name"], *c.get("aliases", [])]))[:5]
            adj, _ = self.gw.call("entities.adjudicate", "operational",
                                  {"mention": {"surface": mention["surface"], "entity_type": mention["entity_type"], "attributes": mention.get("attributes", {}), "context": mention.get("context", "")},
                                   "candidates": [{"id": c["id"], "canonical_name": c["canonical_name"], "aliases": c.get("aliases", []), "attributes": c.get("attributes", {})} for c in top]}, _Adjudication)
            if adj.decision == "link" and adj.entity_id in {c["id"] for c in top} and adj.confidence >= 0.6:
                return Decision("link", adj.entity_id, adj.confidence, "adjudicated: " + adj.rationale)
            if adj.decision == "new" and adj.confidence >= 0.6:
                return Decision("new", None, adj.confidence, "adjudicated: " + adj.rationale)
            return Decision("unresolved", None, adj.confidence, "adjudicated unresolved: " + adj.rationale)
        return Decision("new", None, best_score / 100, f"best similarity {best_score:.0f} below ambiguity band")


class SplinkResolver:
    """Optional backend. Requires `splink` and `pandas`. Same interface as SimpleResolver."""
    name = "splink"

    def __init__(self, link_threshold: float = 0.85, ambiguous_threshold: float = 0.4, gateway=None):
        import pandas as pd  # noqa: F401
        from splink import Linker, SettingsCreator, DuckDBAPI, block_on  # noqa: F401
        self.link_t, self.amb_t, self.gw = link_threshold, ambiguous_threshold, gateway

    def resolve(self, mention: dict, candidates: list[dict]) -> Decision:
        import logging, pandas as pd
        from splink import Linker, SettingsCreator, DuckDBAPI, block_on
        import splink.comparison_library as cl
        if not candidates:
            return Decision("new", None, 0.0, "no candidates of this type")
        logging.getLogger("splink").setLevel(logging.ERROR)
        def rec(uid, name, etype): n = normalize_name(name); return {"unique_id": uid, "name": n, "last": (n.split() or [""])[-1], "etype": etype}
        left = pd.DataFrame([rec("m", mention["surface"], mention["entity_type"])])
        right = pd.DataFrame([rec(c["id"], c["canonical_name"], c["entity_type"]) for c in candidates])
        settings = SettingsCreator(link_type="link_only", probability_two_random_records_match=0.2,
            comparisons=[cl.JaroWinklerAtThresholds("name", [0.95, 0.85]).configure(m_probabilities=[0.6, 0.2, 0.15, 0.05], u_probabilities=[0.005, 0.015, 0.05, 0.93]),
                         cl.ExactMatch("last").configure(m_probabilities=[0.9, 0.1], u_probabilities=[0.05, 0.95])],
            blocking_rules_to_generate_predictions=[block_on("etype")], retain_intermediate_calculation_columns=False)
        linker = Linker([left, right], settings, db_api=DuckDBAPI(), input_table_aliases=["m", "e"])
        df = linker.inference.predict(threshold_match_probability=0.0).as_pandas_dataframe()
        if df.empty:
            return Decision("new", None, 0.0, "no comparable candidates")
        df["cand"] = df.apply(lambda r: r["unique_id_r"] if r["unique_id_l"] == "m" else r["unique_id_l"], axis=1)
        top = df.sort_values("match_probability", ascending=False).iloc[0]
        p = float(top["match_probability"])
        if p >= self.link_t: return Decision("link", top["cand"], p, f"splink p={p:.2f}")
        if p >= self.amb_t: return Decision("unresolved", None, p, f"splink ambiguous p={p:.2f}")
        return Decision("new", None, p, f"splink p={p:.2f} below ambiguity band")
