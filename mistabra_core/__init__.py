"""Mistabra Core — the reasoning layer of an analyst's co-pilot that turns intuition into reasoning.

One call: `assess(mission, operational, trajectory, *, scope, detail)` (spec v3, docs/mistabra_core_api_v3.md).
`Casebook` keeps a case between calls: hypotheses, trajectories, look-back, analyst decisions, export.
"""
from .casebook import Casebook, Ingested
from .core import Core, assess
from .settings import Settings
from .types import (AssessedTrajectory, Assessment, AssessmentGroup, Document, Episode, Hypothesis, HypothesisDraft,
                    HypothesisRef, MissionContext, OperationalContext, Provenance, Risk, SignalContent, SignalDraft,
                    Standing, StoredTrajectory, TrajectoryInput)

__version__ = "0.3.0"
__all__ = ["Casebook", "Ingested", "Core", "assess", "Settings", "AssessedTrajectory", "Assessment", "AssessmentGroup", "Document", "Episode",
           "Hypothesis", "HypothesisDraft", "HypothesisRef", "MissionContext", "OperationalContext", "Provenance", "Risk",
           "SignalContent", "SignalDraft", "Standing", "StoredTrajectory", "TrajectoryInput"]
