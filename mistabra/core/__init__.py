"""Mistabra Core — the reasoning layer: one stateless call per episode (spec v3, docs/mistabra_core_api_v3.md).

    from mistabra.core import Core, Episode, Settings
    result = Core(Settings.from_env()).assess(mission, operational, TrajectoryInput(episode=Episode(...)))

It keeps no state; `mistabra.casebook` is the layer that keeps a case between calls.
"""
from .engine import Core, assess
from .settings import Settings
from .types import (AssessedTrajectory, Assessment, AssessmentGroup, Document, Episode, Hypothesis, HypothesisDraft,
                    HypothesisRef, MissionContext, OperationalContext, Provenance, Risk, SignalContent, SignalDraft,
                    Standing, StoredTrajectory, TrajectoryInput)

__all__ = ["Core", "assess", "Settings", "AssessedTrajectory", "Assessment", "AssessmentGroup", "Document", "Episode",
           "Hypothesis", "HypothesisDraft", "HypothesisRef", "MissionContext", "OperationalContext", "Provenance", "Risk",
           "SignalContent", "SignalDraft", "Standing", "StoredTrajectory", "TrajectoryInput"]
