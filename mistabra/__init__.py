"""Mistabra — an analyst's co-pilot that turns intuition into reasoning.

Two layers, imported separately:

- `mistabra.core`: the reasoning layer. One stateless call, `Core.assess(mission, operational, trajectory)`, per
  episode (spec: docs/mistabra_core_api_v3.md).
- `mistabra.casebook`: the layer above it. `Casebook` keeps a case between calls (hypotheses, trajectories,
  look-back, analyst decisions, export) and drives the core episode by episode.
"""
__version__ = "0.4.0"
