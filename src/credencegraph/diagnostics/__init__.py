"""Diagnostics: structural checks, stated-versus-computed consistency, and the weak points of a target.

Every diagnostic returns a list of ``Finding`` records, each with an id, the ids of the nodes and
relations involved, a headline value and a one-line explanation; ``Finding.to_dict`` gives its JSON
form.
"""

from __future__ import annotations

from credencegraph.diagnostics.claims import DEFAULT_CLAIM_THRESHOLD, claims
from credencegraph.diagnostics.records import (
    CRUX,
    FAILURE_IMPACT,
    MISSING_PARAMETER,
    OVERCLAIM,
    SENSITIVITY,
    SINGLE_POINT_OF_FAILURE,
    UNANCHORED,
    UNDERCLAIM,
    VALUE_OF_INFORMATION,
    Finding,
)
from credencegraph.diagnostics.report import Report, diagnose, diagnose_many
from credencegraph.diagnostics.structure import missing_parameters, unanchored
from credencegraph.diagnostics.weak_points import (
    DEFAULT_FAILURE_THRESHOLD,
    crux,
    derivatives,
    failure_impact,
    sensitivity,
    single_points_of_failure,
    value_of_information,
)

__all__ = [
    "CRUX",
    "DEFAULT_CLAIM_THRESHOLD",
    "DEFAULT_FAILURE_THRESHOLD",
    "FAILURE_IMPACT",
    "MISSING_PARAMETER",
    "OVERCLAIM",
    "SENSITIVITY",
    "SINGLE_POINT_OF_FAILURE",
    "UNANCHORED",
    "UNDERCLAIM",
    "VALUE_OF_INFORMATION",
    "Finding",
    "Report",
    "claims",
    "crux",
    "derivatives",
    "diagnose",
    "diagnose_many",
    "failure_impact",
    "missing_parameters",
    "sensitivity",
    "single_points_of_failure",
    "unanchored",
    "value_of_information",
]
