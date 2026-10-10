"""The record every diagnostic returns."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from credencegraph.core.attributes import JSON
from credencegraph.core.credence import Point
from credencegraph.core.errors import ValidationError

MISSING_PARAMETER = "missing-parameter"
UNANCHORED = "unanchored"
CORRELATED_SUPPORT = "correlated-support"
OVERCLAIM = "overclaim"
UNDERCLAIM = "underclaim"
SENSITIVITY = "sensitivity"
CRUX = "crux"
SINGLE_POINT_OF_FAILURE = "single-point-of-failure"
FAILURE_IMPACT = "failure-impact"
VALUE_OF_INFORMATION = "value-of-information"

# Probabilities within this distance of a threshold count as equal to it. A value that is exactly
# at the threshold on paper, such as 1 - (1 - 0.1) against 0.1, can land a few ulps either side of
# it in floating point, and should not flip a finding on or off.
ROUNDING = 1e-12


@dataclass(frozen=True, slots=True)
class Finding:
    """One result of a diagnostic.

    Attributes:
        id: Identifies the finding within a report: ``<diagnostic>:<subject>``, where the subject is
            a node id, ``base:<node id>`` or ``strength:<relation id>`` for a parameter, or
            ``<node id>:<provenance>`` for a group of supports, such as ``claim:document:doi:10.1/x``.
        diagnostic: The diagnostic that produced it, such as ``"crux"``.
        message: A one-line explanation for a human reader.
        nodes: The ids of the nodes involved.
        relations: The ids of the relations involved.
        value: The diagnostic's headline number, or ``None`` for a purely structural finding.
        details: The other numbers behind the finding, by name.
    """

    id: str
    diagnostic: str
    message: str
    nodes: tuple[str, ...] = ()
    relations: tuple[str, ...] = ()
    value: float | None = None
    details: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Store read-only copies of the id tuples and the details."""
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "relations", tuple(self.relations))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    def to_dict(self) -> dict[str, JSON]:
        """Return the finding as JSON-ready data.

        Returns:
            ``{"id", "diagnostic", "nodes", "relations", "value", "details", "message"}``.
        """
        return {
            "id": self.id,
            "diagnostic": self.diagnostic,
            "nodes": list(self.nodes),
            "relations": list(self.relations),
            "value": self.value,
            "details": dict(self.details),
            "message": self.message,
        }


def render_item(node_id: str, value: bool) -> str:
    """Write one node's value the way it is typed on the command line.

    The command line uses this form too, in its own messages, so the two always read alike.

    Args:
        node_id: The node's id.
        value: Its value.

    Returns:
        ``NODE=true`` or ``NODE=false``.
    """
    return f"{node_id}={str(value).lower()}"


def render_evidence(evidence: Mapping[str, bool]) -> str:
    """Write evidence for a finding's message the way it is typed on the command line.

    Args:
        evidence: Node ids and their observed values.

    Returns:
        ``a=true, b=false``.
    """
    return ", ".join(render_item(node_id, value) for node_id, value in evidence.items())


def probability_threshold(value: object, name: str) -> float:
    """Validate a threshold on a probability.

    Args:
        value: The candidate threshold.
        name: The argument's name, used in the error message.

    Returns:
        The threshold as a float.

    Raises:
        ValidationError: If ``value`` is not a real number in [0, 1].
    """
    try:
        return Point(value).p  # type: ignore[arg-type]
    except ValidationError as error:
        msg = f"{name}: {error}"
        raise ValidationError(msg) from error
