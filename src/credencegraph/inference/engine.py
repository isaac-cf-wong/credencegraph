"""Queries and the interface every inference engine implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from credencegraph.core.errors import ValidationError
from credencegraph.inference.errors import ZeroProbabilityError
from credencegraph.semantics.network import Network


def _frozen_assignment(assignment: Mapping[str, bool], role: str) -> Mapping[str, bool]:
    """Validate and copy an assignment of node ids to truth values.

    Args:
        assignment: Node ids and values.
        role: What the assignment is for, used in error messages.

    Returns:
        A read-only copy.

    Raises:
        ValidationError: If the assignment is not a mapping, or a value is not a bool.
    """
    if not isinstance(assignment, Mapping):
        msg = f"{role} must be a mapping of node ids to True or False, got {assignment!r}"
        raise ValidationError(msg)
    for node_id, value in assignment.items():
        if not isinstance(value, bool):
            msg = f"{role} value for {node_id!r} must be True or False, got {value!r}"
            raise ValidationError(msg)
    return MappingProxyType(dict(assignment))


@dataclass(frozen=True, slots=True)
class Query:
    """A probability query: ``P(target | evidence)`` after the interventions are applied.

    Attributes:
        target: Node ids and the values whose joint probability is asked for.
        evidence: Node ids and the observed values conditioned on.
        interventions: Node ids and the values they are set to; relations into them are cut.
    """

    target: Mapping[str, bool]
    evidence: Mapping[str, bool] = field(default_factory=dict)
    interventions: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the three assignments and store read-only copies."""
        object.__setattr__(self, "target", _frozen_assignment(self.target, "target"))
        object.__setattr__(self, "evidence", _frozen_assignment(self.evidence, "evidence"))
        object.__setattr__(self, "interventions", _frozen_assignment(self.interventions, "interventions"))


def _resolve(network: Network, assignment: Mapping[str, bool]) -> dict[int, int] | None:
    """Map node ids to variable indices.

    Args:
        network: The network.
        assignment: Node ids and values.

    Returns:
        Variable indices and values, or ``None`` when merged nodes are given different values, which
        no world satisfies.

    Raises:
        ValidationError: If a node is not an inference variable of the network.
    """
    resolved: dict[int, int] = {}
    for node_id, value in assignment.items():
        index = network.index(node_id)
        if resolved.get(index, int(value)) != int(value):
            return None
        resolved[index] = int(value)
    return resolved


def _combine(first: dict[int, int] | None, second: dict[int, int] | None) -> dict[int, int] | None:
    """Join two assignments.

    Args:
        first: An assignment, or ``None`` for an unsatisfiable one.
        second: Another assignment, or ``None``.

    Returns:
        Their union, or ``None`` when either is unsatisfiable or they disagree on a variable.
    """
    if first is None or second is None:
        return None
    for index, value in second.items():
        if first.get(index, value) != value:
            return None
    return {**first, **second}


class Engine(ABC):
    """An inference engine.

    An engine only has to compute the probability of a partial assignment. ``query`` turns that into
    conditional and interventional answers, and observes every ``exclusive`` constraint as true.
    """

    @abstractmethod
    def probability(self, network: Network, assignment: Mapping[int, int]) -> float:
        """Compute the probability that the given variables take the given values.

        Args:
            network: The network.
            assignment: Variable indices and values (0 or 1); every other variable is summed out.

        Returns:
            The probability.
        """

    def query(self, network: Network, query: Query) -> float:
        """Answer a query.

        Args:
            network: The network, at the parameter values the answer should use.
            query: The query.

        Returns:
            ``P(target | evidence, constraints)`` in the network after the interventions.

        Raises:
            ValidationError: If the query names a node that is not an inference variable, or the
                interventions set merged nodes to different values.
            ZeroProbabilityError: If the evidence and constraints have probability zero.
        """
        if query.interventions:
            network = network.intervene(query.interventions)
        evidence = _combine(_resolve(network, query.evidence), {i: int(v) for i, v in network.constraints.items()})
        target = _resolve(network, query.target)
        if evidence is None:
            denominator = 0.0
        elif evidence:
            denominator = self.probability(network, evidence)
        else:
            denominator = 1.0
        if denominator <= 0.0 or evidence is None:
            msg = "the evidence, together with the exclusive constraints, has probability zero"
            raise ZeroProbabilityError(msg)
        joint = _combine(evidence, target)
        if joint is None:
            return 0.0
        if len(joint) == len(evidence):
            return 1.0
        answer = self.probability(network, joint) / denominator
        # Rounding can carry the ratio a few ulps past the unit interval.
        return min(1.0, max(0.0, answer))
