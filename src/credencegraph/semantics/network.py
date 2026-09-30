"""The compiled network: binary variables, their tables, and the parameters behind them."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, NamedTuple

import numpy as np

from credencegraph.core.credence import Credence, Point
from credencegraph.core.errors import ValidationError
from credencegraph.semantics.cpt import Term, exclusion_table, proposition_table

PROPOSITION = "proposition"
CONSTRAINT = "constraint"


class ParameterKey(NamedTuple):
    """Names one parameter of a network.

    Attributes:
        kind: ``"base"`` for a node's base, ``"strength"`` for a relation's strength.
        id: The id of the node or relation that carries the parameter.
    """

    kind: Literal["base", "strength"]
    id: str


@dataclass(frozen=True, slots=True)
class Link:
    """An inferential relation into a proposition variable.

    Attributes:
        relation: The relation id, which is also the id of its strength parameter.
        type: ``requires``, ``supports`` or ``refutes``.
        parent: Index of the parent variable.
    """

    relation: str
    type: str
    parent: int


@dataclass(frozen=True, slots=True)
class Variable:
    """A binary variable of the network.

    A proposition variable stands for one node, or for several nodes merged by ``equivalent``
    relations. A constraint variable encodes one ``exclusive`` relation and is always observed true.

    Attributes:
        index: Position of the variable in the network.
        name: The representative node id, or ``exclusive[<relation id>]`` for a constraint.
        kind: ``"proposition"`` or ``"constraint"``.
        members: The node ids the variable stands for; empty for a constraint.
        parents: Indices of the parent variables, ascending and distinct.
        base: The node id whose ``base`` is the variable's base; ``None`` for a constraint.
        links: The inferential relations into the variable.
        relation: The ``exclusive`` relation id of a constraint; ``None`` for a proposition.
    """

    index: int
    name: str
    kind: Literal["proposition", "constraint"]
    members: tuple[str, ...]
    parents: tuple[int, ...]
    base: str | None = None
    links: tuple[Link, ...] = ()
    relation: str | None = None


@dataclass(frozen=True, slots=True, eq=False)
class Factor:
    """A non-negative table over binary variables.

    Attributes:
        scope: The variable indices, one per table axis.
        table: A read-only array of shape ``(2,) * len(scope)``.
    """

    scope: tuple[int, ...]
    table: np.ndarray


def _frozen(array: np.ndarray) -> np.ndarray:
    """Return ``array`` marked read-only.

    Args:
        array: The array.

    Returns:
        The same array, now read-only.
    """
    array.setflags(write=False)
    return array


def _probability(value: object, name: str) -> float:
    """Validate a probability.

    Args:
        value: The candidate probability.
        name: Description of the value, used in the error message.

    Returns:
        The probability as a float.

    Raises:
        ValidationError: If ``value`` is not a real number in [0, 1].
    """
    try:
        return Point(value).p  # type: ignore[arg-type]
    except ValidationError as error:
        msg = f"{name}: {error}"
        raise ValidationError(msg) from error


class Network:
    """An immutable Bayesian network compiled from a graph.

    The network keeps each parameter's credence and the value each parameter currently takes; the
    tables are built from those values. A freshly compiled network uses the credence means, which
    give the point answer of every query. ``with_parameters`` and ``intervene`` return new networks
    and leave this one unchanged.
    """

    def __init__(
        self,
        variables: tuple[Variable, ...],
        parameters: Mapping[ParameterKey, Credence],
        carried: frozenset[str] = frozenset(),
        values: Mapping[ParameterKey, float] | None = None,
        interventions: Mapping[int, bool] | None = None,
    ) -> None:
        """Build the network's tables.

        Args:
            variables: The variables, with ``variables[i].index == i``.
            parameters: The credence of every parameter the variables refer to.
            carried: Ids of graph nodes that are not inference variables.
            values: Parameter values; a missing parameter takes its credence mean.
            interventions: Variables fixed by intervention, whose incoming relations are cut.

        Raises:
            ValidationError: If a variable is out of place or refers to an unknown parameter, or a
                value names an unknown parameter or is not a probability.
        """
        self._variables = tuple(variables)
        self._parameters = MappingProxyType(dict(parameters))
        self._carried = frozenset(carried)
        given: dict[ParameterKey, float] = {}
        for key, value in (values or {}).items():
            if key not in self._parameters:
                msg = f"network has no parameter {key!r}"
                raise ValidationError(msg)
            given[key] = _probability(value, f"parameter {key.kind} of {key.id!r}")
        self._values = MappingProxyType({key: given.get(key, credence.mean) for key, credence in parameters.items()})
        self._interventions = MappingProxyType(dict(interventions or {}))
        self._node_index: dict[str, int] = {}
        for position, variable in enumerate(self._variables):
            if variable.index != position:
                msg = f"variable {variable.name!r} has index {variable.index}, expected {position}"
                raise ValidationError(msg)
            for node_id in variable.members:
                self._node_index[node_id] = position
        self._factors = tuple(self._factor(variable) for variable in self._variables)

    def _factor(self, variable: Variable) -> Factor:
        """Build the factor of one variable from the current parameter values.

        Args:
            variable: The variable.

        Returns:
            Its conditional probability table, or a point mass if it is intervened on.

        Raises:
            ValidationError: If the variable refers to an unknown parameter.
        """
        if variable.index in self._interventions:
            table = np.zeros(2)
            table[int(self._interventions[variable.index])] = 1.0
            return Factor((variable.index,), _frozen(table))
        if variable.kind == CONSTRAINT:
            table = exclusion_table(len(variable.parents))
        else:
            position = {parent: axis for axis, parent in enumerate(variable.parents)}
            terms = [
                Term(position[link.parent], link.type, self._value(ParameterKey("strength", link.relation)))
                for link in variable.links
            ]
            base = self._value(ParameterKey("base", variable.base or ""))
            table = proposition_table(base, terms, len(variable.parents))
        return Factor((*variable.parents, variable.index), _frozen(table))

    def _value(self, key: ParameterKey) -> float:
        """Look up a parameter's current value.

        Args:
            key: The parameter.

        Returns:
            Its value.

        Raises:
            ValidationError: If the network has no such parameter.
        """
        if key not in self._values:
            msg = f"network has no parameter {key.kind} of {key.id!r}"
            raise ValidationError(msg)
        return self._values[key]

    @property
    def variables(self) -> tuple[Variable, ...]:
        """The variables, propositions first in topological order, then constraints."""
        return self._variables

    @property
    def factors(self) -> tuple[Factor, ...]:
        """One factor per variable: its table given its parents, over ``(*parents, variable)``."""
        return self._factors

    @property
    def parameters(self) -> Mapping[ParameterKey, Credence]:
        """Read-only view of every parameter's credence."""
        return self._parameters

    @property
    def values(self) -> Mapping[ParameterKey, float]:
        """Read-only view of the value every parameter currently takes."""
        return self._values

    @property
    def interventions(self) -> Mapping[int, bool]:
        """Read-only view of the variables fixed by intervention, by index."""
        return self._interventions

    @property
    def constraints(self) -> Mapping[int, bool]:
        """The constraint variables, each observed true in every query."""
        return MappingProxyType({v.index: True for v in self._variables if v.kind == CONSTRAINT})

    def __len__(self) -> int:
        """Return the number of variables, constraints included."""
        return len(self._variables)

    def __repr__(self) -> str:
        """Summarise the network."""
        n_constraints = len(self.constraints)
        return f"Network(propositions={len(self) - n_constraints}, constraints={n_constraints})"

    def index(self, node_id: str) -> int:
        """Find the variable that stands for a node.

        Args:
            node_id: A node id; nodes merged by ``equivalent`` share one variable.

        Returns:
            The variable index.

        Raises:
            ValidationError: If the node is unknown or is not an inference variable.
        """
        if node_id in self._node_index:
            return self._node_index[node_id]
        if node_id in self._carried:
            msg = f"node {node_id!r} is not an inference variable: it has no base and no inferential relation"
            raise ValidationError(msg)
        msg = f"no node {node_id!r} in the network"
        raise ValidationError(msg)

    def with_parameters(self, values: Mapping[ParameterKey, float]) -> Network:
        """Return a copy of the network in which some parameters take the given values.

        Args:
            values: New values by parameter; every other parameter keeps its current value.

        Returns:
            The new network, with the same interventions.

        Raises:
            ValidationError: If a key is not a parameter of the network or a value is not in [0, 1].
        """
        merged = {**self._values, **values}
        return Network(self._variables, self._parameters, self._carried, merged, self._interventions)

    def intervene(self, assignment: Mapping[str, bool]) -> Network:
        """Return a copy of the network with some propositions fixed by intervention.

        Intervening on a node cuts the relations into its variable and fixes the variable's value.
        Unlike conditioning, it leaves the node's own premises unchanged. A later intervention on the
        same variable replaces an earlier one. ``exclusive`` constraints stay observed.

        Args:
            assignment: Node ids and the value each is set to.

        Returns:
            The intervened network.

        Raises:
            ValidationError: If a node is not an inference variable, a value is not a bool, or two
                merged nodes are set to different values.
        """
        fixed: dict[int, bool] = {}
        for node_id, value in assignment.items():
            if not isinstance(value, bool):
                msg = f"intervention on {node_id!r} must be True or False, got {value!r}"
                raise ValidationError(msg)
            index = self.index(node_id)
            if fixed.get(index, value) != value:
                msg = f"intervention sets {self._variables[index].name!r} both true and false"
                raise ValidationError(msg)
            fixed[index] = value
        return Network(self._variables, self._parameters, self._carried, self._values, {**self._interventions, **fixed})
