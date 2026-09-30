"""The four query types: marginal, joint, conditional and interventional probabilities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from credencegraph.core.errors import ValidationError
from credencegraph.inference.elimination import VariableElimination
from credencegraph.inference.engine import Engine, Query
from credencegraph.semantics.network import Network


@dataclass(frozen=True, slots=True)
class Answer:
    """The answer to a query.

    Attributes:
        point: The point answer: the probability with every parameter at its credence mean. Because
            the network is multilinear in its independent parameters, this equals the predictive
            probability averaged over the parameters' distributions.
    """

    point: float

    def to_dict(self) -> dict[str, float]:
        """Return the answer as JSON-ready data.

        Returns:
            ``{"point": ...}``.
        """
        return {"point": self.point}


def _as_assignment(target: str | Mapping[str, bool] | Iterable[str], role: str) -> dict[str, bool]:
    """Read a target: a node id, an iterable of node ids that are all true, or an assignment.

    Args:
        target: The target.
        role: What the target is for, used in error messages.

    Returns:
        The assignment.

    Raises:
        ValidationError: If ``target`` is none of those forms.
    """
    if isinstance(target, str):
        return {target: True}
    if isinstance(target, Mapping):
        return dict(target)
    if isinstance(target, Iterable):
        ids = list(target)
        if all(isinstance(node_id, str) for node_id in ids):
            return dict.fromkeys(ids, True)
    msg = f"{role} must be a node id, an iterable of node ids or a mapping of node ids to bools, got {target!r}"
    raise ValidationError(msg)


def _run(network: Network, query: Query, engine: Engine | None) -> Answer:
    """Answer a query with the given engine, variable elimination by default.

    Args:
        network: The network.
        query: The query.
        engine: The engine, or ``None`` for ``VariableElimination()``.

    Returns:
        The answer.
    """
    return Answer((engine or VariableElimination()).query(network, query))


def marginal(network: Network, node: str, value: bool = True, *, engine: Engine | None = None) -> Answer:
    """Compute ``P(node = value)``.

    Args:
        network: The network.
        node: A node id.
        value: The value asked about.
        engine: The inference engine; variable elimination by default.

    Returns:
        The answer.
    """
    return _run(network, Query({node: value}), engine)


def joint(network: Network, assignment: Mapping[str, bool] | Iterable[str], *, engine: Engine | None = None) -> Answer:
    """Compute the probability of a joint assignment, ``P(X = x and Y = y and ...)``.

    Args:
        network: The network.
        assignment: Node ids and values, or an iterable of node ids that are all asked to be true.
        engine: The inference engine; variable elimination by default.

    Returns:
        The answer.
    """
    return _run(network, Query(_as_assignment(assignment, "assignment")), engine)


def conditional(
    network: Network,
    target: str | Mapping[str, bool],
    given: Mapping[str, bool],
    *,
    engine: Engine | None = None,
) -> Answer:
    """Compute ``P(target | given)``.

    Conditioning on a node also updates the node's own premises: observing that a conclusion is
    false is evidence against what supports it.

    Args:
        network: The network.
        target: A node id, asked to be true, or an assignment.
        given: The observed values.
        engine: The inference engine; variable elimination by default.

    Returns:
        The answer.
    """
    return _run(network, Query(_as_assignment(target, "target"), given), engine)


def intervene(
    network: Network,
    target: str | Mapping[str, bool],
    set: Mapping[str, bool],  # noqa: A002 - the documented keyword: intervene(X, set={Y: False})
    *,
    given: Mapping[str, bool] | None = None,
    engine: Engine | None = None,
) -> Answer:
    """Compute ``P(target | do(set), given)``: cut the relations into the set nodes and fix them.

    This answers "suppose this premise is simply wrong": unlike conditioning, it does not update the
    set node's own premises.

    Args:
        network: The network.
        target: A node id, asked to be true, or an assignment.
        set: The nodes to set and their values.
        given: Values observed after the intervention, if any.
        engine: The inference engine; variable elimination by default.

    Returns:
        The answer.
    """
    return _run(network, Query(_as_assignment(target, "target"), given or {}, set), engine)
