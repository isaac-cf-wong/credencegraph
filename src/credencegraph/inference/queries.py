"""The four query types: marginal, joint, conditional and interventional probabilities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from credencegraph.core.errors import ValidationError
from credencegraph.inference.elimination import VariableElimination
from credencegraph.inference.engine import Engine, Query
from credencegraph.inference.uncertainty import DEFAULT_DRAWS, Band, spread
from credencegraph.semantics.network import Network

# A random generator, or a seed for one; ``None`` draws fresh entropy.
Seed = np.random.Generator | int | None


@dataclass(frozen=True, slots=True)
class Answer:
    """The answer to a query.

    Attributes:
        point: The point answer: the probability with every parameter at its credence mean. Because
            the network is multilinear in its independent parameters, this equals the predictive
            ratio ``E[P(A, E)] / E[P(E)]``, where each expectation is taken over the parameters'
            distributions and the ``exclusive`` constraints count as evidence on both sides. It is
            not the average of ``P(A | E)`` over the parameters, which generally differs.
        band: The 5%, 50% and 95% quantiles of the answer over parameter draws, or ``None`` when no
            draws were made because every parameter is a ``Point`` or ``draws`` was 0.
        mean_over_draws: The average of the per-draw answers, reported separately from ``point``
            because it is a different quantity; ``None`` exactly when ``band`` is.
        draws: The number of parameter draws behind ``band`` and ``mean_over_draws``; 0 when none.
    """

    point: float
    band: Band | None = None
    mean_over_draws: float | None = None
    draws: int = 0

    def __post_init__(self) -> None:
        """Check that the band, its mean and the draw count come together or not at all.

        Raises:
            ValidationError: If some but not all of them are given.
        """
        if (self.band is None) != (self.mean_over_draws is None) or (self.band is None) != (self.draws == 0):
            msg = "band, mean_over_draws and a positive draws count must be given together or not at all"
            raise ValidationError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return the answer as JSON-ready data.

        Returns:
            ``{"point": ...}``, plus ``"band"``, ``"mean_over_draws"`` and ``"draws"`` when there is a band.
        """
        if self.band is None:
            return {"point": self.point}
        return {
            "point": self.point,
            "band": self.band.to_dict(),
            "mean_over_draws": self.mean_over_draws,
            "draws": self.draws,
        }


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


def _run(network: Network, query: Query, engine: Engine | None, draws: int, rng: Seed) -> Answer:
    """Answer a query with the given engine, variable elimination by default.

    Args:
        network: The network.
        query: The query.
        engine: The engine, or ``None`` for ``VariableElimination()``.
        draws: The number of parameter draws for the band; 0 skips it.
        rng: A random generator, or a seed for one, for the draws.

    Returns:
        The answer.
    """
    engine = engine or VariableElimination()
    point = engine.query(network, query)
    estimate = spread(network, query, engine, draws, rng)
    if estimate is None:
        return Answer(point)
    band, mean = estimate
    return Answer(point, band, mean, draws)


def marginal(  # noqa: PLR0913 - the options after the query itself are keyword-only
    network: Network,
    node: str,
    value: bool = True,
    *,
    engine: Engine | None = None,
    draws: int = DEFAULT_DRAWS,
    rng: Seed = None,
) -> Answer:
    """Compute ``P(node = value)``.

    Args:
        network: The network.
        node: A node id.
        value: The value asked about.
        engine: The inference engine; variable elimination by default.
        draws: The number of parameter draws behind the band; 0 skips the band.
        rng: A random generator, or a seed for one, for the draws; ``None`` draws fresh entropy.

    Returns:
        The answer.
    """
    return _run(network, Query({node: value}), engine, draws, rng)


def joint(
    network: Network,
    assignment: Mapping[str, bool] | Iterable[str],
    *,
    engine: Engine | None = None,
    draws: int = DEFAULT_DRAWS,
    rng: Seed = None,
) -> Answer:
    """Compute the probability of a joint assignment, ``P(X = x and Y = y and ...)``.

    Args:
        network: The network.
        assignment: Node ids and values, or an iterable of node ids that are all asked to be true.
        engine: The inference engine; variable elimination by default.
        draws: The number of parameter draws behind the band; 0 skips the band.
        rng: A random generator, or a seed for one, for the draws; ``None`` draws fresh entropy.

    Returns:
        The answer.
    """
    return _run(network, Query(_as_assignment(assignment, "assignment")), engine, draws, rng)


def conditional(  # noqa: PLR0913 - the options after the query itself are keyword-only
    network: Network,
    target: str | Mapping[str, bool],
    given: Mapping[str, bool],
    *,
    engine: Engine | None = None,
    draws: int = DEFAULT_DRAWS,
    rng: Seed = None,
) -> Answer:
    """Compute ``P(target | given)``.

    Conditioning on a node also updates the node's own premises: observing that a conclusion is
    false is evidence against what supports it.

    Args:
        network: The network.
        target: A node id, asked to be true, or an assignment.
        given: The observed values.
        engine: The inference engine; variable elimination by default.
        draws: The number of parameter draws behind the band; 0 skips the band.
        rng: A random generator, or a seed for one, for the draws; ``None`` draws fresh entropy.

    Returns:
        The answer.
    """
    return _run(network, Query(_as_assignment(target, "target"), given), engine, draws, rng)


def intervene(  # noqa: PLR0913 - the options after the query itself are keyword-only
    network: Network,
    target: str | Mapping[str, bool],
    set: Mapping[str, bool],  # noqa: A002 - the documented keyword: intervene(X, set={Y: False})
    *,
    given: Mapping[str, bool] | None = None,
    engine: Engine | None = None,
    draws: int = DEFAULT_DRAWS,
    rng: Seed = None,
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
        draws: The number of parameter draws behind the band; 0 skips the band.
        rng: A random generator, or a seed for one, for the draws; ``None`` draws fresh entropy.

    Returns:
        The answer.
    """
    return _run(network, Query(_as_assignment(target, "target"), given or {}, set), engine, draws, rng)
