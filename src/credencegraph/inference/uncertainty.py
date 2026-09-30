"""Parameter uncertainty: how far an answer moves when the parameters are drawn from their credences.

Every parameter (each node's ``base`` and each relation's ``strength``) is treated as independent of
the others. The point answer of a query needs no sampling: the joint probabilities of the network
are multilinear in independent parameters, so the predictive answer ``E[P(A, E)] / E[P(E)]`` equals
``P(A | E)`` with every parameter at its mean.

The spread is found by Monte Carlo. Each ``Beta`` parameter is drawn independently, the query is
answered exactly at every draw, and the 5%, 50% and 95% quantiles of those answers form the band.
The band says how much the answer would move if the inputs were different; it does not update the
parameters on the evidence, so the average of the per-draw answers, ``mean_over_draws``, generally
differs from the point answer. Both are reported, under their own names.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from credencegraph.core.credence import Beta
from credencegraph.core.errors import ValidationError
from credencegraph.inference.engine import Engine, Query
from credencegraph.inference.errors import ZeroProbabilityError
from credencegraph.semantics.network import Network, ParameterKey

DEFAULT_DRAWS = 1000

# Draws are kept inside the open unit interval. A Beta with a small shape parameter can round to
# exactly 0 or 1 in floating point, which would switch off a table entry that is positive at the
# mean and could leave evidence with probability zero in that draw only.
_LOWEST = float(np.finfo(float).tiny)
_HIGHEST = float(np.nextafter(1.0, 0.0))


@dataclass(frozen=True, slots=True)
class Band:
    """Quantiles of a query's answer over parameter draws.

    Attributes:
        q05: The 5% quantile.
        q50: The median.
        q95: The 95% quantile.
    """

    q05: float
    q50: float
    q95: float

    def to_dict(self) -> dict[str, float]:
        """Return the band as JSON-ready data.

        Returns:
            ``{"q05": ..., "q50": ..., "q95": ...}``.
        """
        return {"q05": self.q05, "q50": self.q50, "q95": self.q95}


def has_spread(network: Network) -> bool:
    """Tell whether any parameter of the network is uncertain.

    Args:
        network: The network.

    Returns:
        ``True`` if at least one parameter has a ``Beta`` credence, ``False`` if every one is a ``Point``.
    """
    return any(isinstance(credence, Beta) for credence in network.parameters.values())


def _draw_count(draws: object) -> int:
    """Validate a number of draws.

    Args:
        draws: The candidate number.

    Returns:
        The number of draws.

    Raises:
        ValidationError: If ``draws`` is not a non-negative integer.
    """
    if isinstance(draws, bool) or not isinstance(draws, int | np.integer) or draws < 0:
        msg = f"draws must be a non-negative integer, got {draws!r}"
        raise ValidationError(msg)
    return int(draws)


def sample_parameters(
    network: Network, draws: int, rng: np.random.Generator | int | None = None
) -> dict[ParameterKey, np.ndarray]:
    """Draw every ``Beta`` parameter of a network independently from its credence.

    Parameters are drawn in the order ``network.parameters`` lists them, so a given seed always
    yields the same draws for the same network. ``Point`` parameters are left out: they keep their
    value in every draw.

    Args:
        network: The network.
        draws: The number of draws.
        rng: A random generator, or a seed for one; ``None`` draws fresh entropy.

    Returns:
        For each ``Beta`` parameter, an array of ``draws`` values inside the open unit interval.

    Raises:
        ValidationError: If ``draws`` is not a non-negative integer.
    """
    count = _draw_count(draws)
    generator = np.random.default_rng(rng)
    return {
        key: np.clip(generator.beta(credence.alpha, credence.beta, size=count), _LOWEST, _HIGHEST)
        for key, credence in network.parameters.items()
        if isinstance(credence, Beta)
    }


def answers_over_draws(
    network: Network, query: Query, engine: Engine, samples: Mapping[ParameterKey, np.ndarray]
) -> np.ndarray:
    """Answer a query exactly at every parameter draw.

    Args:
        network: The network.
        query: The query.
        engine: The exact engine used at each draw.
        samples: Values by parameter, one array entry per draw, as from ``sample_parameters``.

    Returns:
        The answer at each draw.

    Raises:
        ValidationError: If the sample arrays differ in length, or a key or value is invalid.
        ZeroProbabilityError: If the evidence has probability zero at some draw.
    """
    lengths = {len(values) for values in samples.values()}
    if len(lengths) > 1:
        msg = f"every parameter needs the same number of draws, got lengths {sorted(lengths)}"
        raise ValidationError(msg)
    count = lengths.pop() if lengths else 0
    answers = np.empty(count)
    for draw in range(count):
        drawn = network.with_parameters({key: float(values[draw]) for key, values in samples.items()})
        try:
            answers[draw] = engine.query(drawn, query)
        except ZeroProbabilityError as error:
            msg = f"parameter draw {draw}: {error}"
            raise ZeroProbabilityError(msg) from error
    return answers


def spread(
    network: Network,
    query: Query,
    engine: Engine,
    draws: int = DEFAULT_DRAWS,
    rng: np.random.Generator | int | None = None,
) -> tuple[Band, float] | None:
    """Estimate how a query's answer spreads over the parameters' credences.

    Every ``Beta`` parameter is drawn from its credence, including one that ``Network.with_parameters``
    has set to another value: the band describes the credences, not the values the network holds.

    Args:
        network: The network.
        query: The query.
        engine: The exact engine used at each draw.
        draws: The number of Monte Carlo draws; 0 skips the estimate.
        rng: A random generator, or a seed for one; ``None`` draws fresh entropy.

    Returns:
        The band and the mean of the per-draw answers, or ``None`` when ``draws`` is 0 or every
        parameter is a ``Point``, since the answer then cannot move.

    Raises:
        ValidationError: If ``draws`` is not a non-negative integer.
        ZeroProbabilityError: If the evidence has probability zero at some draw.
    """
    count = _draw_count(draws)
    if count == 0 or not has_spread(network):
        return None
    answers = answers_over_draws(network, query, engine, sample_parameters(network, count, rng))
    q05, q50, q95 = (float(q) for q in np.quantile(answers, (0.05, 0.5, 0.95)))
    return Band(q05, q50, q95), float(answers.mean())
