"""The default exact engine: variable elimination over binary factors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache

import numpy as np

from credencegraph.inference.engine import Engine
from credencegraph.inference.errors import ProblemTooLargeError
from credencegraph.semantics.network import Network

DEFAULT_MAX_FACTOR_SIZE = 2**22

# Elimination orders kept for reuse, keyed by the scopes after evidence is applied. The scopes of a
# network do not depend on its parameter values, so the draws of a band and the parameter overrides
# of a diagnostic all share one entry.
_ORDER_CACHE_SIZE = 256

# One factor: its variable indices and a table with one length-2 axis per variable.
_Factor = tuple[tuple[int, ...], np.ndarray]


def _reduce(scope: tuple[int, ...], table: np.ndarray, assignment: Mapping[int, int]) -> _Factor:
    """Fix the assigned variables of a factor at their values.

    Args:
        scope: The factor's variables.
        table: The factor's table.
        assignment: Variable indices and values.

    Returns:
        The factor over the unassigned variables.
    """
    index = tuple(assignment.get(variable, slice(None)) for variable in scope)
    return tuple(variable for variable in scope if variable not in assignment), table[index]


def min_fill_order(scopes: Iterable[Iterable[int]]) -> list[int]:
    """Choose an elimination order by the greedy min-fill heuristic.

    At each step the variable whose elimination adds the fewest new edges to the interaction graph is
    eliminated next. Ties go to the variable with fewer neighbours, then to the lower index.

    Args:
        scopes: The scopes of the factors; two variables interact when they share a scope.

    Returns:
        Every variable that appears in a scope, in elimination order.
    """
    neighbours: dict[int, set[int]] = {}
    for scope in scopes:
        members = set(scope)
        for variable in members:
            neighbours.setdefault(variable, set()).update(members - {variable})
    order: list[int] = []
    while neighbours:

        def cost(variable: int) -> tuple[int, int, int]:
            around = sorted(neighbours[variable])
            fill = sum(1 for i, a in enumerate(around) for b in around[i + 1 :] if b not in neighbours[a])
            return fill, len(around), variable

        chosen = min(neighbours, key=cost)
        around = neighbours.pop(chosen)
        for variable in around:
            neighbours[variable].discard(chosen)
            neighbours[variable].update(around - {variable})
        order.append(chosen)
    return order


@lru_cache(maxsize=_ORDER_CACHE_SIZE)
def _order(scopes: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    """Return the min-fill order of the given scopes, computed once per distinct scopes.

    Args:
        scopes: The scopes of the factors after the assignment is applied, in factor order.

    Returns:
        The order ``min_fill_order`` gives for these scopes.
    """
    return tuple(min_fill_order(scopes))


def _eliminate(factors: Sequence[_Factor], variable: int) -> tuple[list[_Factor], _Factor]:
    """Multiply the factors that mention ``variable`` and sum it out.

    Args:
        factors: The current factors.
        variable: The variable to eliminate.

    Returns:
        The factors that do not mention it, and the new factor.
    """
    touching = [factor for factor in factors if variable in factor[0]]
    rest = [factor for factor in factors if variable not in factor[0]]
    scope = sorted({v for factor_scope, _ in touching for v in factor_scope})
    # einsum wants small integer labels, so relabel the local variables 0..k-1.
    label = {v: position for position, v in enumerate(scope)}
    operands: list[object] = []
    for factor_scope, table in touching:
        operands += [table, [label[v] for v in factor_scope]]
    kept = tuple(v for v in scope if v != variable)
    table = np.einsum(*operands, [label[v] for v in kept], optimize=False)
    return rest, (kept, table)


class VariableElimination(Engine):
    """Exact inference by variable elimination with a min-fill order.

    The order depends only on the factor scopes once the assignment is applied, so it is computed
    once for each distinct set of scopes and reused by later queries, including those on networks
    that differ only in their parameter values.

    Before any table is built, the engine works out the scope of every intermediate factor the
    order produces. If the largest would hold more than ``max_factor_size`` entries, the query fails
    with ``ProblemTooLargeError``; the engine never falls back to an approximation.
    """

    def __init__(self, max_factor_size: int = DEFAULT_MAX_FACTOR_SIZE) -> None:
        """Configure the engine.

        Args:
            max_factor_size: The largest intermediate factor allowed, in table entries.
        """
        self.max_factor_size = max_factor_size

    def probability(self, network: Network, assignment: Mapping[int, int]) -> float:
        """Compute the probability of ``assignment`` by summing out every other variable.

        Args:
            network: The network.
            assignment: Variable indices and values (0 or 1).

        Returns:
            The probability.

        Raises:
            ProblemTooLargeError: If an intermediate factor would exceed ``max_factor_size`` entries.
        """
        factors = [_reduce(factor.scope, factor.table, assignment) for factor in network.factors]
        order = _order(tuple(scope for scope, _ in factors))
        self._check_size(factors, order)
        for variable in order:
            factors, new = _eliminate(factors, variable)
            factors.append(new)
        result = 1.0
        for _, table in factors:
            result *= float(table)
        return result

    def _check_size(self, factors: Sequence[_Factor], order: Sequence[int]) -> None:
        """Raise if eliminating in ``order`` would build a factor above the limit.

        Args:
            factors: The factors after the assignment is applied.
            order: The elimination order.

        Raises:
            ProblemTooLargeError: If an intermediate factor would exceed ``max_factor_size`` entries.
        """
        scopes = [set(scope) for scope, _ in factors]
        for variable in order:
            touching = [scope for scope in scopes if variable in scope]
            union = set().union(*touching)
            size = 2 ** len(union)
            if size > self.max_factor_size:
                msg = (
                    f"variable elimination would build an intermediate factor over {len(union)} variables "
                    f"({size} entries), above the limit of {self.max_factor_size} entries (max_factor_size). "
                    "Raise max_factor_size to run the query exactly; exact inference does not fall back to "
                    "an approximation on its own"
                )
                raise ProblemTooLargeError(msg, size, self.max_factor_size)
            scopes = [scope for scope in scopes if variable not in scope]
            scopes.append(union - {variable})
