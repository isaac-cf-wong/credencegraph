"""The reference engine: sum the joint distribution over every world."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from credencegraph.inference.engine import Engine
from credencegraph.inference.errors import ProblemTooLargeError
from credencegraph.semantics.network import Network

DEFAULT_MAX_FREE_VARIABLES = 20


class Enumeration(Engine):
    """Exhaustive enumeration of every world consistent with the assignment.

    Each world's probability is the product of one entry from every table, and the answer is the sum
    over worlds. It shares no code with variable elimination, which makes it the independent
    reference that engine is tested against. The cost doubles with every variable left free.
    """

    def __init__(self, max_free_variables: int = DEFAULT_MAX_FREE_VARIABLES) -> None:
        """Configure the engine.

        Args:
            max_free_variables: The largest number of summed-out variables allowed; a query needing
                more raises ``ProblemTooLargeError``.
        """
        self.max_free_variables = max_free_variables

    def probability(self, network: Network, assignment: Mapping[int, int]) -> float:
        """Sum the joint probability of every world that agrees with ``assignment``.

        Args:
            network: The network.
            assignment: Variable indices and values (0 or 1).

        Returns:
            The probability.

        Raises:
            ProblemTooLargeError: If more than ``max_free_variables`` variables would be summed out.
        """
        n = len(network)
        free = [index for index in range(n) if index not in assignment]
        if len(free) > self.max_free_variables:
            msg = (
                f"enumeration would sum over {len(free)} free variables, above the limit of "
                f"{self.max_free_variables} (max_free_variables); use variable elimination"
            )
            raise ProblemTooLargeError(msg, len(free), self.max_free_variables)
        worlds = np.zeros((2 ** len(free), n), dtype=np.intp)
        codes = np.arange(2 ** len(free))
        for bit, index in enumerate(free):
            worlds[:, index] = (codes >> bit) & 1
        for index, value in assignment.items():
            worlds[:, index] = value
        weight = np.ones(len(worlds))
        for factor in network.factors:
            weight = weight * factor.table[tuple(worlds[:, index] for index in factor.scope)]
        return float(weight.sum())
