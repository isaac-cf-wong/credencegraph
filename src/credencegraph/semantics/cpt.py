"""Conditional probability tables for the inferential relations.

A proposition X with ``requires`` parents R, ``supports`` parents S and ``refutes`` parents F, of
strengths r, s and f, and base b, is true with probability

    P(X = 1 | parents) = N * O * I

    N = prod over i in R with R_i = 0 of (1 - r_i)          necessity gate (noisy-AND)
    O = 1 - (1 - b) * prod over j in S with S_j = 1 of (1 - s_j)   sufficiency (noisy-OR, leak b)
    I = prod over k in F with F_k = 1 of (1 - f_k)          inhibition

so ``b`` is the probability of X when every required premise holds and no support or refuter is
active. With no parents, ``b`` is the prior.

Every table built here has one axis per parent, in the order given, followed by the child's own
axis; index 0 on an axis means false and 1 means true.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np

from credencegraph.core.relation import REFUTES, REQUIRES, STRENGTH_TYPES, SUPPORTS


@dataclass(frozen=True, slots=True)
class Term:
    """One inferential relation's contribution to a child's table.

    Attributes:
        parent: Position of the parent among the table's parent axes.
        type: ``requires``, ``supports`` or ``refutes``.
        strength: The relation's strength, a probability.
    """

    parent: int
    type: str
    strength: float


def proposition_table(base: float, terms: Sequence[Term], n_parents: int) -> np.ndarray:
    """Build the table of a proposition from its base and inferential relations.

    A parent may appear in several terms, for example when two relations from nodes that were
    merged as ``equivalent`` reach the same child; each term then acts as an independent mechanism.

    Args:
        base: The base probability ``b``.
        terms: The inferential relations into the child.
        n_parents: The number of parent axes.

    Returns:
        An array of shape ``(2,) * (n_parents + 1)`` holding ``P(X = x | parents)``.

    Raises:
        ValueError: If a term names an unknown relation type or a parent axis out of range.
    """
    shape = (2,) * n_parents
    grid = np.indices(shape)
    necessity = np.ones(shape)
    no_support = np.ones(shape)
    inhibition = np.ones(shape)
    for term in terms:
        if term.type not in STRENGTH_TYPES:
            msg = f"unknown inferential relation type {term.type!r}"
            raise ValueError(msg)
        if not 0 <= term.parent < n_parents:
            msg = f"term parent {term.parent} out of range for {n_parents} parents"
            raise ValueError(msg)
        parent = grid[term.parent]
        if term.type == REQUIRES:
            necessity = necessity * np.where(parent == 0, 1.0 - term.strength, 1.0)
        elif term.type == SUPPORTS:
            no_support = no_support * np.where(parent == 1, 1.0 - term.strength, 1.0)
        elif term.type == REFUTES:
            inhibition = inhibition * np.where(parent == 1, 1.0 - term.strength, 1.0)
    # 1 - (1 - b) * P rewritten as b * P + (1 - P): two non-negative terms, so a tiny base is not
    # lost to the cancellation of 1 - (1 - b), and with no active support it is exactly b.
    sufficiency = base * no_support + (1.0 - no_support)
    true = necessity * sufficiency * inhibition
    return np.stack([1.0 - true, true], axis=-1)


def true_probability(base: Real, terms: Sequence[Term], parents: Sequence[int]) -> Real:
    """Evaluate ``P(X = 1 | parents)`` for one assignment of the parents.

    The same formula as ``proposition_table``, one row at a time and in whatever arithmetic the
    numbers bring: given ``Fraction`` values it is exact, which the diagnostics use to decide whether
    a parameter cancels out of a table without any rounding.

    Args:
        base: The base probability ``b``.
        terms: The inferential relations into the child; ``Term.strength`` in the same arithmetic.
        parents: The value, 0 or 1, of each parent axis.

    Returns:
        ``N * O * I`` at that assignment.
    """
    necessity = no_support = inhibition = 1
    for term in terms:
        value = parents[term.parent]
        if term.type == REQUIRES and value == 0:
            necessity *= 1 - term.strength
        elif term.type == SUPPORTS and value == 1:
            no_support *= 1 - term.strength
        elif term.type == REFUTES and value == 1:
            inhibition *= 1 - term.strength
    return necessity * (base * no_support + (1 - no_support)) * inhibition


def exclusion_table(n_parents: int) -> np.ndarray:
    """Build the table of the constraint variable C that encodes "not all of these are true".

    ``P(C = 1 | parents)`` is 0 when every parent is true and 1 otherwise. ``exclusive(A, B)`` uses
    two parents; when A and B were merged as ``equivalent`` the constraint has the single parent A,
    and observing C = 1 then forces A to be false.

    Args:
        n_parents: The number of parent axes, at least 1.

    Returns:
        An array of shape ``(2,) * (n_parents + 1)`` holding ``P(C = c | parents)``.

    Raises:
        ValueError: If ``n_parents`` is below 1.
    """
    if n_parents < 1:
        msg = f"an exclusion constraint needs at least one parent, got {n_parents}"
        raise ValueError(msg)
    true = np.ones((2,) * n_parents)
    true[(1,) * n_parents] = 0.0
    return np.stack([1.0 - true, true], axis=-1)
