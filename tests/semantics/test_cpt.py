"""Tests for the conditional probability tables."""

from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np
import pytest

from credencegraph.semantics import Term, exclusion_table, proposition_table
from credencegraph.semantics.cpt import true_probability


def close(actual, expected):
    """Assert agreement to 1e-12 relative, with no absolute slack."""
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)


class TestPropositionTable:
    """Entries of N * O * I, checked cell by cell against hand-computed values."""

    def test_no_parents_is_the_base(self):
        """Test that a root's table is (1 - b, b)."""
        close(proposition_table(0.3, [], 0), [0.7, 0.3])

    def test_requires(self):
        """Test that a false required premise multiplies by 1 - r and a true one leaves b."""
        table = proposition_table(0.8, [Term(0, "requires", 0.75)], 1)
        close(table[0, 1], 0.8 * 0.25)
        close(table[1, 1], 0.8)

    def test_supports(self):
        """Test that an active support gives 1 - (1 - b)(1 - s) and an inactive one leaves b."""
        table = proposition_table(0.2, [Term(0, "supports", 0.5)], 1)
        close(table[0, 1], 0.2)
        close(table[1, 1], 1 - 0.8 * 0.5)

    def test_refutes(self):
        """Test that an active refuter multiplies by 1 - f."""
        table = proposition_table(0.6, [Term(0, "refutes", 0.9)], 1)
        close(table[0, 1], 0.6)
        close(table[1, 1], 0.6 * 0.1)

    def test_mixed_cells(self):
        """Test every cell of a table with one relation of each type."""
        b, r, s, f = 0.3, 0.9, 0.5, 0.4
        table = proposition_table(b, [Term(0, "requires", r), Term(1, "supports", s), Term(2, "refutes", f)], 3)
        assert table.shape == (2, 2, 2, 2)
        for req in (0, 1):
            for sup in (0, 1):
                for ref in (0, 1):
                    n = 1.0 if req else 1 - r
                    o = 1 - (1 - b) * ((1 - s) if sup else 1.0)
                    i = (1 - f) if ref else 1.0
                    close(table[req, sup, ref, 1], n * o * i)
                    close(table[req, sup, ref, 0], 1 - n * o * i)

    def test_repeated_parent_combines(self):
        """Test that two supports from one parent act as independent mechanisms."""
        table = proposition_table(0.1, [Term(0, "supports", 0.5), Term(0, "supports", 0.2)], 1)
        close(table[1, 1], 1 - 0.9 * 0.5 * 0.8)

    def test_tiny_base_is_kept(self):
        """Test that a base of 1e-15 survives to 1e-12 relative wherever support is inactive.

        Formed as 1 - (1 - b), a base of 1e-15 comes back as 9.992e-16, off by 8e-4 relative.
        """
        table = proposition_table(1e-15, [Term(0, "requires", 1.0), Term(1, "supports", 0.5)], 2)
        close(table[1, 0, 1], 1e-15)
        close(table[1, 1, 1], 0.5 + 0.5e-15)
        assert table[0, 0, 1] == 0.0
        close(proposition_table(1e-15, [], 0)[1], 1e-15)

    def test_rows_sum_to_one(self):
        """Test that each parent configuration gives a distribution."""
        table = proposition_table(0.4, [Term(0, "supports", 0.3), Term(1, "refutes", 0.6)], 2)
        close(table.sum(axis=-1), np.ones((2, 2)))

    def test_rejects_bad_terms(self):
        """Test that unknown types and out-of-range parents are rejected."""
        with pytest.raises(ValueError, match="unknown inferential relation type 'cites'"):
            proposition_table(0.5, [Term(0, "cites", 0.5)], 1)
        with pytest.raises(ValueError, match="out of range"):
            proposition_table(0.5, [Term(1, "supports", 0.5)], 1)


class TestTrueProbability:
    """The one-row form of the proposition table, which the diagnostics evaluate exactly."""

    TERMS = (
        Term(0, "requires", 0.75),
        Term(1, "supports", 0.5),
        Term(2, "refutes", 0.9),
        Term(1, "supports", 0.2),
        Term(0, "requires", 1.0),
    )

    def test_matches_the_table(self):
        """Test that every row agrees with ``proposition_table``, so the two forms cannot drift."""
        table = proposition_table(0.3, self.TERMS, 3)
        for parents in itertools.product((0, 1), repeat=3):
            close(true_probability(0.3, self.TERMS, parents), table[(*parents, 1)])

    def test_is_exact_on_fractions(self):
        """Test that rational inputs give the exact rational entry, with no rounding."""
        terms = [
            Term(0, "requires", Fraction(3, 4)),
            Term(1, "supports", Fraction(1, 2)),
            Term(2, "refutes", Fraction(9, 10)),
            Term(1, "supports", Fraction(1, 5)),
        ]
        # N = 1 - 3/4, O = 1 - (1 - 3/10)(1 - 1/2)(1 - 1/5), I = 1 - 9/10 at A = 0, S = 1, F = 1.
        expected = Fraction(1, 4) * (1 - Fraction(7, 10) * Fraction(1, 2) * Fraction(4, 5)) * Fraction(1, 10)
        assert true_probability(Fraction(3, 10), terms, (0, 1, 1)) == expected


class TestExclusionTable:
    """The constraint "not all of these are true"."""

    def test_two_parents(self):
        """Test that C is false only when both parents are true."""
        table = exclusion_table(2)
        np.testing.assert_array_equal(table[..., 1], [[1.0, 1.0], [1.0, 0.0]])
        np.testing.assert_array_equal(table[..., 0], [[0.0, 0.0], [0.0, 1.0]])

    def test_one_parent(self):
        """Test that with a single parent, C = 1 forces the parent false."""
        np.testing.assert_array_equal(exclusion_table(1)[:, 1], [1.0, 0.0])

    def test_needs_a_parent(self):
        """Test that a constraint without parents is rejected."""
        with pytest.raises(ValueError, match="at least one parent"):
            exclusion_table(0)
