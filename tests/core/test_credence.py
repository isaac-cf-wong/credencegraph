"""Tests for the credence types."""

from __future__ import annotations

import math
import sys
from fractions import Fraction

import pytest

from credencegraph.core import Beta, Point, ValidationError, as_credence


class TestPoint:
    """Validation of Point."""

    @pytest.mark.parametrize("p", [0, 1, 0.0, 0.25, 1.0])
    def test_accepts_unit_interval(self, p):
        """Test that both endpoints and interior values are valid."""
        assert Point(p).p == float(p)
        assert isinstance(Point(p).p, float)

    @pytest.mark.parametrize("p", [-0.01, 1.01, math.nan, math.inf, -math.inf])
    def test_rejects_out_of_range_and_non_finite(self, p):
        """Test that values outside [0, 1] and non-finite values are rejected."""
        with pytest.raises(ValidationError, match=r"Point\.p"):
            Point(p)

    @pytest.mark.parametrize("p", [True, False, "0.5", None, [0.5]])
    def test_rejects_non_numbers(self, p):
        """Test that booleans and non-numeric types are not read as numbers."""
        with pytest.raises(ValidationError, match="real number"):
            Point(p)

    def test_moments(self):
        """Test the mean and variance of a point."""
        assert Point(0.3).mean == 0.3
        assert Point(0.3).variance == 0.0


class TestBeta:
    """Validation and moments of Beta."""

    @pytest.mark.parametrize(("alpha", "beta"), [(0, 1), (1, 0), (-1, 2), (2, -1), (0.0, 0.0)])
    def test_rejects_non_positive_parameters(self, alpha, beta):
        """Test that alpha and beta must both be strictly positive."""
        with pytest.raises(ValidationError, match="must be > 0"):
            Beta(alpha, beta)

    @pytest.mark.parametrize(("alpha", "beta"), [(math.nan, 1), (1, math.inf), (True, 1), ("1", 1)])
    def test_rejects_non_finite_or_non_numeric(self, alpha, beta):
        """Test that NaN, infinity, booleans and strings are rejected."""
        with pytest.raises(ValidationError):
            Beta(alpha, beta)

    def test_moments(self):
        """Test mean, concentration and variance against the closed forms."""
        b = Beta(2, 6)
        assert b.concentration == 8.0
        assert b.mean == 0.25
        assert b.variance == pytest.approx(2 * 6 / (8**2 * 9), rel=1e-15)

    @pytest.mark.parametrize(
        ("alpha", "beta"),
        [
            (sys.float_info.max, sys.float_info.max),
            (sys.float_info.max, 1e308),
            (1e308, sys.float_info.max),
            (1.5 * 2.0**1023, 0.5 * 2.0**1023),
        ],
    )
    def test_mean_when_concentration_overflows(self, alpha, beta):
        """Test that the mean stays exact when ``alpha + beta`` overflows to infinity."""
        b = Beta(alpha, beta)
        assert math.isinf(alpha + beta)
        exact = float(Fraction(alpha) / (Fraction(alpha) + Fraction(beta)))
        assert b.mean == pytest.approx(exact, rel=1e-15, abs=0.0)
        assert 0.0 < b.mean < 1.0

    def test_from_mean_concentration(self):
        """Test the mean/concentration constructor."""
        b = Beta.from_mean_concentration(0.8, 10)
        assert (b.alpha, b.beta) == (8.0, pytest.approx(2.0))
        assert b.mean == pytest.approx(0.8, rel=1e-15)

    @pytest.mark.parametrize(("mean", "conc"), [(0, 5), (1, 5), (-0.1, 5), (1.1, 5), (0.5, 0), (0.5, -1)])
    def test_from_mean_concentration_rejects_out_of_range(self, mean, conc):
        """Test that the mean must be in (0, 1) and the concentration positive."""
        with pytest.raises(ValidationError, match=r"^(mean must lie strictly|concentration must be)"):
            Beta.from_mean_concentration(mean, conc)

    @pytest.mark.parametrize(("mean", "sd"), [(0.5, 0.1), (0.9, 0.05), (0.1, 0.2), (0.5, 0.49)])
    def test_from_mean_sd_recovers_moments(self, mean, sd):
        """Test that the mean/sd constructor hits the requested moments."""
        b = Beta.from_mean_sd(mean, sd)
        assert b.mean == pytest.approx(mean, rel=1e-12)
        assert math.sqrt(b.variance) == pytest.approx(sd, rel=1e-9)

    def test_from_mean_sd_known_value(self):
        """Test against a hand computation: mean 0.5, sd 0.25 is the uniform Beta(1, 1) ... only at sd^2 = 1/12."""
        b = Beta.from_mean_sd(0.5, math.sqrt(1 / 12))
        assert b.alpha == pytest.approx(1.0, rel=1e-12)
        assert b.beta == pytest.approx(1.0, rel=1e-12)

    @pytest.mark.parametrize(("mean", "sd"), [(0, 0.1), (1, 0.1), (0.5, 0), (0.5, -0.1)])
    def test_from_mean_sd_rejects_out_of_range(self, mean, sd):
        """Test that the mean must be in (0, 1) and the sd positive."""
        with pytest.raises(ValidationError, match=r"^(mean must lie strictly|sd must be > 0)"):
            Beta.from_mean_sd(mean, sd)

    @pytest.mark.parametrize("sd", [0.5, 0.6])
    def test_from_mean_sd_rejects_unattainable_spread(self, sd):
        """Test that sd at or above sqrt(m(1-m)) is rejected: no Beta has that spread."""
        with pytest.raises(ValidationError, match="below sqrt"):
            Beta.from_mean_sd(0.5, sd)


class TestAsCredence:
    """The float shortcut."""

    def test_float_is_point(self):
        """Test that a bare float becomes a Point."""
        assert as_credence(0.4) == Point(0.4)
        assert as_credence(1) == Point(1.0)

    def test_credences_pass_through(self):
        """Test that Beta and Point are returned unchanged."""
        b = Beta(1, 2)
        assert as_credence(b) is b
        p = Point(0.5)
        assert as_credence(p) is p

    @pytest.mark.parametrize("value", [1.5, True, "0.5"])
    def test_rejects_bad_values(self, value):
        """Test that out-of-range floats, booleans and strings are rejected."""
        with pytest.raises(ValidationError):
            as_credence(value)
