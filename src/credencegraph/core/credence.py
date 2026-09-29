"""Credences: a Beta distribution over a probability, or a single point value."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Self

from credencegraph.core.errors import ValidationError


def _real(value: object, name: str) -> float:
    """Return ``value`` as a finite float, rejecting booleans and non-numbers.

    Args:
        value: The candidate number.
        name: Description of the value, used in the error message.

    Returns:
        The value as a float.

    Raises:
        ValidationError: If the value is not a finite real number.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        msg = f"{name} must be a real number, got {value!r}"
        raise ValidationError(msg)
    number = float(value)
    if not math.isfinite(number):
        msg = f"{name} must be finite, got {value!r}"
        raise ValidationError(msg)
    return number


@dataclass(frozen=True, slots=True)
class Point:
    """A credence with no spread: the probability is exactly ``p``.

    Attributes:
        p: The probability, ``0 <= p <= 1``.
    """

    p: float

    def __post_init__(self) -> None:
        """Validate and normalise ``p`` to a float."""
        p = _real(self.p, "Point.p")
        if not 0.0 <= p <= 1.0:
            msg = f"Point.p must lie in [0, 1], got {p!r}"
            raise ValidationError(msg)
        object.__setattr__(self, "p", p)

    @property
    def mean(self) -> float:
        """The mean, equal to ``p``."""
        return self.p

    @property
    def variance(self) -> float:
        """The variance, always zero."""
        return 0.0


@dataclass(frozen=True, slots=True)
class Beta:
    """A Beta(alpha, beta) distribution over the probability.

    Attributes:
        alpha: First shape parameter, ``alpha > 0``.
        beta: Second shape parameter, ``beta > 0``.
    """

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        """Validate and normalise both shape parameters to floats."""
        alpha = _real(self.alpha, "Beta.alpha")
        beta = _real(self.beta, "Beta.beta")
        if alpha <= 0.0:
            msg = f"Beta.alpha must be > 0, got {alpha!r}"
            raise ValidationError(msg)
        if beta <= 0.0:
            msg = f"Beta.beta must be > 0, got {beta!r}"
            raise ValidationError(msg)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)

    @property
    def concentration(self) -> float:
        """The concentration ``alpha + beta``."""
        return self.alpha + self.beta

    @property
    def mean(self) -> float:
        """The mean ``alpha / (alpha + beta)``."""
        return self.alpha / self.concentration

    @property
    def variance(self) -> float:
        """The variance ``mean * (1 - mean) / (alpha + beta + 1)``."""
        return self.mean * (1.0 - self.mean) / (self.concentration + 1.0)

    @classmethod
    def from_mean_concentration(cls, mean: float, concentration: float) -> Self:
        """Build a Beta from its mean and concentration ``alpha + beta``.

        Args:
            mean: The mean, strictly between 0 and 1.
            concentration: The concentration, greater than 0.

        Returns:
            The Beta with ``alpha = mean * concentration`` and ``beta = (1 - mean) * concentration``.

        Raises:
            ValidationError: If an argument is out of range.
        """
        m = _real(mean, "mean")
        k = _real(concentration, "concentration")
        if not 0.0 < m < 1.0:
            msg = f"mean must lie strictly between 0 and 1, got {m!r}"
            raise ValidationError(msg)
        if k <= 0.0:
            msg = f"concentration must be > 0, got {k!r}"
            raise ValidationError(msg)
        return cls(m * k, (1.0 - m) * k)

    @classmethod
    def from_mean_sd(cls, mean: float, sd: float) -> Self:
        """Build a Beta from its mean and standard deviation.

        Args:
            mean: The mean, strictly between 0 and 1.
            sd: The standard deviation, greater than 0 and below ``sqrt(mean * (1 - mean))``,
                the largest spread any Beta with that mean can have.

        Returns:
            The Beta with the requested mean and standard deviation.

        Raises:
            ValidationError: If an argument is out of range or the pair is unattainable.
        """
        m = _real(mean, "mean")
        s = _real(sd, "sd")
        if not 0.0 < m < 1.0:
            msg = f"mean must lie strictly between 0 and 1, got {m!r}"
            raise ValidationError(msg)
        if s <= 0.0:
            msg = f"sd must be > 0 (use Point for a credence with no spread), got {s!r}"
            raise ValidationError(msg)
        ceiling = m * (1.0 - m)
        if s * s >= ceiling:
            msg = f"sd must be below sqrt(mean * (1 - mean)) = {math.sqrt(ceiling)!r} for mean {m!r}, got {s!r}"
            raise ValidationError(msg)
        return cls.from_mean_concentration(m, ceiling / (s * s) - 1.0)


Credence = Beta | Point
"""A credence: ``Beta`` or ``Point``. Anywhere one is accepted, a bare float ``p`` means ``Point(p)``."""


def as_credence(value: Credence | float) -> Credence:
    """Coerce ``value`` to a credence, reading a bare float as a ``Point``.

    Args:
        value: A ``Beta``, a ``Point`` or a probability.

    Returns:
        The credence.

    Raises:
        ValidationError: If ``value`` is none of those or the probability is out of range.
    """
    if isinstance(value, Beta | Point):
        return value
    return Point(value)
