"""Exceptions raised by the inference engines."""

from __future__ import annotations

from credencegraph.core.errors import CredenceGraphError


class InferenceError(CredenceGraphError):
    """Base class for errors raised while answering a query."""


class ZeroProbabilityError(InferenceError, ValueError):
    """The evidence of a query, together with the ``exclusive`` constraints, has probability zero."""


class ProblemTooLargeError(InferenceError):
    """An exact engine would need more memory than its configured limit allows.

    Exact engines never fall back to an approximation on their own: they raise this error instead.

    Attributes:
        required: The size the query would need, in the unit ``limit`` is expressed in.
        limit: The configured limit.
    """

    def __init__(self, message: str, required: int, limit: int) -> None:
        """Store the sizes alongside the message.

        Args:
            message: Human-readable description naming the limit.
            required: The size the query would need.
            limit: The configured limit.
        """
        super().__init__(message)
        self.required = required
        self.limit = limit
