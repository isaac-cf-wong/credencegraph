"""Exceptions raised when a graph is compiled into a network."""

from __future__ import annotations

from credencegraph.core.errors import CredenceGraphError


class CompileError(CredenceGraphError, ValueError):
    """A graph cannot be compiled into a network.

    Raised for an inference variable without a ``base``, for ``equivalent`` nodes whose bases
    conflict, and for a cycle that only appears once ``equivalent`` nodes are merged.
    """
