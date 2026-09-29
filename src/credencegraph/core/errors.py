"""Exceptions raised by the core data model."""

from __future__ import annotations


class CredenceGraphError(Exception):
    """Base class for every error raised by credencegraph."""


class ValidationError(CredenceGraphError, ValueError):
    """A value, node, relation or document breaks a rule of the data model."""


class CycleError(ValidationError):
    """A relation would close a cycle among ``requires``, ``supports`` and ``refutes`` relations.

    Attributes:
        cycle: The node ids around the cycle, starting and ending at the same node.
        relations: The ids of the relations that form the cycle, in the order they are traversed.
    """

    def __init__(self, message: str, cycle: tuple[str, ...], relations: tuple[str, ...]) -> None:
        """Store the cycle alongside the message.

        Args:
            message: Human-readable description naming the cycle.
            cycle: Node ids around the cycle, first equal to last.
            relations: Relation ids along the cycle.
        """
        super().__init__(message)
        self.cycle = cycle
        self.relations = relations
