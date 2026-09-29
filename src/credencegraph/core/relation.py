"""Relations between nodes."""

from __future__ import annotations

from dataclasses import dataclass, field

from credencegraph.core.anchor import require_text
from credencegraph.core.attributes import JSON, copy_attributes
from credencegraph.core.credence import Credence, as_credence
from credencegraph.core.errors import ValidationError

REQUIRES = "requires"
SUPPORTS = "supports"
REFUTES = "refutes"
EQUIVALENT = "equivalent"
EXCLUSIVE = "exclusive"

# Inferential types that carry a strength.
STRENGTH_TYPES = frozenset({REQUIRES, SUPPORTS, REFUTES})

# The fixed set of inferential relation types. Any other type string is an annotation.
INFERENTIAL_TYPES = STRENGTH_TYPES | {EQUIVALENT, EXCLUSIVE}

# Types whose subgraph must stay acyclic.
ACYCLIC_TYPES = STRENGTH_TYPES


def is_inferential(relation_type: str) -> bool:
    """Tell whether ``relation_type`` is inferential rather than an annotation.

    Args:
        relation_type: The relation type string.

    Returns:
        ``True`` for ``requires``, ``supports``, ``refutes``, ``equivalent`` and ``exclusive``.
    """
    return relation_type in INFERENTIAL_TYPES


@dataclass(frozen=True, slots=True)
class Relation:
    """A directed relation between two nodes.

    ``strength`` is required for ``requires``, ``supports`` and ``refutes``, and forbidden for
    every other type, including ``equivalent``, ``exclusive`` and all annotations. A bare float is read
    as a ``Point``. ``equivalent`` and ``exclusive`` relations may not join a node to itself.

    Attributes:
        id: Stable identifier, unique within the graph.
        type: An inferential type or any other string, which makes the relation an annotation.
        source: Id of the source node.
        target: Id of the target node.
        strength: The relation's strength, per the rules above.
        attributes: Open JSON metadata.
    """

    id: str
    type: str
    source: str
    target: str
    strength: Credence | None = None
    attributes: dict[str, JSON] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the fields and enforce the strength rules."""
        require_text(self.id, "Relation.id")
        require_text(self.type, f"Relation {self.id!r} type")
        require_text(self.source, f"Relation {self.id!r} source")
        require_text(self.target, f"Relation {self.id!r} target")
        if self.type in STRENGTH_TYPES:
            if self.strength is None:
                msg = f"Relation {self.id!r} of type {self.type!r} requires a strength"
                raise ValidationError(msg)
            object.__setattr__(self, "strength", as_credence(self.strength))
        elif self.strength is not None:
            kind = "an inferential relation that carries no strength" if is_inferential(self.type) else "an annotation"
            msg = f"Relation {self.id!r} of type {self.type!r} is {kind}; it must not have a strength"
            raise ValidationError(msg)
        if self.type in (EQUIVALENT, EXCLUSIVE) and self.source == self.target:
            msg = f"Relation {self.id!r} of type {self.type!r} joins node {self.source!r} to itself"
            raise ValidationError(msg)
        object.__setattr__(self, "attributes", copy_attributes(self.attributes, f"Relation {self.id!r} attributes"))

    @property
    def is_inferential(self) -> bool:
        """Whether the relation is inferential rather than an annotation."""
        return is_inferential(self.type)
