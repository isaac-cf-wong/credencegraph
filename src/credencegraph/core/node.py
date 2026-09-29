"""Graph nodes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from credencegraph.core.anchor import SourceAnchor, require_text
from credencegraph.core.attributes import JSON, copy_attributes
from credencegraph.core.credence import Credence, as_credence
from credencegraph.core.errors import ValidationError

DEFAULT_KIND = "proposition"


@dataclass(frozen=True, slots=True)
class Node:
    """A node of a graph: usually a proposition, but any carried entity.

    ``base`` and ``stated`` accept a bare float, read as a ``Point``. ``sources`` accepts any iterable
    of anchors. ``attributes`` is deep-copied on construction.

    Attributes:
        id: Stable identifier, unique within the graph.
        kind: Free label the engine attaches no meaning to.
        statement: The normalised proposition.
        sources: Where the proposition came from.
        base: For a node with no inferential parents, its prior credence.
        stated: The credence the source itself asserts, used only by diagnostics.
        attributes: Open JSON metadata.
    """

    id: str
    kind: str = DEFAULT_KIND
    statement: str | None = None
    sources: tuple[SourceAnchor, ...] = ()
    base: Credence | None = None
    stated: Credence | None = None
    attributes: dict[str, JSON] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate every field and normalise the flexible ones."""
        require_text(self.id, "Node.id")
        require_text(self.kind, "Node.kind")
        require_text(self.statement, f"Node {self.id!r} statement", optional=True)
        sources = self.sources
        if isinstance(sources, str | bytes) or not isinstance(sources, Iterable):
            msg = f"Node {self.id!r} sources must be a sequence of SourceAnchor, got {sources!r}"
            raise ValidationError(msg)
        sources = tuple(sources)
        for anchor in sources:
            if not isinstance(anchor, SourceAnchor):
                msg = f"Node {self.id!r} sources must hold SourceAnchor objects, got {anchor!r}"
                raise ValidationError(msg)
        object.__setattr__(self, "sources", sources)
        for name in ("base", "stated"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, as_credence(value))
        object.__setattr__(self, "attributes", copy_attributes(self.attributes, f"Node {self.id!r} attributes"))
