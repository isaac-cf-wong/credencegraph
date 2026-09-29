"""Anchors tying a node back to the text it came from."""

from __future__ import annotations

from dataclasses import dataclass

from credencegraph.core.errors import ValidationError


def require_text(value: object, name: str, *, optional: bool = False) -> None:
    """Check that ``value`` is a non-empty string (or ``None`` when optional).

    Args:
        value: The candidate.
        name: Description of the value, used in the error message.
        optional: Whether ``None`` is acceptable.

    Raises:
        ValidationError: If the value is neither ``None`` (when optional) nor a non-empty string.
    """
    if value is None and optional:
        return
    if not isinstance(value, str) or not value:
        msg = f"{name} must be a non-empty string{' or None' if optional else ''}, got {value!r}"
        raise ValidationError(msg)


@dataclass(frozen=True, slots=True)
class SourceAnchor:
    """Where a proposition came from.

    Attributes:
        document: Caller-chosen identifier of the source (a DOI, a path, a URL).
        locator: A page, section or character span within the document.
        quote: The anchored text.
        digest: A hash of the anchored text, for detecting later changes to the source.
    """

    document: str
    locator: str | None = None
    quote: str | None = None
    digest: str | None = None

    def __post_init__(self) -> None:
        """Validate the fields."""
        require_text(self.document, "SourceAnchor.document")
        require_text(self.locator, "SourceAnchor.locator", optional=True)
        require_text(self.quote, "SourceAnchor.quote", optional=True)
        require_text(self.digest, "SourceAnchor.digest", optional=True)
