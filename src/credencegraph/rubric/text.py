"""Text normalisation and the digest of an anchored text.

These are the only implementation: whatever stores a verbatim text normalises it with
``normalise_text``, and every check recomputes digests with ``text_digest``.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

DIGEST_PREFIX = "sha256:"

_WHITESPACE = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    """Normalise a text for storage: Unicode NFC, whitespace runs collapsed to one space, ends trimmed.

    Args:
        text: The text as read from the source.

    Returns:
        The normalised text.
    """
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def text_digest(text: str) -> str:
    """Return the digest of a text: ``sha256:`` and the hex SHA-256 of its normalised UTF-8 bytes.

    Args:
        text: The text.

    Returns:
        The digest, such as ``sha256:9f86d0...``.
    """
    return DIGEST_PREFIX + hashlib.sha256(normalise_text(text).encode("utf-8")).hexdigest()
