"""Text normalisation and the digest."""

from __future__ import annotations

import hashlib

from credencegraph.rubric import normalise_text, text_digest


def test_normalise_collapses_whitespace_and_composes():
    """Test that whitespace runs become one space, the ends are trimmed and the text is NFC."""
    assert normalise_text("  a\n\tb   c \r\n") == "a b c"
    assert normalise_text("café") == "café"


def test_digest_of_known_text():
    """Test the digest against the published SHA-256 test vector for "abc" (FIPS 180-2)."""
    assert text_digest("abc") == "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_digest_is_of_the_normalised_text():
    """Test that texts equal after normalisation share a digest, and the digest is of their UTF-8 bytes."""
    assert text_digest(" café\n bar ") == text_digest("café bar")
    assert text_digest("café bar") == "sha256:" + hashlib.sha256("café bar".encode()).hexdigest()
    assert text_digest("a") != text_digest("b")
