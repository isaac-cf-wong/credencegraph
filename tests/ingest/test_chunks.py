"""The sentence-boundary rule the readers share."""

from __future__ import annotations

import pytest

from credencegraph.ingest import ends_sentence


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        ("The bias vanishes", "It is small.", True),
        ("The bias vanishes", "and it is small.", False),
        ("as in Eq", "(3) the bias", False),
        ("as shown by J", "Doe, the bias", False),
        ("as shown by", "", True),
        ("listed in Table I", "The records", True),
        ("listed in Sec. V", "The records", True),
        ("written by A. N", "Author", False),
    ],
)
def test_ends_sentence(before, after, expected):
    """Test abbreviations, initials, a lower-case continuation, and a capital that labels a table or section."""
    assert ends_sentence(before, after) is expected
