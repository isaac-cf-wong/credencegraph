"""The sentence-boundary rule the readers share."""

from __future__ import annotations

import time

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
        ("the mean drift is 0.6 K", "Lee reports", True),
        ("a supply of 5 V", "The current", True),
        ("a force of -3 N", "It acts", True),
        ("a gain of +2.5 A", "The loss", True),
        ("a load of 1,000 W", "The load", True),
        ("the mean drift is $0.6$ K", "Lee reports", True),
        (r"the mean drift is \(0.6\) K", "Lee reports", True),
        (r"a rate of $1.2 \times 10^{-3}$ K", "Lee reports", True),
        (r"a load of $1\,000$ W", "The load", True),
        (r"a rate of \(1.2 \times 10^{-3}\) K", "Lee reports", True),
        (r"a value of $x + 1$ K", "Lee reports", False),
        (r"the price is \$5; the drift is $0.6$ K", "Lee reports", True),
        (r"we report about$0.6$ K", "Lee reports", False),
        (r"the value is \\$0.6$ K", "Lee reports", True),
        (r"the price is \\\$5 and the drift $0.6$ K", "Lee reports", True),
        ("reported by C", "Lee, the drift", False),
        ("as told by J. R. R", "Tolkien, the tale", False),
        (r"as told by J.\,R.~R", "Tolkien, the tale", False),
        (r"the mean drift is 0.6\,K", "Lee reports", True),
        (r"listed in Table\,I", "The records", True),
        (r"the mean drift is 0.6\quad K", "Lee reports", True),
        (r"the mean drift is 0.6\thinspace K", "Lee reports", True),
        (r"a supply of 5\enspace V", "The current", True),
        (r"a gain of 2\nobreakspace A", "The loss", True),
        (r"as told by J.\thinspace R.\enspace R", "Tolkien, the tale", False),
        (r"the mean drift is 0.6\quadrant K", "Lee reports", False),
    ],
)
def test_ends_sentence(before, after, expected):
    """Test abbreviations, initials, a lower-case continuation, and a capital that labels or is a unit.

    A single capital after a word such as "Table" labels it, and one after a numeral is a unit, as in
    "0.6 K."; both end the sentence, where a capital after a name's other parts is an initial.
    """
    assert ends_sentence(before, after) is expected


def test_ends_sentence_reads_unclosed_math_openers_in_linear_time():
    """Test that a text of many unclosed inline-math openers is read at once, not rescanned per opener."""
    started = time.perf_counter()
    assert ends_sentence("\\(" * 100_000 + " 0.6 K", "Lee reports") is True
    assert time.perf_counter() - started < 5
