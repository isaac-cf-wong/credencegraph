"""Tests for re-checking discharges and for the sorry report, on synthetic graphs with real scripts."""

from __future__ import annotations

from pathlib import Path

import pytest

from credencegraph.core import Discharge, Graph, Node, Obligation, Relation, SourceAnchor, ValidationError
from credencegraph.core.obligation import file_digest
from credencegraph.diagnostics.obligations import (
    CANNOT_FAIL,
    CHECK_FAILED,
    CONTRADICTED,
    FLAGGED,
    MISSING_SCRIPT,
    NEVER_RUN,
    NO_OBLIGATION,
    NOT_CHECKABLE,
    OUT_OF_TOLERANCE,
    PASSED,
    QUOTE_DIGEST_MISMATCH,
    QUOTE_NOT_FOUND,
    RECORDED_FAILURE,
    RESULT_MISSING,
    SCRIPT_CHANGED,
    TIMED_OUT,
    UNCHECKED,
    UNDISCHARGED,
    can_fail,
    dependencies,
    sorries,
    verify,
)
from credencegraph.rubric.text import text_digest

PASSES = "assert 1 + 1 == 2\n"
FAILS = "import sys\nsys.exit(1)\n"
CANNOT = "print('ok')\n"
# A bias check: it fails when the bias reaches 1%, and prints the bias for verify to hold to 0 ± 0.01.
BIAS = "bias = {bias}\nassert abs(bias) < 0.01\nprint(bias)\n"
# The same bias, printed by a script whose own assert cannot catch it.
LAX = "bias = {bias}\nassert isinstance(bias, float)\nprint(bias)\n"


def script(directory: Path, name: str, text: str) -> str:
    """Write a script and return its name."""
    (directory / name).write_text(text, encoding="utf-8")
    return name


def recorded(directory: Path, name: str, text: str, status: int = 0, kind: str = "symbolic") -> Discharge:
    """Write a script and return a discharge recording a run of exactly that text with ``status``.

    A numerical discharge holds the printed result to 0 ± 0.01.
    """
    script(directory, name, text)
    extra = {"value": 0.0, "tolerance": 0.01} if kind == "numerical" else {}
    return Discharge(kind, script=name, digest=file_digest(text.encode()), status=status, **extra)


def single(discharge: Discharge | None, *, obligation: bool = True) -> Graph:
    """Return ``premise -> claim``, whose one relation ``r`` carries ``discharge``."""
    g = Graph()
    g.add_node(Node("premise", base=0.5))
    g.add_node(Node("claim", base=0.1))
    g.add_relation(
        Relation(
            "r",
            "supports",
            "premise",
            "claim",
            strength=0.8,
            obligation=Obligation("holds", discharge) if obligation else None,
        )
    )
    return g


def diagnostics(check) -> list[str]:
    """Return the diagnostics of a check's findings, in order."""
    return [finding.diagnostic for finding in check.findings]


class TestCanFail:
    """A script can fail when it asserts, raises or may exit with a non-zero status."""

    @pytest.mark.parametrize(
        "source",
        [
            "assert x == 1",
            "raise ValueError('no')",
            "raise SystemExit(1)",
            "import sys\nsys.exit(1)",
            "import sys\nsys.exit(main())",
            "import sys\nsys.exit('bias too large')",
            "import os\nos._exit(2)",
            "exit(3)",
            "quit(status)",
            "if bias >= 0.01:\n    raise SystemExit('bias')",
            "x = 0.5\nassert x < 1",
            "if True:\n    assert x",
            "if False:\n    pass\nelse:\n    assert x",
            "while True:\n    assert x",
            "if x:\n    if False:\n        pass\n    raise ValueError",
            "ok = check() if True else None\nassert ok",
            "def main():\n    assert x\nmain()",
        ],
    )
    def test_failure_paths(self, source):
        """Test that each way to fail is recognised."""
        assert can_fail(source)

    @pytest.mark.parametrize(
        "source",
        [
            "",
            "print('ok')",
            "import sys\nsys.exit()",
            "import sys\nsys.exit(0)",
            "import sys\nsys.exit(None)",
            "exit(False)",
            "def f(:\n    assert x",
            "x = 'assert raise sys.exit(1)'",
            "# assert x\n",
            "if False: assert False",
            "if False:\n    assert False",
            "if 0:\n    raise ValueError",
            "if None:\n    import sys\n    sys.exit(1)",
            "if not True:\n    assert x",
            "if True:\n    pass\nelse:\n    raise ValueError",
            "while False:\n    assert x",
            "x = exit(1) if False else 0",
            "assert True",
            "assert 1, 'never fails'",
            "print('ok')\nif False:\n    if x:\n        raise ValueError",
        ],
    )
    def test_no_failure_path(self, source):
        """Test that a script whose only asserts, raises and non-zero exits are unreachable cannot fail.

        A string and a comment that mention the keywords, which a text search would mistake for a
        failure path, do not count either, nor does a check under a constant-false guard, the dead
        branch of a constant condition, or an assert of a constantly true condition.
        """
        assert not can_fail(source)


class TestVerifyScripts:
    """Each kind of trouble with a script discharge is flagged, and a sound one passes."""

    @pytest.mark.parametrize(("kind", "text"), [("symbolic", PASSES), ("numerical", BIAS.format(bias=0.004))])
    def test_sound_script_passes(self, tmp_path, kind, text):
        """Test that an unchanged script that can fail and still passes is verified."""
        (check,) = verify(single(recorded(tmp_path, "ok.py", text, kind=kind)), tmp_path)
        assert (check.relation, check.kind, check.outcome, check.status, check.findings) == ("r", kind, PASSED, 0, ())

    def test_runs_from_the_script_directory(self, tmp_path):
        """Test that a script runs from its own directory, so it can read the files beside it."""
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "data.txt").write_text("1\n", encoding="utf-8")
        text = "from pathlib import Path\nassert Path('data.txt').read_text() == '1\\n'\n"
        (tmp_path / "sub" / "ok.py").write_text(text, encoding="utf-8")
        discharge = Discharge("symbolic", script="sub/ok.py", digest=file_digest(text.encode()), status=0)
        (check,) = verify(single(discharge), tmp_path)
        assert check.outcome == PASSED

    def test_changed_script(self, tmp_path):
        """Test that a script edited since its recorded run is flagged, though it still passes."""
        discharge = recorded(tmp_path, "eq.py", PASSES)
        script(tmp_path, "eq.py", PASSES + "assert 2 + 2 == 4\n")
        (check,) = verify(single(discharge), tmp_path)
        assert (check.outcome, diagnostics(check), check.status) == (FLAGGED, [SCRIPT_CHANGED], 0)

    def test_script_that_cannot_fail(self, tmp_path):
        """Test that a script with no failure path is flagged, though its run passes."""
        (check,) = verify(single(recorded(tmp_path, "eq.py", CANNOT)), tmp_path)
        assert (check.outcome, diagnostics(check), check.status) == (FLAGGED, [CANNOT_FAIL], 0)

    def test_recorded_pass_now_fails(self, tmp_path):
        """Test that a script whose recorded run passed and which now fails contradicts its record."""
        discharge = Discharge("symbolic", script="eq.py", digest=file_digest(PASSES.encode()), status=0)
        script(tmp_path, "eq.py", PASSES)
        (check,) = verify(single(discharge), tmp_path)
        assert check.outcome == PASSED
        script(tmp_path, "eq.py", FAILS)
        (check,) = verify(single(discharge), tmp_path)
        assert diagnostics(check) == [SCRIPT_CHANGED, CONTRADICTED]
        assert check.findings[1].details == {"status": 1, "recorded": 0}

    def test_recorded_failure_now_passes(self, tmp_path):
        """Test that a script recorded as failing which now passes, unchanged, contradicts its record."""
        (check,) = verify(single(recorded(tmp_path, "eq.py", PASSES, status=1)), tmp_path)
        assert (check.outcome, diagnostics(check), check.status) == (FLAGGED, [CONTRADICTED], 0)

    def test_recorded_failure_still_fails(self, tmp_path):
        """Test that a script recorded as failing which still fails is flagged as a failed check."""
        (check,) = verify(single(recorded(tmp_path, "eq.py", FAILS, status=1)), tmp_path)
        assert (check.outcome, diagnostics(check), check.status) == (FLAGGED, [CHECK_FAILED], 1)

    def test_nonzero_statuses_agree(self, tmp_path):
        """Test that two different non-zero statuses are both failures, so neither contradicts the other."""
        (check,) = verify(single(recorded(tmp_path, "eq.py", "raise SystemExit(3)\n", status=1)), tmp_path)
        assert (diagnostics(check), check.status) == ([CHECK_FAILED], 3)

    def test_never_run(self, tmp_path):
        """Test that a script without a recorded run is flagged, and is run now all the same."""
        script(tmp_path, "eq.py", FAILS)
        (check,) = verify(single(Discharge("symbolic", script="eq.py")), tmp_path)
        assert (check.outcome, diagnostics(check), check.status) == (FLAGGED, [NEVER_RUN, CHECK_FAILED], 1)

    def test_missing_script(self, tmp_path):
        """Test that a script that cannot be read is flagged, and nothing is run."""
        (check,) = verify(single(Discharge("symbolic", script="absent.py")), tmp_path)
        assert (check.outcome, diagnostics(check), check.status) == (FLAGGED, [MISSING_SCRIPT], None)

    def test_timeout(self, tmp_path):
        """Test that a script stopped at the timeout counts as a failed run."""
        discharge = recorded(tmp_path, "slow.py", "import time\ntime.sleep(30)\nassert time.time() > 0\n")
        (check,) = verify(single(discharge), tmp_path, timeout=0.5)
        assert (diagnostics(check), check.status) == ([CONTRADICTED], TIMED_OUT)
        assert "timed out" in check.findings[0].message

    def test_unreachable_check_cannot_fail(self, tmp_path):
        """Test that a script whose only assert sits under ``if False`` is flagged, and remains a sorry."""
        discharge = recorded(tmp_path, "eq.py", "if False: assert False\n")
        (check,) = verify(single(discharge), tmp_path)
        assert (check.outcome, diagnostics(check), check.status) == (FLAGGED, [CANNOT_FAIL], 0)
        (sorry,) = sorries(single(discharge), "claim", tmp_path).sorries
        assert sorry.reason == CANNOT_FAIL

    def test_result_outside_tolerance_fails_though_the_script_passes(self, tmp_path):
        """Test that a numerical result outside value ± tolerance is flagged even when the script exits 0.

        The script's own assert cannot catch the bias, so only the recorded tolerance can.
        """
        (check,) = verify(single(recorded(tmp_path, "bias.py", LAX.format(bias=0.05), kind="numerical")), tmp_path)
        assert (check.outcome, check.status, diagnostics(check)) == (FLAGGED, 0, [OUT_OF_TOLERANCE])
        assert check.findings[0].details == {"result": 0.05, "value": 0.0, "tolerance": 0.01}
        assert "computed 0.05, outside 0 ± 0.01" in check.findings[0].message

    @pytest.mark.parametrize(
        ("bias", "outcome"),
        [(0.01, PASSED), (-0.01, PASSED), (0.0100001, FLAGGED), (-0.0100001, FLAGGED), ("float('nan')", FLAGGED)],
    )
    def test_tolerance_is_inclusive(self, tmp_path, bias, outcome):
        """Test the edge of the band: a result exactly at value ± tolerance passes, beyond it or NaN does not."""
        (check,) = verify(single(recorded(tmp_path, "bias.py", LAX.format(bias=bias), kind="numerical")), tmp_path)
        assert check.outcome == outcome
        assert diagnostics(check) == ([] if outcome == PASSED else [OUT_OF_TOLERANCE])

    def test_last_line_is_the_result(self, tmp_path):
        """Test that the result is the last non-blank line, after any other output."""
        text = "print('computing')\nbias = 0.5\nassert bias == bias\nprint(0.002)\nprint()\n"
        (check,) = verify(single(recorded(tmp_path, "bias.py", text, kind="numerical")), tmp_path)
        assert check.outcome == PASSED

    @pytest.mark.parametrize(("text", "printed"), [(PASSES, "nothing"), (PASSES + "print('ok')\n", "'ok'")])
    def test_result_missing(self, tmp_path, text, printed):
        """Test that a numerical script that prints no number is flagged rather than passed."""
        (check,) = verify(single(recorded(tmp_path, "bias.py", text, kind="numerical")), tmp_path)
        assert (check.outcome, diagnostics(check)) == (FLAGGED, [RESULT_MISSING])
        assert f"printed {printed} as its last line" in check.findings[0].message

    def test_failed_run_is_not_held_to_the_tolerance(self, tmp_path):
        """Test that a run that already failed is reported once, not again for its result."""
        (check,) = verify(single(recorded(tmp_path, "bias.py", BIAS.format(bias=0.5), kind="numerical")), tmp_path)
        assert (diagnostics(check), check.status) == ([CONTRADICTED], 1)

    def test_symbolic_output_is_ignored(self, tmp_path):
        """Test that a symbolic script, which records no value, is not held to what it prints."""
        (check,) = verify(single(recorded(tmp_path, "eq.py", PASSES + "print(99)\n")), tmp_path)
        assert check.outcome == PASSED

    def test_finding_names_the_relation(self, tmp_path):
        """Test that a finding names the relation and both of its nodes."""
        (check,) = verify(single(recorded(tmp_path, "eq.py", CANNOT)), tmp_path)
        (finding,) = check.findings
        assert (finding.id, finding.relations, finding.nodes) == ("cannot-fail:r", ("r",), ("premise", "claim"))


class TestVerifyOthers:
    """Citations are read where they can be; formal and judgement discharges are not checked."""

    QUOTE = "The noise is   stationary\nover the band."

    def anchor(self, document: str, digest: str | None = None) -> Discharge:
        """Return a citation of the quote in ``document``."""
        return Discharge(
            "citation", anchor=SourceAnchor(document, "sec2", "The noise is stationary over the band.", digest)
        )

    def test_quote_found(self, tmp_path):
        """Test that a quote in the cited file passes, whatever its whitespace."""
        (tmp_path / "paper.tex").write_text(f"Intro. {self.QUOTE} More.", encoding="utf-8")
        (check,) = verify(single(self.anchor("paper.tex", text_digest(self.QUOTE))), tmp_path)
        assert (check.outcome, check.findings) == (PASSED, ())

    def test_quote_not_found(self, tmp_path):
        """Test that a quote missing from the cited file is flagged."""
        (tmp_path / "paper.tex").write_text("The noise is not stationary.", encoding="utf-8")
        (check,) = verify(single(self.anchor("paper.tex")), tmp_path)
        assert (check.outcome, diagnostics(check)) == (FLAGGED, [QUOTE_NOT_FOUND])

    def test_quote_digest_mismatch(self, tmp_path):
        """Test that an anchor whose digest is not its quote's is flagged."""
        (tmp_path / "paper.tex").write_text(self.QUOTE, encoding="utf-8")
        (check,) = verify(single(self.anchor("paper.tex", text_digest("something else"))), tmp_path)
        assert (check.outcome, diagnostics(check)) == (FLAGGED, [QUOTE_DIGEST_MISMATCH])

    def test_citation_of_a_non_file_is_unchecked(self, tmp_path):
        """Test that a citation of a DOI, not a file, is reported as unchecked with the reason."""
        (check,) = verify(single(self.anchor("doi:10.0000/x")), tmp_path)
        assert (check.outcome, check.findings) == (UNCHECKED, ())
        assert "is not a file" in check.reason

    def test_formal_is_unchecked(self, tmp_path):
        """Test that a formal discharge is reported as unchecked, naming its theorem."""
        (check,) = verify(single(Discharge("formal", theorem="Paper.eq3")), tmp_path)
        assert (check.outcome, check.kind) == (UNCHECKED, "formal")
        assert "Paper.eq3" in check.reason

    def test_judgement_has_nothing_to_check(self, tmp_path):
        """Test that a judgement discharge is listed, with nothing to check."""
        (check,) = verify(single(Discharge("judgement")), tmp_path)
        assert (check.outcome, check.findings, check.status) == (NOT_CHECKABLE, (), None)

    def test_only_discharges_are_checked(self, tmp_path):
        """Test that relations with no obligation, or an open one, give no check."""
        assert verify(single(None), tmp_path) == []
        assert verify(single(None, obligation=False), tmp_path) == []

    def test_to_dict(self, tmp_path):
        """Test the JSON form of a check."""
        (check,) = verify(single(recorded(tmp_path, "eq.py", CANNOT)), tmp_path)
        data = check.to_dict()
        assert set(data) == {"relation", "kind", "outcome", "status", "reason", "findings"}
        assert data["findings"][0]["diagnostic"] == CANNOT_FAIL


def dummy(tmp_path: Path) -> Graph:
    """The example of the proposal.

    ``eq3`` follows from ``eq2`` symbolically; the result rests on ``eq3``, on a bias below 1%,
    checked numerically, and on stationary noise, a judgement. ``eq2`` and ``unrelated`` are joined
    by a judgement too, but the result does not depend on it.
    """
    g = Graph()
    for node_id, base in [("eq2", 0.9), ("eq3", 0.05), ("bias", 0.6), ("stationary", 0.7), ("result", 0.05)]:
        g.add_node(Node(node_id, base=base))
    g.add_node(Node("unrelated", base=0.5))
    eq = Obligation("Eq. (3) follows from Eq. (2) by setting x = 0", recorded(tmp_path, "eq3.py", PASSES))
    bias = Obligation(
        "the bias is below 1%",
        recorded(tmp_path, "bias.py", BIAS.format(bias=0.004), kind="numerical"),
    )
    noise = Obligation("the noise is stationary", Discharge("judgement"))
    g.add_relation(Relation("eq-step", "requires", "eq2", "eq3", strength=0.99, obligation=eq))
    g.add_relation(Relation("noise-step", "supports", "stationary", "result", strength=0.6, obligation=noise))
    g.add_relation(Relation("bias-step", "supports", "bias", "result", strength=0.5, obligation=bias))
    g.add_relation(Relation("eq-result", "supports", "eq3", "result", strength=0.9))
    g.add_relation(Relation("side", "supports", "unrelated", "eq2", strength=0.3, obligation=noise))
    g.add_relation(Relation("note", "cites", "result", "eq2"))
    return g


class TestDependencies:
    """The steps a target depends on are the inferential relations into it and its ancestors."""

    def test_ancestors_only(self, tmp_path):
        """Test that relations into ancestors count, while annotations and relations out of the target do not."""
        g = dummy(tmp_path)
        assert [r.id for r in dependencies(g, "eq3")] == ["eq-step", "side"]
        assert [r.id for r in dependencies(g, "result")] == ["eq-step", "noise-step", "bias-step", "eq-result", "side"]
        assert dependencies(g, "unrelated") == []

    def test_equivalent_and_exclusive_bring_in_the_other_side(self):
        """Test that a node merged with, or exclusive of, a node the target depends on brings its own parents."""
        g = Graph()
        for node_id in "abcdet":
            g.add_node(Node(node_id, base=0.5))
        g.add_relation(Relation("ab", "supports", "a", "t", strength=0.5))
        g.add_relation(Relation("eq", "equivalent", "a", "b"))
        g.add_relation(Relation("cb", "supports", "c", "b", strength=0.5))
        g.add_relation(Relation("x", "exclusive", "t", "d"))
        g.add_relation(Relation("out", "supports", "t", "e", strength=0.5))
        assert [r.id for r in dependencies(g, "t")] == ["ab", "eq", "cb", "x"]


class TestSorries:
    """A target's sorries are its steps with nothing but judgement behind them, ranked."""

    def test_dummy_example(self, tmp_path):
        """Test the proposal's example: the stationarity step and the unobligated step are the sorries."""
        report = sorries(dummy(tmp_path), "result", tmp_path)
        assert report.target == "result"
        assert report.steps == ("eq-step", "noise-step", "bias-step", "eq-result", "side")
        assert report.discharged == ("eq-step", "bias-step")
        assert {s.relation: s.reason for s in report.sorries} == {
            "noise-step": "judgement",
            "eq-result": NO_OBLIGATION,
            "side": "judgement",
        }
        noise = next(s for s in report.sorries if s.relation == "noise-step")
        assert (noise.obligation, noise.type, noise.source, noise.target) == (
            "the noise is stationary",
            "supports",
            "stationary",
            "result",
        )

    def test_ranked_by_exact_sensitivity(self):
        """Test the ranking against the derivative worked by hand, listed in the opposite order.

        With supports, P(t) = 1 - (1 - 0.1)(1 - s_a P(a))(1 - s_b P(b)), so
        dP/ds_a = 0.9 * 0.5 * (1 - 0.6 * 0.2) = 0.396 and dP/ds_b = 0.9 * 0.2 * (1 - 0.8 * 0.5) = 0.108.
        """
        g = Graph()
        g.add_node(Node("a", base=0.5))
        g.add_node(Node("b", base=0.2))
        g.add_node(Node("c", base=0.5))
        g.add_node(Node("t", base=0.1))
        g.add_relation(Relation("rb", "supports", "b", "t", strength=0.6))
        g.add_relation(Relation("x", "exclusive", "c", "t"))
        g.add_relation(Relation("ra", "supports", "a", "t", strength=0.8))
        report = sorries(g, "t", ".")
        assert [s.relation for s in report.sorries] == ["ra", "rb", "x"]
        assert report.sorries[2].sensitivity is None
        # The exclusive constraint conditions the target, so compare without it.
        g2 = Graph()
        for node in g:
            if node.id != "c":
                g2.add_node(node)
        g2.add_relation(g.relations["rb"])
        g2.add_relation(g.relations["ra"])
        ranked = sorries(g2, "t", ".").sorries
        assert [s.relation for s in ranked] == ["ra", "rb"]
        assert ranked[0].sensitivity == pytest.approx(0.396, rel=1e-12, abs=0.0)
        assert ranked[1].sensitivity == pytest.approx(0.108, rel=1e-12, abs=0.0)

    @pytest.mark.parametrize(
        ("make", "reason"),
        [
            (lambda d: None, UNDISCHARGED),
            (lambda d: Discharge("judgement"), "judgement"),
            (lambda d: Discharge("symbolic", script=script(d, "eq.py", PASSES)), NEVER_RUN),
            (lambda d: recorded(d, "eq.py", FAILS, status=1), RECORDED_FAILURE),
            (lambda d: Discharge("symbolic", script="absent.py", digest=file_digest(b""), status=0), MISSING_SCRIPT),
            (lambda d: recorded(d, "eq.py", CANNOT), CANNOT_FAIL),
        ],
    )
    def test_reasons(self, tmp_path, make, reason):
        """Test each way a discharged-looking step still rests on judgement only."""
        (sorry,) = sorries(single(make(tmp_path)), "claim", tmp_path).sorries
        assert (sorry.relation, sorry.reason) == ("r", reason)
        assert sorry.sensitivity == pytest.approx(0.9 * 0.5, rel=1e-12, abs=0.0)

    @pytest.mark.parametrize(
        "make",
        [
            lambda d: recorded(d, "eq.py", PASSES),
            lambda d: recorded(d, "eq.py", PASSES, kind="numerical"),
            lambda d: Discharge("formal", theorem="T"),
            lambda d: Discharge("citation", anchor=SourceAnchor("doi:10.0/x", quote="q")),
        ],
    )
    def test_discharged(self, tmp_path, make):
        """Test that a recorded passing script that can fail, a theorem and a citation each discharge a step."""
        report = sorries(single(make(tmp_path)), "claim", tmp_path)
        assert (report.discharged, report.sorries) == (("r",), ())

    def test_record_is_read_not_rerun(self, tmp_path):
        """Test that the report trusts the record: a recorded pass counts though the script now fails."""
        discharge = recorded(tmp_path, "eq.py", PASSES)
        script(tmp_path, "eq.py", "raise SystemExit(1)\n")
        assert sorries(single(discharge), "claim", tmp_path).discharged == ("r",)

    def test_target_must_be_a_variable(self, tmp_path):
        """Test that a target outside inference is refused."""
        g = single(None)
        g.add_node(Node("alice", kind="person"))
        with pytest.raises(ValidationError):
            sorries(g, "alice", tmp_path)

    def test_to_dict(self, tmp_path):
        """Test the JSON form of the report."""
        data = sorries(dummy(tmp_path), "result", tmp_path).to_dict()
        assert set(data) == {"target", "steps", "discharged", "sorries"}
        assert set(data["sorries"][0]) == {
            "relation",
            "type",
            "source",
            "target",
            "reason",
            "obligation",
            "sensitivity",
        }
