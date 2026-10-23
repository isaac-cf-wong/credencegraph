"""Re-checking the discharges of a graph's obligations, and the sorries a target still rests on.

``verify`` re-runs every script discharge, holds a numerical script's printed result to its
recorded value and tolerance, and re-reads every citation it can reach. It runs the scripts the
graph names, as the current user, so run it only on graphs whose scripts you trust.

``sorries`` reads the record alone, without running anything: a step the target depends on is
discharged by a ``formal`` or ``citation`` discharge, or by a script whose recorded run passed and
which has a reachable way to fail. Every other step is a *sorry*, resting on judgement only, and the
report ranks the sorries by how far the target's probability moves with each one's strength.
"""

from __future__ import annotations

import ast
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from credencegraph.core.attributes import JSON
from credencegraph.core.graph import Graph
from credencegraph.core.obligation import (
    CITATION,
    FORMAL,
    JUDGEMENT,
    NUMERICAL,
    SCRIPT_KINDS,
    Discharge,
    file_digest,
)
from credencegraph.core.relation import EQUIVALENT, EXCLUSIVE, STRENGTH_TYPES, Relation
from credencegraph.diagnostics.records import Finding
from credencegraph.diagnostics.weak_points import derivatives
from credencegraph.inference.engine import Engine
from credencegraph.rubric.text import normalise_text, text_digest
from credencegraph.semantics.compiler import compile_graph
from credencegraph.semantics.network import ParameterKey

# Findings of verify.
MISSING_SCRIPT = "missing-script"
NEVER_RUN = "never-run"
SCRIPT_CHANGED = "script-changed"
CANNOT_FAIL = "cannot-fail"
CONTRADICTED = "contradicted"
CHECK_FAILED = "check-failed"
RESULT_MISSING = "result-missing"
OUT_OF_TOLERANCE = "out-of-tolerance"
QUOTE_NOT_FOUND = "quote-not-found"
QUOTE_DIGEST_MISMATCH = "quote-digest-mismatch"

# Outcomes of a check.
PASSED = "passed"
FLAGGED = "flagged"
UNCHECKED = "unchecked"
NOT_CHECKABLE = "judgement"

# Why a step is a sorry.
NO_OBLIGATION = "no-obligation"
UNDISCHARGED = "undischarged"
RECORDED_FAILURE = "recorded-failure"
SORRY_REASONS = (NO_OBLIGATION, UNDISCHARGED, JUDGEMENT, NEVER_RUN, RECORDED_FAILURE, MISSING_SCRIPT, CANNOT_FAIL)

# Seconds a script may run before verify stops it and counts the run as failed.
DEFAULT_TIMEOUT = 300.0

# The exit status recorded for a run stopped at the timeout.
TIMED_OUT = -1

_EXITS = {("sys", "exit"), ("os", "_exit"), (None, "exit"), (None, "quit")}


def _exits_nonzero(call: ast.Call) -> bool:
    """Tell whether a call is an exit that may carry a non-zero status.

    ``sys.exit()``, ``sys.exit(0)`` and ``sys.exit(None)`` always succeed; any other argument may not.
    """
    func = call.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        name = (func.value.id, func.attr)
    elif isinstance(func, ast.Name):
        name = (None, func.id)
    else:
        return False
    if name not in _EXITS or not call.args:
        return False
    argument = call.args[0]
    return not (isinstance(argument, ast.Constant) and argument.value in {0, None})


def _constant(test: ast.expr) -> bool | None:
    """Return the truth of a condition fixed in the source, such as ``False``, ``0`` or ``not True``.

    A literal, the ``not`` of a constant condition, and a conditional expression whose condition and
    taken branch are both constant, such as ``False if True else x``, are constant at any depth.

    Returns:
        ``True`` or ``False`` for a constant condition, ``None`` for one only a run can decide.
    """
    if isinstance(test, ast.Constant):
        return bool(test.value)
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = _constant(test.operand)
        return None if inner is None else not inner
    if isinstance(test, ast.IfExp):
        taken = _constant(test.test)
        return None if taken is None else _constant(test.body if taken else test.orelse)
    return None


class _DeadCode(ast.NodeTransformer):
    """Drop the code a constant condition makes unreachable, and the asserts that cannot fail."""

    def _block(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        """Visit a block of statements and return them as one flat list, as ``generic_visit`` does.

        A statement may come back as a list, from a constant ``if`` or ``while`` nested in the block,
        or as ``None``, from a dropped assert; the list is spliced in and the ``None`` left out, so no
        statement a nested fold keeps is lost from the tree.
        """
        out: list[ast.stmt] = []
        for stmt in stmts:
            visited = self.visit(stmt)
            if visited is None:
                continue
            out.extend(visited if isinstance(visited, list) else [visited])
        return out

    def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt]:
        """Keep only the branch a constant condition takes."""
        taken = _constant(node.test)
        if taken is None:
            return self.generic_visit(node)
        return self._block(node.body if taken else node.orelse)

    def visit_While(self, node: ast.While) -> ast.AST | list[ast.stmt]:
        """Drop the body of a loop whose condition is constantly false."""
        if _constant(node.test) is False:
            return self._block(node.orelse)
        return self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        """Keep only the branch of a conditional expression that a constant condition takes."""
        taken = _constant(node.test)
        if taken is None:
            return self.generic_visit(node)
        return self.visit(node.body if taken else node.orelse)

    def visit_Assert(self, node: ast.Assert) -> ast.AST | None:
        """Drop an assert of a condition that is constantly true."""
        if _constant(node.test) is True:
            return None
        return self.generic_visit(node)


def can_fail(source: str) -> bool:
    """Tell whether a Python script has a reachable way to fail: an ``assert``, a ``raise``, or a non-zero exit.

    Code that a constant condition makes unreachable is ignored first: the body of ``if False:``,
    ``if 0:`` or ``while False:``, the ``else`` of ``if True:``, the dead branch of a conditional
    expression, and an ``assert`` of a constantly true condition such as ``assert True``. A condition
    counts as constant when it is a literal, the ``not`` of a constant condition, or a conditional
    expression whose condition and taken branch are constant, at any depth of nesting, so
    ``if (False if True else False):`` is pruned like ``if False:`` and
    ``assert (True if True else False)`` cannot fail; a check behind any other condition is taken to be
    reachable. A script that does not parse has no checked failure path, only a crash, and counts as
    unable to fail.

    Args:
        source: The script's text.

    Returns:
        ``True`` when the reachable code holds an ``assert`` or ``raise`` statement, or an exit call
        (``sys.exit``, ``os._exit``, ``exit`` or ``quit``) whose argument is not ``0`` or ``None``.
    """
    try:
        tree = _DeadCode().visit(ast.parse(source))
    except (SyntaxError, ValueError):
        return False
    return any(
        isinstance(node, ast.Assert | ast.Raise) or (isinstance(node, ast.Call) and _exits_nonzero(node))
        for node in ast.walk(tree)
    )


@dataclass(frozen=True, slots=True)
class Check:
    """The re-check of one discharge.

    Attributes:
        relation: The id of the relation whose obligation the discharge meets.
        kind: The discharge's kind.
        outcome: ``passed``; ``flagged`` when there are findings; ``unchecked`` when this package
            cannot check it, a ``formal`` discharge or a citation of a document that is not a local
            file; or ``judgement``, which has nothing to check.
        findings: What the re-check found wrong.
        status: The exit status of the script's run now, ``-1`` if it timed out; ``None`` if no
            script ran.
        reason: Why an unchecked discharge was not checked.
    """

    relation: str
    kind: str
    outcome: str
    findings: tuple[Finding, ...] = ()
    status: int | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, JSON]:
        """Return the check as JSON-ready data.

        Returns:
            ``{"relation", "kind", "outcome", "status", "reason", "findings"}``.
        """
        return {
            "relation": self.relation,
            "kind": self.kind,
            "outcome": self.outcome,
            "status": self.status,
            "reason": self.reason,
            "findings": [finding.to_dict() for finding in self.findings],
        }


def _finding(diagnostic: str, relation: Relation, message: str, **details: float) -> Finding:
    return Finding(
        f"{diagnostic}:{relation.id}",
        diagnostic,
        f"relation {relation.id!r}: {message}",
        nodes=(relation.source, relation.target),
        relations=(relation.id,),
        details=details,
    )


def _run(script: Path, timeout: float) -> tuple[int, str]:
    """Run a Python script from its own directory.

    Returns:
        Its exit status, ``TIMED_OUT`` on timeout, and its standard output.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - the interpreter and a recorded script, no shell
            [sys.executable, str(script)],
            cwd=script.parent,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return TIMED_OUT, ""
    return completed.returncode, completed.stdout.decode("utf-8", errors="replace")


def _check_result(relation: Relation, discharge: Discharge, output: str) -> Finding | None:
    """Check the result a numerical script printed, its last non-blank line, against value and tolerance."""
    assert discharge.value is not None  # noqa: S101 - Discharge guarantees both for a numerical discharge
    assert discharge.tolerance is not None  # noqa: S101
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    try:
        result = float(lines[-1])
    except (IndexError, ValueError):
        printed = repr(lines[-1]) if lines else "nothing"
        return _finding(
            RESULT_MISSING,
            relation,
            f"the script {discharge.script!r} printed {printed} as its last line, not a number to hold to "
            f"{discharge.value:g} ± {discharge.tolerance:g}",
        )
    if math.isfinite(result) and abs(result - discharge.value) <= discharge.tolerance:
        return None
    return _finding(
        OUT_OF_TOLERANCE,
        relation,
        f"the script {discharge.script!r} computed {result:g}, outside {discharge.value:g} ± {discharge.tolerance:g}",
        **({"result": result} if math.isfinite(result) else {}),
        value=discharge.value,
        tolerance=discharge.tolerance,
    )


def _check_script(relation: Relation, discharge: Discharge, directory: Path, timeout: float) -> Check:
    path = directory / str(discharge.script)
    try:
        data = path.read_bytes()
    except OSError as error:
        finding = _finding(MISSING_SCRIPT, relation, f"cannot read the script {discharge.script!r}: {error}")
        return Check(relation.id, discharge.kind, FLAGGED, (finding,))
    findings = []
    if discharge.status is None:
        findings.append(_finding(NEVER_RUN, relation, f"the script {discharge.script!r} has no recorded run"))
    elif discharge.digest != file_digest(data):
        findings.append(
            _finding(SCRIPT_CHANGED, relation, f"the script {discharge.script!r} changed since its recorded run")
        )
    if not can_fail(data.decode("utf-8", errors="replace")):
        findings.append(
            _finding(
                CANNOT_FAIL,
                relation,
                f"the script {discharge.script!r} has no reachable assert, raise or non-zero exit, so it cannot fail",
            )
        )
    status, output = _run(path, timeout)
    ran = "timed out" if status == TIMED_OUT else f"exited with {status}"
    if discharge.status is not None and (status == 0) != (discharge.status == 0):
        findings.append(
            _finding(
                CONTRADICTED,
                relation,
                f"the script {discharge.script!r} {ran}; its recorded run exited with {discharge.status}",
                status=status,
                recorded=discharge.status,
            )
        )
    elif status != 0:
        findings.append(_finding(CHECK_FAILED, relation, f"the script {discharge.script!r} {ran}", status=status))
    if status == 0 and discharge.kind == NUMERICAL:
        result = _check_result(relation, discharge, output)
        if result is not None:
            findings.append(result)
    return Check(relation.id, discharge.kind, FLAGGED if findings else PASSED, tuple(findings), status)


def _check_citation(relation: Relation, discharge: Discharge, directory: Path) -> Check:
    anchor = discharge.anchor
    assert anchor is not None  # noqa: S101 - Discharge guarantees it for a citation
    assert anchor.quote is not None  # noqa: S101
    path = directory / anchor.document
    if not path.is_file():
        reason = f"the cited document {anchor.document!r} is not a file next to the graph"
        return Check(relation.id, discharge.kind, UNCHECKED, reason=reason)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        return Check(relation.id, discharge.kind, UNCHECKED, reason=f"cannot read {anchor.document!r}: {error}")
    findings = []
    if anchor.digest is not None and anchor.digest != text_digest(anchor.quote):
        findings.append(_finding(QUOTE_DIGEST_MISMATCH, relation, "the anchor's digest is not the digest of its quote"))
    if normalise_text(anchor.quote) not in normalise_text(text):
        findings.append(_finding(QUOTE_NOT_FOUND, relation, f"the quote is not in {anchor.document!r}"))
    return Check(relation.id, discharge.kind, FLAGGED if findings else PASSED, tuple(findings))


def verify(graph: Graph, directory: str | Path, *, timeout: float = DEFAULT_TIMEOUT) -> list[Check]:
    """Re-check every discharge in a graph.

    A script discharge is run with the current interpreter, from the script's directory, and is
    flagged when the script cannot be read, was never run, changed since its recorded run, has no
    reachable way to fail (see ``can_fail``), or now fails: as ``contradicted`` when its recorded run
    passed, or when it now passes though its recorded run failed, and as ``check-failed`` otherwise.
    When a numerical script exits 0, the last non-blank line it printed is its result, flagged as
    ``result-missing`` when it is not a number and as ``out-of-tolerance`` when it lies further than
    ``tolerance`` from ``value``. A citation is flagged when its
    quote is not in the cited document, read as a file relative to ``directory``, or its digest is
    not the quote's; a citation of anything else is unchecked. A ``formal`` discharge is unchecked,
    since no proof checker ships with this package.

    Args:
        graph: The graph.
        directory: The directory that script paths and cited documents are relative to, normally
            the graph file's.
        timeout: Seconds a script may run before it is stopped and its run counted as failed.

    Returns:
        One check per discharge, in the graph's relation order.
    """
    directory = Path(directory)
    checks = []
    for relation in graph.relations.values():
        discharge = relation.obligation and relation.obligation.discharge
        if discharge is None:
            continue
        if discharge.kind in SCRIPT_KINDS:
            checks.append(_check_script(relation, discharge, directory, timeout))
        elif discharge.kind == CITATION:
            checks.append(_check_citation(relation, discharge, directory))
        elif discharge.kind == FORMAL:
            reason = f"no proof checker ships with this package to build theorem {discharge.theorem!r}"
            checks.append(Check(relation.id, discharge.kind, UNCHECKED, reason=reason))
        else:
            checks.append(Check(relation.id, discharge.kind, NOT_CHECKABLE))
    return checks


@dataclass(frozen=True, slots=True)
class Sorry:
    """A step a target depends on that rests on judgement only.

    Attributes:
        relation: The relation's id.
        type: The relation's type.
        source: Its source node.
        target: Its target node.
        reason: Why it counts as a sorry, one of ``SORRY_REASONS``.
        obligation: The obligation's statement, if the relation has one.
        sensitivity: ``dP(target)/d strength``, or ``None`` for a relation without a strength.
    """

    relation: str
    type: str
    source: str
    target: str
    reason: str
    obligation: str | None = None
    sensitivity: float | None = None

    def to_dict(self) -> dict[str, JSON]:
        """Return the sorry as JSON-ready data.

        Returns:
            ``{"relation", "type", "source", "target", "reason", "obligation", "sensitivity"}``.
        """
        return {
            "relation": self.relation,
            "type": self.type,
            "source": self.source,
            "target": self.target,
            "reason": self.reason,
            "obligation": self.obligation,
            "sensitivity": self.sensitivity,
        }


@dataclass(frozen=True, slots=True)
class SorryReport:
    """The steps a target depends on, split into the discharged ones and the sorries.

    Attributes:
        target: The target node.
        steps: The ids of the inferential relations the target depends on, in the graph's order.
        discharged: The ids of those relations whose obligation is discharged by more than judgement.
        sorries: The rest, by decreasing absolute sensitivity, those without a strength last.
    """

    target: str
    steps: tuple[str, ...]
    discharged: tuple[str, ...]
    sorries: tuple[Sorry, ...] = ()

    def to_dict(self) -> dict[str, JSON]:
        """Return the report as JSON-ready data.

        Returns:
            ``{"target", "steps", "discharged", "sorries"}``.
        """
        return {
            "target": self.target,
            "steps": list(self.steps),
            "discharged": list(self.discharged),
            "sorries": [sorry.to_dict() for sorry in self.sorries],
        }


def dependencies(graph: Graph, target: str) -> list[Relation]:
    """Find the inferential relations a target depends on.

    These are the ``requires``, ``supports`` and ``refutes`` relations into the target or into any
    node it depends on, and the ``equivalent`` and ``exclusive`` relations that touch either; the
    other end of each also becomes a node the target depends on.

    Args:
        graph: The graph.
        target: The target node's id.

    Returns:
        The relations, in the graph's order.
    """
    reached = {target}
    found: set[str] = set()
    grew = True
    while grew:
        grew = False
        for relation in graph.relations.values():
            if relation.id in found:
                continue
            if relation.type in STRENGTH_TYPES:
                touches = relation.target in reached
            elif relation.type in {EQUIVALENT, EXCLUSIVE}:
                touches = relation.source in reached or relation.target in reached
            else:
                continue
            if touches:
                found.add(relation.id)
                reached.update((relation.source, relation.target))
                grew = True
    return [relation for relation in graph.relations.values() if relation.id in found]


def sorry_reason(relation: Relation, directory: str | Path) -> str | None:  # noqa: PLR0911 - one return per reason
    """Tell why a relation's step rests on judgement only, reading only what is recorded.

    A step is discharged by a ``formal`` or ``citation`` discharge, or by a script discharge whose
    recorded run passed and whose script, read from ``directory``, has a way to fail. ``verify``
    is what re-runs the scripts.

    Args:
        relation: The relation.
        directory: The directory script paths are relative to.

    Returns:
        ``None`` for a discharged step, otherwise one of ``SORRY_REASONS``.
    """
    if relation.obligation is None:
        return NO_OBLIGATION
    discharge = relation.obligation.discharge
    if discharge is None:
        return UNDISCHARGED
    if discharge.kind == JUDGEMENT:
        return JUDGEMENT
    if discharge.kind not in SCRIPT_KINDS:
        return None
    if discharge.status is None:
        return NEVER_RUN
    if discharge.status != 0:
        return RECORDED_FAILURE
    try:
        source = (Path(directory) / str(discharge.script)).read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return MISSING_SCRIPT
    return None if can_fail(source) else CANNOT_FAIL


def sorries(graph: Graph, target: str, directory: str | Path, *, engine: Engine | None = None) -> SorryReport:
    """Report the steps a target depends on that remain sorries, ranked by the target's sensitivity.

    Each sorry with a strength is ranked by ``dP(target)/d strength``, taken exactly as in
    ``derivatives`` with every parameter at its credence mean. A discharged step is not ranked:
    credence applies only to what remains unproven.

    Args:
        graph: The graph, which must compile for inference.
        target: The target node's id.
        directory: The directory script paths are relative to.
        engine: The exact engine; variable elimination by default.

    Returns:
        The report.

    Raises:
        CompileError: If the graph cannot be compiled.
        ValidationError: If ``target`` is not an inference variable.
        ZeroProbabilityError: If no world satisfies the ``exclusive`` constraints.
    """
    slopes = derivatives(compile_graph(graph), target, engine=engine)
    steps = dependencies(graph, target)
    discharged = []
    found = []
    for relation in steps:
        reason = sorry_reason(relation, directory)
        if reason is None:
            discharged.append(relation.id)
            continue
        found.append(
            Sorry(
                relation.id,
                relation.type,
                relation.source,
                relation.target,
                reason,
                relation.obligation and relation.obligation.statement,
                slopes.get(ParameterKey("strength", relation.id)),
            )
        )
    found.sort(key=lambda sorry: (sorry.sensitivity is None, -abs(sorry.sensitivity or 0.0)))
    return SorryReport(target, tuple(relation.id for relation in steps), tuple(discharged), tuple(found))
