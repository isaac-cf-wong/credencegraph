"""The ``diagnose`` and ``check`` commands: weak points of a target, and whether a graph file is usable."""

from __future__ import annotations

from typing import Annotated

import typer

from credencegraph.cli.common import (
    GraphPath,
    JsonOption,
    Result,
    read_graph,
    require_nodes,
    respond,
    translate,
)
from credencegraph.diagnostics.claims import DEFAULT_CLAIM_THRESHOLD
from credencegraph.diagnostics.records import Finding
from credencegraph.diagnostics.report import diagnose
from credencegraph.diagnostics.structure import missing_parameters, unanchored
from credencegraph.diagnostics.weak_points import DEFAULT_FAILURE_THRESHOLD
from credencegraph.semantics.compiler import compile_graph
from credencegraph.semantics.errors import CompileError


def _lines(findings: list[Finding]) -> list[str]:
    """Render findings one per line.

    Args:
        findings: The findings.

    Returns:
        ``<id>: <message>`` for each finding.
    """
    return [f"{finding.id}: {finding.message}" for finding in findings]


def diagnose_command(
    path: GraphPath,
    target: Annotated[
        str | None,
        typer.Option(help="The node whose weak points are wanted; without it only the graph-wide checks run."),
    ] = None,
    claim_threshold: Annotated[
        float, typer.Option(help="Report |stated - computed| above this as an overclaim or underclaim.")
    ] = DEFAULT_CLAIM_THRESHOLD,
    failure_threshold: Annotated[
        float, typer.Option(help="Report a premise whose failure leaves the target below this.")
    ] = DEFAULT_FAILURE_THRESHOLD,
    as_json: JsonOption = False,
) -> None:
    """Report the weak points of a graph file and, with --target, of one of its nodes."""

    def action() -> Result:
        graph = read_graph(path)
        if target is not None:
            require_nodes(graph, (target,), "--target")
        findings = diagnose(graph, target, claim_threshold=claim_threshold, failure_threshold=failure_threshold)
        payload = {"path": str(path), "target": target, "findings": [finding.to_dict() for finding in findings]}
        return Result(payload, _lines(findings) or ["no findings"])

    respond("diagnose", as_json, action, path)


def check_command(path: GraphPath, as_json: JsonOption = False) -> None:
    """Check that a graph file is valid and can be compiled for inference.

    The file must parse as a credencegraph graph: known fields, unique ids, relations between existing
    nodes, and no cycle among requires, supports and refutes. Every inference variable needs a base.
    An unanchored variable is reported but does not fail the check. The exit status is 1 when the
    check fails.
    """

    def action() -> Result:
        graph = read_graph(path)
        errors = missing_parameters(graph)
        warnings = unanchored(graph)
        compile_error = None
        if not errors:
            try:
                compile_graph(graph)
            except CompileError as error:
                compile_error = translate(error, path).to_dict()
        ok = not errors and compile_error is None
        payload = {
            "path": str(path),
            "ok": ok,
            "nodes": len(graph.nodes),
            "relations": len(graph.relations),
            "errors": [finding.to_dict() for finding in errors],
            "compile_error": compile_error,
            "warnings": [finding.to_dict() for finding in warnings],
        }
        verdict = "ok" if ok else "failed"
        text = [f"{path}: {verdict} ({len(graph.nodes)} nodes, {len(graph.relations)} relations)"]
        text.extend(f"error: {line}" for line in _lines(errors))
        if compile_error is not None:
            text.append(f"error: {compile_error['message']}")
        text.extend(f"warning: {line}" for line in _lines(warnings))
        return Result(payload, text, 0 if ok else 1)

    respond("check", as_json, action, path)
