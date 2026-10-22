"""The ``verify`` and ``sorries`` commands: re-checking discharges, and what a target still rests on."""

from __future__ import annotations

import math
from typing import Annotated

import typer

from credencegraph.cli.common import (
    INVALID_ARGUMENT,
    VERIFICATION_FAILED,
    CliError,
    GraphPath,
    JsonOption,
    Result,
    compile_checked,
    read_graph,
    require_variables,
    respond,
)
from credencegraph.diagnostics.obligations import DEFAULT_TIMEOUT, FLAGGED, sorries, verify


def verify_command(
    path: GraphPath,
    timeout: Annotated[
        float, typer.Option(help="Seconds a script may run before it is stopped and its run counted as failed.")
    ] = DEFAULT_TIMEOUT,
    as_json: JsonOption = False,
) -> None:
    """Re-check every discharge in a graph file, running each recorded script.

    Scripts and cited documents are found relative to the graph file's directory, and each script is
    run with the current Python from its own directory, so verify only graphs whose scripts you trust.
    A script is flagged when it cannot be read, was never run, changed since its recorded run, has no
    reachable assert, raise or non-zero exit, or now fails. A numerical script that exits 0 is also
    flagged when the result it prints last lies outside its value plus or minus its tolerance, or is
    not a number. A citation of a file is flagged when the quote is not
    in it. Formal discharges, and citations of anything but a file, are reported as unchecked.
    Any flagged discharge fails the command with a verification-failed error.
    """

    def action() -> Result:
        if not 0.0 < timeout < math.inf:
            raise CliError(
                INVALID_ARGUMENT,
                f"--timeout must be a finite number of seconds above 0, got {timeout!r}",
                f"the default is {DEFAULT_TIMEOUT:g} seconds",
            )
        graph = read_graph(path)
        checks = verify(graph, path.parent, timeout=timeout)
        payload = {"path": str(path), "checks": [check.to_dict() for check in checks]}
        flagged = [check for check in checks if check.outcome == FLAGGED]
        if flagged:
            listed = "; ".join(finding.message for check in flagged for finding in check.findings)
            raise CliError(
                VERIFICATION_FAILED,
                f"{len(flagged)} of {len(checks)} discharge(s) failed verification: {listed}",
                "fix or re-run each script named, then record its new digest and exit status in the graph",
                payload,
            )
        text = [
            f"{check.relation}: {check.kind} {check.outcome}" + (f" ({check.reason})" if check.reason else "")
            for check in checks
        ]
        return Result(payload, text or ["no discharges"])

    respond("verify", as_json, action)


def sorries_command(
    path: GraphPath,
    target: Annotated[str, typer.Option(help="The node whose remaining sorries are wanted.", show_default=False)],
    as_json: JsonOption = False,
) -> None:
    """Report the steps a target depends on that rest on judgement only, ranked by the target's sensitivity.

    A step is discharged by a formal or citation discharge, or by a script whose recorded run passed
    and which can fail; every other step the target depends on is a sorry. The record is read as it
    stands, without running anything: run verify to re-check it.
    """

    def action() -> Result:
        graph = read_graph(path)
        require_variables(graph, [target], "--target")
        compile_checked(graph)
        report = sorries(graph, target, path.parent)
        payload = {"path": str(path), **report.to_dict()}
        text = [f"{target}: {len(report.sorries)} sorry(ies) of {len(report.steps)} step(s)"]
        for sorry in report.sorries:
            moved = "" if sorry.sensitivity is None else f", dP/dstrength = {sorry.sensitivity:.4g}"
            text.append(f"{sorry.relation}: {sorry.reason}{moved}")
        return Result(payload, text)

    respond("sorries", as_json, action)
