"""Shared pieces of the command line: argument parsing, graph files, and the JSON or text response.

Every command takes ``--json``. With it, the command prints exactly one JSON object on stdout, on
success and on failure alike: ``{"command": ..., ...}`` or ``{"command": ..., "error": {...}}``.
Without it, a success prints a short summary on stdout and a failure prints ``error: ...`` and a
hint on stderr. A failure exits with status 1 in both modes.
"""

from __future__ import annotations

import difflib
import json
import os
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from credencegraph.core.anchor import SourceAnchor
from credencegraph.core.credence import Beta, Credence, Point
from credencegraph.core.errors import CredenceGraphError, CycleError, ValidationError
from credencegraph.core.graph import Graph
from credencegraph.core.serialization import credence_to_json, dumps, loads
from credencegraph.diagnostics.records import render_item
from credencegraph.diagnostics.structure import missing_parameters
from credencegraph.inference.errors import ProblemTooLargeError, ZeroProbabilityError
from credencegraph.semantics.compiler import compile_graph, inference_sets
from credencegraph.semantics.errors import CompileError
from credencegraph.semantics.network import Network

# Error codes. They are part of the JSON output and stay stable across releases.
FILE_EXISTS = "file-exists"
FILE_NOT_FOUND = "file-not-found"
IO_ERROR = "io-error"
INVALID_GRAPH = "invalid-graph"
INVALID_ARGUMENT = "invalid-argument"
DUPLICATE_ID = "duplicate-id"
UNKNOWN_NODE = "unknown-node"
CYCLE = "cycle"
COMPILE_ERROR = "compile-error"
ZERO_PROBABILITY = "zero-probability"
PROBLEM_TOO_LARGE = "problem-too-large"

ERROR_CODES = (
    FILE_EXISTS,
    FILE_NOT_FOUND,
    IO_ERROR,
    INVALID_GRAPH,
    INVALID_ARGUMENT,
    DUPLICATE_ID,
    UNKNOWN_NODE,
    CYCLE,
    COMPILE_ERROR,
    ZERO_PROBABILITY,
    PROBLEM_TOO_LARGE,
)

# Why no world satisfies the exclusive relations. If every base and strength lies strictly between 0
# and 1, the world in which every proposition is false has positive probability and satisfies every
# exclusive relation, so a probability of zero needs a parameter at exactly 0 or 1.
IMPOSSIBLE_GRAPH_HINT = (
    "no world satisfies every exclusive relation under the graph's credences, which needs some base or "
    "strength of exactly 0 or 1; soften one of those or remove an exclusive relation"
)

CREDENCE_FORMS = "a probability such as 0.3, or beta:ALPHA,BETA such as beta:8,2"

JsonOption = Annotated[bool, typer.Option("--json", help="Print one JSON object on stdout instead of text.")]
GraphPath = Annotated[Path, typer.Argument(help="The graph file.", show_default=False)]


@dataclass
class CliError(Exception):
    """A failure to report to the caller.

    Attributes:
        code: A stable identifier of the kind of failure, such as ``"unknown-node"``.
        message: What went wrong.
        hint: What to do about it, if there is a single obvious remedy.
        details: Structured data behind the failure, such as the node ids around a cycle.
    """

    code: str
    message: str
    hint: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Return the message."""
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Return the error as JSON-ready data.

        Returns:
            ``{"code", "message", "hint", "details"}``.
        """
        return {"code": self.code, "message": self.message, "hint": self.hint, "details": self.details}


@dataclass
class Result:
    """The outcome of a successful command.

    Attributes:
        payload: The JSON fields of the response, besides ``command``.
        text: The summary printed without ``--json``, one line per entry.
    """

    payload: dict[str, Any]
    text: list[str]


def translate(error: CredenceGraphError, command: str | None = None) -> CliError:
    """Turn a library error into a ``CliError`` with a code and a remedy.

    Commands check their arguments before calling the library, and explain a graph that cannot be
    compiled with ``compile_failure``, so the last branch is a safety net for a validation error
    no command anticipated.

    Args:
        error: The library error.
        command: The command that raised it, named in the fallback hint.

    Returns:
        The error to report.
    """
    if isinstance(error, CycleError):
        return CliError(
            CYCLE,
            str(error),
            "requires, supports and refutes relations must not form a cycle; drop or reverse one of the relations named",
            {"cycle": list(error.cycle), "relations": list(error.relations)},
        )
    if isinstance(error, ZeroProbabilityError):
        return CliError(ZERO_PROBABILITY, str(error), IMPOSSIBLE_GRAPH_HINT)
    if isinstance(error, ProblemTooLargeError):
        return CliError(
            PROBLEM_TOO_LARGE,
            str(error),
            "reduce how many relations meet at one node, for example by merging related premises into one; "
            "from Python, raise max_factor_size on VariableElimination instead",
            {"required": error.required, "limit": error.limit},
        )
    usage = f"'credencegraph {command} --help'" if command else "'credencegraph --help'"
    return CliError(INVALID_ARGUMENT, str(error), f"check the arguments against {usage}")


def format_credence(credence: Credence) -> str:
    """Write a credence the way it is typed on the command line.

    Args:
        credence: The credence.

    Returns:
        ``0.3`` for a ``Point``, ``beta:8,2`` for a ``Beta``.
    """

    def number(value: float) -> str:
        return str(int(value)) if value.is_integer() else repr(value)

    if isinstance(credence, Point):
        return number(credence.p)
    return f"beta:{number(credence.alpha)},{number(credence.beta)}"


def compile_failure(graph: Graph, error: CompileError) -> CliError:
    """Explain why a graph cannot be compiled, from the graph itself rather than the message.

    Args:
        graph: The graph.
        error: The compiler's error, reported as is when the graph shows no missing or conflicting base.

    Returns:
        A ``compile-error`` naming the nodes without a base and the equivalent nodes whose bases
        conflict, with ``details.errors`` (the missing-parameter findings) and
        ``details.conflicting_bases``.
    """
    missing = missing_parameters(graph)
    conflicts = []
    for ids in inference_sets(graph).values():
        carriers = [node_id for node_id in ids if graph.nodes[node_id].base is not None]
        conflicts.extend(
            (carriers[0], other) for other in carriers[1:] if graph.nodes[other].base != graph.nodes[carriers[0]].base
        )
    details = {
        "errors": [finding.to_dict() for finding in missing],
        "conflicting_bases": [
            {"nodes": [a, b], "bases": [credence_to_json(graph.nodes[a].base), credence_to_json(graph.nodes[b].base)]}  # type: ignore[arg-type]
            for a, b in conflicts
        ],
    }
    messages = []
    hints = []
    if conflicts:
        messages.extend(
            f"equivalent nodes {a!r} and {b!r} have conflicting bases "
            f"{format_credence(graph.nodes[a].base)} and {format_credence(graph.nodes[b].base)}"  # type: ignore[arg-type]
            for a, b in conflicts
        )
        hints.append("give equivalent nodes the same base, or a base on only one of them")
    if missing:
        nodes = ", ".join(repr(finding.nodes[0]) for finding in missing)
        messages.append(f"inference variables without a base: {nodes}")
        hints.append("set a base on each node listed in the graph file; there is no default credence")
    if not messages:
        return CliError(
            COMPILE_ERROR,
            str(error),
            "an equivalent relation merges nodes that the relations named then join in a cycle; "
            "remove the equivalent relation or one relation of the cycle",
            details,
        )
    return CliError(COMPILE_ERROR, "; ".join(messages), "; ".join(hints), details)


def compile_checked(graph: Graph) -> Network:
    """Compile a graph, explaining a failure with ``compile_failure``.

    Args:
        graph: The graph.

    Returns:
        The network.

    Raises:
        CliError: If the graph cannot be compiled.
    """
    try:
        return compile_graph(graph)
    except CompileError as error:
        raise compile_failure(graph, error) from None


def respond(command: str, as_json: bool, action: Callable[[], Result]) -> NoReturn:
    """Run a command's action and print its result or its error.

    Args:
        command: The command's name, echoed in the JSON output.
        as_json: Whether to print JSON instead of text.
        action: The work of the command.

    Raises:
        typer.Exit: With status 0 on success, or 1 on an error.
    """
    try:
        result = action()
    except CliError as error:
        _fail(command, as_json, error)
    except CredenceGraphError as error:
        _fail(command, as_json, translate(error, command))
    if as_json:
        typer.echo(json.dumps({"command": command, **result.payload}, indent=2, allow_nan=False))
    else:
        for line in result.text:
            typer.echo(line)
    raise typer.Exit(0)


def _fail(command: str, as_json: bool, error: CliError) -> NoReturn:
    """Print an error and exit with status 1.

    Args:
        command: The command's name.
        as_json: Whether to print JSON instead of text.
        error: The error.

    Raises:
        typer.Exit: Always, with status 1.
    """
    if as_json:
        typer.echo(json.dumps({"command": command, "error": error.to_dict()}, indent=2, allow_nan=False))
    else:
        typer.echo(f"error: {error.message}", err=True)
        if error.hint:
            typer.echo(f"hint: {error.hint}", err=True)
    raise typer.Exit(1)


def read_graph(path: Path) -> Graph:
    """Load a graph file.

    Args:
        path: The file.

    Returns:
        The graph.

    Raises:
        CliError: If the file is missing or unreadable, or is not a valid graph.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise CliError(
            FILE_NOT_FOUND, f"no graph file at {str(path)!r}", f"create one with 'credencegraph init {path}'"
        ) from None
    except (OSError, UnicodeDecodeError) as error:
        raise CliError(IO_ERROR, f"cannot read {str(path)!r}: {error}") from None
    try:
        return loads(text)
    except CycleError as error:
        cli_error = translate(error)
        cli_error.code = INVALID_GRAPH
        raise cli_error from None
    except ValidationError as error:
        raise CliError(
            INVALID_GRAPH,
            f"{str(path)!r} is not a valid credencegraph file: {error}",
            "fix the item named in the message; the file was not changed",
        ) from None


def write_graph(graph: Graph, path: Path) -> None:
    """Write a graph file, replacing it in one step so a failure leaves the old file intact.

    Args:
        graph: The graph.
        path: The destination.

    Raises:
        CliError: If the file cannot be written.
    """
    text = dumps(graph) + "\n"
    directory = path.parent
    try:
        handle, temporary = tempfile.mkstemp(dir=directory, prefix=f".{path.name}.", suffix=".tmp")
    except FileNotFoundError as error:
        raise CliError(
            IO_ERROR,
            f"cannot write {str(path)!r}: {error}",
            f"the directory {str(directory)!r} does not exist; create it and run the command again",
        ) from None
    except OSError as error:
        raise CliError(IO_ERROR, f"cannot write {str(path)!r}: {error}") from None
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        Path(temporary).replace(path)
    except OSError as error:
        Path(temporary).unlink(missing_ok=True)
        raise CliError(IO_ERROR, f"cannot write {str(path)!r}: {error}") from None


def _numbers(text: str, count: int) -> list[float] | None:
    """Parse ``count`` comma-separated floats, or return ``None`` if ``text`` is not that."""
    parts = text.split(",")
    if len(parts) != count:
        return None
    try:
        return [float(part) for part in parts]
    except ValueError:
        return None


def parse_credence(text: str, option: str) -> Credence:
    """Parse a credence: a bare probability is a ``Point``, ``beta:A,B`` is a ``Beta``.

    Args:
        text: The command-line value.
        option: The option it came from, used in the error message.

    Returns:
        The credence.

    Raises:
        CliError: If the value is neither form, or its numbers are out of range.
    """
    kind, colon, rest = text.partition(":")
    try:
        if not colon:
            try:
                return Point(float(text))
            except ValueError:
                pass
        elif kind.strip().lower() == "beta":
            numbers = _numbers(rest, 2)
            if numbers is not None:
                return Beta(*numbers)
    except ValidationError as error:
        raise CliError(INVALID_ARGUMENT, f"{option} {text!r}: {error}", f"{option} takes {CREDENCE_FORMS}") from None
    raise CliError(INVALID_ARGUMENT, f"{option} {text!r} is not a credence", f"{option} takes {CREDENCE_FORMS}")


def parse_source(text: str) -> SourceAnchor:
    """Parse a source anchor, ``DOCUMENT`` or ``DOCUMENT#LOCATOR``, splitting at the last ``#``.

    Args:
        text: The command-line value.

    Returns:
        The anchor.

    Raises:
        CliError: If the document or the locator is empty.
    """
    document, hash_sign, locator = text.rpartition("#")
    if not hash_sign:
        document, locator = text, ""
    if not document or (hash_sign and not locator):
        raise CliError(
            INVALID_ARGUMENT,
            f"--source {text!r} needs a document{' and a locator after #' if hash_sign else ''}",
            "--source takes DOCUMENT or DOCUMENT#LOCATOR, such as doi:10.1000/xyz#p4",
        )
    return SourceAnchor(document, locator or None)


_TRUE = {"true", "1", "yes"}
_FALSE = {"false", "0", "no"}


def parse_assignment(items: Iterable[str], option: str, *, default: bool | None = None) -> dict[str, bool]:
    """Parse ``NODE=true`` / ``NODE=false`` items into an assignment.

    Args:
        items: The command-line values.
        option: The option or argument they came from, used in error messages.
        default: The value of an item given as a bare ``NODE``; ``None`` makes the value mandatory.

    Returns:
        Node ids and values, in the order given.

    Raises:
        CliError: If an item is malformed or one node is given both values.
    """
    form = "NODE or NODE=true|false" if default is not None else "NODE=true|false"
    assignment: dict[str, bool] = {}
    for item in items:
        node_id, equals, raw = item.rpartition("=")
        if not equals:
            node_id, raw = item, ""
        if not equals and default is not None and node_id:
            value = default
        elif node_id and raw.strip().lower() in _TRUE:
            value = True
        elif node_id and raw.strip().lower() in _FALSE:
            value = False
        else:
            raise CliError(
                INVALID_ARGUMENT, f"{option} {item!r} is not {form}", f"write it as {form}, such as h1=false"
            )
        if assignment.get(node_id, value) != value:
            raise CliError(INVALID_ARGUMENT, f"{option} gives {node_id!r} both true and false")
        assignment[node_id] = value
    return assignment


def check_consistent(network: Network, evidence: dict[str, bool], interventions: dict[str, bool]) -> None:
    """Refuse values that contradict each other outright, before any inference.

    Nodes merged by ``equivalent`` relations are one proposition, so ``--set`` or ``--given`` must give
    them the same value, and ``--given`` must agree with ``--set`` on a proposition it fixes.

    Args:
        network: The compiled network.
        evidence: The ``--given`` values.
        interventions: The ``--set`` values.

    Raises:
        CliError: If two merged nodes get different values from one flag, or ``--given`` contradicts ``--set``.
    """
    for flag, assignment in (("--set", interventions), ("--given", evidence)):
        seen: dict[int, tuple[str, bool]] = {}
        for node_id, value in assignment.items():
            other, previous = seen.setdefault(network.index(node_id), (node_id, value))
            if previous != value:
                raise CliError(
                    INVALID_ARGUMENT,
                    f"{flag} gives the equivalent nodes {other!r} and {node_id!r} different values",
                    f"equivalent nodes are one proposition; give them the same {flag} value, or pass {flag} for only one",
                )
    fixed = {network.index(node_id): (node_id, value) for node_id, value in interventions.items()}
    for node_id, value in evidence.items():
        other, setting = fixed.get(network.index(node_id), (node_id, value))
        if setting != value:
            merged = "" if other == node_id else f", and {other!r} and {node_id!r} are equivalent"
            raise CliError(
                INVALID_ARGUMENT,
                f"--given {render_item(node_id, value)} contradicts --set {render_item(other, setting)}{merged}",
                "a proposition fixed by --set holds that value; drop the --given, or give it the --set value",
            )


def require_nodes(graph: Graph, node_ids: Iterable[str], role: str) -> None:
    """Check that every id names a node of the graph.

    Args:
        graph: The graph.
        node_ids: The ids to check.
        role: What the ids are for, used in the error message.

    Raises:
        CliError: If an id is not a node, suggesting the closest ones that are.
    """
    for node_id in node_ids:
        if node_id in graph:
            continue
        close = difflib.get_close_matches(node_id, list(graph.nodes), n=3)
        hint = f"did you mean {', '.join(repr(c) for c in close)}?" if close else "add it with 'credencegraph add-node'"
        raise CliError(UNKNOWN_NODE, f"{role} {node_id!r} is not a node of the graph", hint, {"node": node_id})


def require_variables(graph: Graph, node_ids: Iterable[str], role: str) -> None:
    """Check that every id names a node of the graph that takes part in inference.

    Args:
        graph: The graph.
        node_ids: The ids to check.
        role: What the ids are for, such as ``--given``, used in the error message.

    Raises:
        CliError: If an id is not a node, or is a node with no base and no inferential relation.
    """
    node_ids = list(node_ids)
    require_nodes(graph, node_ids, role)
    variables = {node_id for ids in inference_sets(graph).values() for node_id in ids}
    for node_id in node_ids:
        if node_id not in variables:
            raise CliError(
                INVALID_ARGUMENT,
                f"{role} {node_id!r} takes no part in inference: it has no base and no inferential relation",
                f"give {node_id!r} a base or an inferential relation in the graph file, or name another node",
                {"node": node_id},
            )


def require_text(value: str | None, option: str) -> None:
    """Check that an option given on the command line is not empty.

    Args:
        value: The value, or ``None`` when the option was not given.
        option: The option, used in the error message.

    Raises:
        CliError: If the value is the empty string.
    """
    if value is not None and not value:
        raise CliError(INVALID_ARGUMENT, f"{option} must not be empty", f"pass a non-empty {option}")
