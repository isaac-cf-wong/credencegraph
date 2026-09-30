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
from credencegraph.core.serialization import dumps, loads
from credencegraph.inference.errors import InferenceError, ProblemTooLargeError, ZeroProbabilityError
from credencegraph.semantics.errors import CompileError

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
INFERENCE_ERROR = "inference-error"

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
        exit_code: The exit status; a command whose outcome is a verdict, such as ``check``, may set 1.
    """

    payload: dict[str, Any]
    text: list[str]
    exit_code: int = 0


def translate(error: CredenceGraphError, graph_path: Path | None = None) -> CliError:
    """Turn a library error into a ``CliError`` with a code and, where there is one, a remedy.

    Args:
        error: The library error.
        graph_path: The graph file the command worked on, used in hints.

    Returns:
        The error to report.
    """
    check = f"run 'credencegraph check {graph_path}' to list what is missing" if graph_path else None
    if isinstance(error, CycleError):
        return CliError(
            CYCLE,
            str(error),
            "requires, supports and refutes relations must not form a cycle; drop or reverse one of the relations named",
            {"cycle": list(error.cycle), "relations": list(error.relations)},
        )
    if isinstance(error, CompileError):
        return CliError(COMPILE_ERROR, str(error), check)
    if isinstance(error, ZeroProbabilityError):
        return CliError(ZERO_PROBABILITY, str(error), "the evidence cannot all hold at once; drop or change a --given")
    if isinstance(error, ProblemTooLargeError):
        return CliError(
            PROBLEM_TOO_LARGE,
            str(error),
            "the graph is too large for exact inference",
            {"required": error.required, "limit": error.limit},
        )
    if isinstance(error, InferenceError):
        return CliError(INFERENCE_ERROR, str(error))
    return CliError(INVALID_ARGUMENT, str(error))


def respond(command: str, as_json: bool, action: Callable[[], Result], graph_path: Path | None = None) -> NoReturn:
    """Run a command's action and print its result or its error.

    Args:
        command: The command's name, echoed in the JSON output.
        as_json: Whether to print JSON instead of text.
        action: The work of the command.
        graph_path: The graph file the command works on, used in hints.

    Raises:
        typer.Exit: With the result's exit status, or 1 on an error.
    """
    try:
        result = action()
    except CliError as error:
        _fail(command, as_json, error)
    except CredenceGraphError as error:
        _fail(command, as_json, translate(error, graph_path))
    if as_json:
        typer.echo(json.dumps({"command": command, **result.payload}, indent=2, allow_nan=False))
    else:
        for line in result.text:
            typer.echo(line)
    raise typer.Exit(result.exit_code)


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
