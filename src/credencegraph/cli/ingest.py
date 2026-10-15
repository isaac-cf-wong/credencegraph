"""The ``ingest`` command: a graph of document nodes built from a LaTeX document.

The reader is imported only when the command runs, so a core installation starts without the
``ingest`` extra and reports ``missing-extra`` if the command is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from credencegraph.cli.common import (
    FILE_EXISTS,
    FILE_NOT_FOUND,
    INVALID_ARGUMENT,
    IO_ERROR,
    MISSING_EXTRA,
    CliError,
    GraphPath,
    JsonOption,
    Result,
    read_rubric,
    require_text,
    respond,
    write_graph,
)
from credencegraph.ingest.chunks import UNITS, chunks_to_graph


def ingest_command(  # noqa: PLR0913, PLR0917 - Typer maps one parameter to each option
    source: Annotated[Path, typer.Argument(help="The main .tex file of the document.", show_default=False)],
    path: GraphPath,
    rubric: Annotated[
        Path | None,
        typer.Option("--rubric", help="A TOML rubric whose [initial] table types the chunks.", show_default=False),
    ] = None,
    document: Annotated[
        str | None,
        typer.Option(help="The document id of every anchor; SOURCE as given by default.", show_default=False),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Replace the graph file if it already exists.")] = False,
    as_json: JsonOption = False,
) -> None:
    """Write a graph of document nodes for a LaTeX document: its sentences, equations, captions, footnotes and references."""

    def action() -> Result:
        require_text(document, "--document")
        if path.exists() and not force:
            raise CliError(
                FILE_EXISTS,
                f"{str(path)!r} already exists",
                "pass --force to replace it, or choose another path",
                {"path": str(path)},
            )
        if source.suffix.lower() != ".tex":
            raise CliError(
                INVALID_ARGUMENT,
                f"{str(source)!r} is not a LaTeX file: only .tex documents can be ingested",
                "pass the main .tex file of the document",
                {"source": str(source)},
            )
        initial = read_rubric(rubric).initial if rubric is not None else {}
        try:
            from credencegraph.ingest.latex import LatexError, read_latex  # noqa: PLC0415 - needs the extra
        except ImportError:
            raise CliError(
                MISSING_EXTRA,
                "ingesting LaTeX needs the optional 'ingest' dependencies, which are not installed",
                "install them with: pip install 'credencegraph[ingest]'",
            ) from None
        try:
            chunks = read_latex(source)
        except FileNotFoundError as error:
            raise CliError(FILE_NOT_FOUND, f"cannot read {str(source)!r}: {error}") from None
        except (OSError, UnicodeDecodeError) as error:
            raise CliError(IO_ERROR, f"cannot read {str(source)!r}: {error}") from None
        except LatexError as error:
            raise CliError(INVALID_ARGUMENT, f"cannot read {str(source)!r}: {error}") from None
        graph, unresolved = chunks_to_graph(chunks, document or str(source), initial)
        write_graph(graph, path)
        units = {unit: sum(chunk.unit == unit for chunk in chunks) for unit in UNITS}
        payload = {
            "path": str(path),
            "source": str(source),
            "nodes": len(graph),
            "relations": len(graph.relations),
            "units": units,
            "unresolved_citations": unresolved,
        }
        text = [
            f"ingested {source} into {path}: {len(graph)} nodes, {len(graph.relations)} cites relations",
            ", ".join(f"{count} {unit}" for unit, count in units.items()),
        ]
        if unresolved:
            text.append(f"citations naming no bibliography entry: {', '.join(unresolved)}")
        return Result(payload, text)

    respond("ingest", as_json, action)
