"""The ``ingest`` command: a graph of document nodes built from a LaTeX or PDF document.

The readers are imported only when the command runs, so a core installation starts without the
``ingest`` extra and reports ``missing-extra`` if the command is used.
"""

from __future__ import annotations

from enum import StrEnum
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
from credencegraph.ingest.chunks import UNITS, Chunk, chunks_to_graph


class Format(StrEnum):
    """The formats ``ingest`` reads."""

    LATEX = "latex"
    PDF = "pdf"


_EXTENSIONS = {".tex": Format.LATEX, ".pdf": Format.PDF}


def _read(source: Path, source_format: Format) -> list[Chunk]:
    """Read a document with the reader for its format.

    Raises:
        CliError: If the reader's dependencies are missing, or the document cannot be read.
    """
    try:
        if source_format is Format.LATEX:
            from credencegraph.ingest.latex import LatexError as ReadError  # noqa: PLC0415 - needs the extra
            from credencegraph.ingest.latex import read_latex as read  # noqa: PLC0415
        else:
            from credencegraph.ingest.pdf import PdfError as ReadError  # noqa: PLC0415 - needs the extra
            from credencegraph.ingest.pdf import read_pdf as read  # noqa: PLC0415
    except ImportError:
        raise CliError(
            MISSING_EXTRA,
            f"ingesting {source_format.value.upper()} needs the optional 'ingest' dependencies, which are not installed",
            "install them with: pip install 'credencegraph[ingest]'",
        ) from None
    try:
        return read(source)
    except FileNotFoundError as error:
        raise CliError(FILE_NOT_FOUND, f"cannot read {str(source)!r}: {error}") from None
    except (OSError, UnicodeDecodeError) as error:
        raise CliError(IO_ERROR, f"cannot read {str(source)!r}: {error}") from None
    except ReadError as error:
        raise CliError(INVALID_ARGUMENT, f"cannot read {str(source)!r}: {error}") from None


def ingest_command(  # noqa: PLR0913, PLR0917 - Typer maps one parameter to each option
    source: Annotated[
        Path, typer.Argument(help="The main .tex file of the document, or its .pdf file.", show_default=False)
    ],
    path: GraphPath,
    rubric: Annotated[
        Path | None,
        typer.Option("--rubric", help="A TOML rubric whose [initial] table types the chunks.", show_default=False),
    ] = None,
    document: Annotated[
        str | None,
        typer.Option(help="The document id of every anchor; SOURCE as given by default.", show_default=False),
    ] = None,
    source_format: Annotated[
        Format | None,
        typer.Option(
            "--format", help="How to read SOURCE; from its extension by default (.tex, .pdf).", show_default=False
        ),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Replace the graph file if it already exists.")] = False,
    as_json: JsonOption = False,
) -> None:
    """Write a graph of document nodes for a LaTeX or PDF document: its sentences, equations, captions, footnotes and references."""

    def action() -> Result:
        require_text(document, "--document")
        if path.exists() and not force:
            raise CliError(
                FILE_EXISTS,
                f"{str(path)!r} already exists",
                "pass --force to replace it, or choose another path",
                {"path": str(path)},
            )
        chosen = source_format or _EXTENSIONS.get(source.suffix.lower())
        if chosen is None:
            raise CliError(
                INVALID_ARGUMENT,
                f"cannot tell the format of {str(source)!r} from its extension",
                "pass a .tex or .pdf file, or name the format with --format latex|pdf",
                {"source": str(source)},
            )
        initial = read_rubric(rubric).initial if rubric is not None else {}
        chunks = _read(source, chosen)
        graph, unresolved = chunks_to_graph(chunks, document or str(source), initial)
        write_graph(graph, path)
        units = {unit: sum(chunk.unit == unit for chunk in chunks) for unit in UNITS}
        payload = {
            "path": str(path),
            "source": str(source),
            "format": chosen.value,
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
