"""The ``ingest`` command, on the invented papers under ``tests/fixtures/latex`` and ``tests/fixtures/dummy-paper``."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

from credencegraph.core import load

ROOT = Path(__file__).parents[2]
PAPER = ROOT / "tests" / "fixtures" / "latex" / "paper" / "main.tex"
PDF = ROOT / "tests" / "fixtures" / "dummy-paper" / "dummy-paper.pdf"
RUBRIC = ROOT / "docs" / "examples" / "methods-paper.toml"


def test_ingest_then_check(cli, tmp_path):
    """Test that ``ingest`` writes the paper's graph, which passes ``anchored`` and fails ``typed`` only on untyped nodes."""
    graph_path = tmp_path / "paper.json"
    output = cli.json("ingest", PAPER, graph_path, "--document", "toy-paper")
    assert output == {
        "command": "ingest",
        "path": str(graph_path),
        "source": str(PAPER),
        "format": "latex",
        "nodes": 13,
        "relations": 4,
        "units": {"sentence": 7, "equation": 2, "caption": 1, "footnote": 1, "reference": 2},
        "unresolved_citations": [],
    }
    graph = load(graph_path)
    assert {node.sources[0].document for node in graph} == {"toy-paper"}
    assert cli.json("check", graph_path, "--rubric", RUBRIC, "--level", "anchored")["violations"] == []
    error = cli.error("check", graph_path, "--rubric", RUBRIC, "--level", "typed")
    assert {violation["diagnostic"] for violation in error["details"]["violations"]} == {"untyped"}


def test_ingest_with_rubric_types_references(cli, tmp_path):
    """Test that ``--rubric`` gives each unit its ``[initial]`` type and the document id defaults to SOURCE."""
    graph_path = tmp_path / "paper.json"
    cli.json("ingest", PAPER, graph_path, "--rubric", RUBRIC)
    graph = load(graph_path)
    assert [node.id for node in graph if node.kind == "reference"] == ["r-1", "r-2"]
    assert {node.sources[0].document for node in graph} == {str(PAPER)}


def test_ingest_text_output_names_unresolved_citations(cli, tmp_path):
    """Test the text summary, which names citations that match no bibliography entry."""
    source = tmp_path / "doc.tex"
    source.write_text("A claim \\cite{nowhere}.\n", encoding="utf-8")
    result = cli.run("ingest", source, tmp_path / "g.json")
    assert result.exit_code == 0, result.stderr
    assert result.stdout.splitlines() == [
        f"ingested {source} into {tmp_path / 'g.json'}: 1 nodes, 0 cites relations",
        "1 sentence, 0 equation, 0 caption, 0 footnote, 0 reference",
        "citations naming no bibliography entry: nowhere",
    ]


def test_ingest_refuses_to_overwrite(cli, tmp_path):
    """Test that an existing graph file is kept unless ``--force`` is passed."""
    graph_path = tmp_path / "paper.json"
    graph_path.write_text("keep", encoding="utf-8")
    assert cli.error("ingest", PAPER, graph_path)["code"] == "file-exists"
    assert graph_path.read_text(encoding="utf-8") == "keep"
    assert cli.json("ingest", PAPER, graph_path, "--force")["nodes"] == 13


def test_ingest_failures(cli, tmp_path):
    """Test a source of unknown format, a missing source, a missing input file and an input loop."""
    graph_path = tmp_path / "g.json"
    assert cli.error("ingest", tmp_path / "paper.docx", graph_path)["code"] == "invalid-argument"
    assert cli.error("ingest", tmp_path / "absent.tex", graph_path)["code"] == "file-not-found"
    source = tmp_path / "doc.tex"
    source.write_text("Text. \\input{absent}\n", encoding="utf-8")
    error = cli.error("ingest", source, graph_path)
    assert error["code"] == "file-not-found"
    assert "absent" in error["message"]
    source.write_text("Text. \\input{doc}\n", encoding="utf-8")
    assert cli.error("ingest", source, graph_path)["code"] == "invalid-argument"
    assert not graph_path.exists()


def test_ingest_without_the_extra(cli, tmp_path, monkeypatch):
    """Test that without the reader's dependencies the command fails with ``missing-extra`` and the install command."""
    monkeypatch.setitem(sys.modules, "credencegraph.ingest.latex", None)
    error = cli.error("ingest", PAPER, tmp_path / "g.json")
    assert error["code"] == "missing-extra"
    assert "credencegraph[ingest]" in error["hint"]


def test_ingest_pdf(cli, tmp_path):
    """Test that ``ingest`` reads a PDF by its extension into a graph that passes ``anchored``."""
    graph_path = tmp_path / "paper.json"
    output = cli.json("ingest", PDF, graph_path, "--document", "dummy-paper")
    assert output == {
        "command": "ingest",
        "path": str(graph_path),
        "source": str(PDF),
        "format": "pdf",
        "nodes": 29,
        "relations": 6,
        "units": {"sentence": 19, "equation": 5, "caption": 2, "footnote": 0, "reference": 3},
        "unresolved_citations": [],
    }
    graph = load(graph_path)
    assert graph.nodes["s-04"].sources[0].locator.startswith("page=1;bbox=")
    assert cli.json("check", graph_path, "--rubric", RUBRIC, "--level", "anchored")["violations"] == []


def test_ingest_format_option(cli, tmp_path):
    """Test that ``--format`` reads a file whose extension names no format, and overrides one that does."""
    source = tmp_path / "paper.bin"
    source.write_bytes(PDF.read_bytes())
    assert cli.json("ingest", source, tmp_path / "a.json", "--format", "pdf")["nodes"] == 29
    assert cli.error("ingest", source, tmp_path / "b.json")["code"] == "invalid-argument"
    error = cli.error("ingest", PAPER, tmp_path / "c.json", "--format", "pdf")
    assert error["code"] == "invalid-argument"
    assert "PDFium cannot open it" in error["message"]


def test_ingest_pdf_failures(cli, tmp_path):
    """Test a missing PDF, and a PDF reader whose dependencies are missing."""
    assert cli.error("ingest", tmp_path / "absent.pdf", tmp_path / "g.json")["code"] == "file-not-found"
    with mock.patch.dict(sys.modules, {"credencegraph.ingest.pdf": None}):
        error = cli.error("ingest", PDF, tmp_path / "g.json")
    assert error["code"] == "missing-extra"
    assert error["message"].startswith("ingesting PDF needs")
