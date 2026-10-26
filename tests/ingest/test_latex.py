"""The LaTeX reader and the graph built from its chunks, on invented documents under ``tests/fixtures/latex``."""

from __future__ import annotations

from pathlib import Path

import pytest

from credencegraph.ingest import Chunk, chunks_to_graph
from credencegraph.ingest.latex import LatexError, read_latex
from credencegraph.rubric import load_rubric, text_digest
from credencegraph.rubric.validate import check_graph

ROOT = Path(__file__).parents[2]
PAPER = ROOT / "tests" / "fixtures" / "latex" / "paper" / "main.tex"
RUBRIC = load_rubric(ROOT / "docs" / "examples" / "methods-paper.toml")

# Every chunk of the invented paper, in document order: main.tex inputs macros.tex in its preamble and
# sections/method.tex in its body, and its bibliography is main.bbl.
PAPER_CHUNKS = [
    Chunk("sentence", r"We study a toy estimator of the mean $\mu \in \mathbb{R}$.", "file=main.tex;lines=10-10"),
    Chunk(
        "sentence",
        r"Estimating a mean is an old problem~\cite{gauss1809}.",
        "file=main.tex;lines=14-14",
        cites=("gauss1809",),
    ),
    Chunk(
        "sentence",
        r"We revisit it, e.g.\ for short segments, and we show that the bias vanishes.",
        "file=main.tex;lines=15-15",
    ),
    Chunk("footnote", r"The proof is in Sec.~\ref{sec:method}.", "file=main.tex;lines=15-15"),
    Chunk("sentence", "Is it consistent?", "file=main.tex;lines=16-16"),
    Chunk(
        "sentence",
        r"The answer is yes \citep{fisher1925,gauss1809}.",
        "file=main.tex;lines=16-16",
        cites=("fisher1925", "gauss1809"),
    ),
    Chunk(
        "sentence",
        r"The estimator averages the samples $\mathbf{x}$, as in Eq.~\eqref{eq:mean}.",
        "file=sections/method.tex;lines=2-2",
    ),
    Chunk(
        "equation",
        r"\begin{equation} \hat{\mu} = \frac{x_1 + x_2}{2} \end{equation}",
        "file=sections/method.tex;lines=3-6",
        labels=("eq:mean",),
    ),
    Chunk("sentence", "Its variance follows directly:", "file=sections/method.tex;lines=7-7"),
    Chunk(
        "equation",
        r"\[ \operatorname{Var}\hat{\mu} = \frac{\sigma^2}{2} \]",
        "file=sections/method.tex;lines=8-10",
    ),
    Chunk(
        "caption",
        r"The bias of $\hat{\mu}$ against the segment length, as in Ref.~\cite{fisher1925}.",
        "file=sections/method.tex;lines=13-13",
        cites=("fisher1925",),
    ),
    Chunk(
        "reference",
        r"R.~A. Fisher. An invented book on statistical methods. Toy Press, 1925.",
        "file=main.bbl;lines=3-6",
        key="fisher1925",
    ),
    Chunk(
        "reference",
        r"C.~F. Gauss. An invented treatise on least squares. 1809.",
        "file=main.bbl;lines=8-11",
        key="gauss1809",
    ),
]


def read(tmp_path: Path, body: str, preamble: str = "", **files: str) -> list[Chunk]:
    """Read a document written to ``tmp_path``: ``main.tex`` with the preamble and body, and other files."""
    for name, content in files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    main = tmp_path / "main.tex"
    main.write_text(f"\\documentclass{{article}}\n{preamble}\n\\begin{{document}}\n{body}\n\\end{{document}}\n")
    return read_latex(main)


def lines(*rows: str) -> str:
    """Join source lines."""
    return "\n".join(rows)


def texts(chunks: list[Chunk], unit: str | None = None) -> list[str]:
    """The texts of the chunks, or of those of one unit."""
    return [chunk.text for chunk in chunks if unit is None or chunk.unit == unit]


def test_paper_chunks():
    """Test every chunk of the invented paper: text, unit, file and lines, citations, labels and keys."""
    assert read_latex(PAPER) == PAPER_CHUNKS


def test_multi_file_input(tmp_path):
    r"""Test that ``\input`` and ``\include`` are followed, with or without ``.tex``, and located in their own file."""
    (tmp_path / "sec").mkdir()
    chunks = read(
        tmp_path,
        "First in main.\n\\input{sec/a}\n\\include{b.tex}\nLast in main.",
        **{"sec/a.tex": "\n\nIn file a.\n", "b.tex": "In file b."},
    )
    assert [(chunk.text, chunk.locator) for chunk in chunks] == [
        ("First in main.", "file=main.tex;lines=4-4"),
        ("In file a.", "file=sec/a.tex;lines=3-3"),
        ("In file b.", "file=b.tex;lines=1-1"),
        ("Last in main.", "file=main.tex;lines=7-7"),
    ]


def test_missing_input(tmp_path):
    r"""Test that an ``\input`` naming no file fails, naming it."""
    with pytest.raises(FileNotFoundError, match="absent"):
        read(tmp_path, "Text. \\input{absent}")


def test_input_loop(tmp_path):
    """Test that a document that inputs itself fails instead of recursing forever."""
    with pytest.raises(LatexError, match="inputs itself"):
        read(tmp_path, "Text. \\input{loop}", **{"loop.tex": "\\input{loop}"})


def test_unclosed_def(tmp_path):
    r"""Test that a ``\def`` whose body is never closed fails."""
    with pytest.raises(LatexError, match="never closed"):
        read(tmp_path, "Text.", r"\def\broken{open")


def test_macros(tmp_path):
    r"""Test ``\newcommand`` with and without parameters and with an optional one, ``\def``, and nesting."""
    preamble = lines(
        r"\newcommand{\R}{\mathbb{R}}",
        r"\newcommand\norm[2][2]{\lVert #2 \rVert_{#1}}",
        r"\def\half#1{\frac{#1}{2}}",
        r"\newcommand{\halfnorm}[1]{\half{\norm{#1}}}",
        r"\providecommand{\R}{\mathbf{R}}",
    )
    chunks = read(tmp_path, r"Take $x \in \R$ with $\norm{x}$, $\norm[1]{x}$ and $\halfnorm{x}$ finite.", preamble)
    assert texts(chunks) == [
        (
            r"Take $x \in \mathbb{R}$ with $\lVert x \rVert_{2}$, $\lVert x \rVert_{1}$ and "
            r"$\frac{\lVert x \rVert_{2}}{2}$ finite."
        )
    ]


def test_commented_out_definitions_define_nothing(tmp_path):
    r"""Test that a ``\def`` or ``\newcommand`` in a comment, whole-line or trailing, leaves its macro as written."""
    preamble = lines(
        r"% \def\foo{WRONG}",
        r"\newcommand{\bar}{RIGHT} % \def\bar{WRONG}",
        r"%\newcommand{\baz}{WRONG}",
    )
    chunks = read(tmp_path, r"\foo, \bar and \baz appear.", preamble)
    assert texts(chunks) == [r"\foo, RIGHT and \baz appear."]


def test_footnotes_inside_groups_and_macros(tmp_path):
    r"""Test that a ``\footnote`` inside a group, a macro argument or a macro's body is a footnote chunk, not text."""
    preamble = r"\newcommand{\note}[1]{\footnote{#1}}"
    body = lines(
        r"A \emph{small\footnote{Hidden note \cite{k}.}} claim.",
        r"A {grouped\footnote{Grouped note.}} and a defined\note{Macro note.} one.",
    )
    chunks = read(tmp_path, body, preamble)
    assert [(chunk.unit, chunk.text, chunk.locator, chunk.cites) for chunk in chunks] == [
        ("sentence", r"A \emph{small} claim.", "file=main.tex;lines=4-4", ()),
        ("footnote", r"Hidden note \cite{k}.", "file=main.tex;lines=4-4", ("k",)),
        ("sentence", "A {grouped} and a defined one.", "file=main.tex;lines=5-5", ()),
        ("footnote", "Grouped note.", "file=main.tex;lines=5-5", ()),
        ("footnote", "Macro note.", "file=main.tex;lines=5-5", ()),
    ]


def test_macros_inside_other_macros_arguments(tmp_path):
    r"""Test that a user macro inside the argument of a macro that is not user-defined is expanded, spacing kept."""
    chunks = read(tmp_path, r"See \emph{$\R$} and \textbf [x]{\R}.", r"\newcommand{\R}{\mathbb{R}}")
    assert texts(chunks) == [r"See \emph{$\mathbb{R}$} and \textbf [x]{\mathbb{R}}."]


def test_macro_defined_in_the_body_and_recursive(tmp_path):
    r"""Test that a body ``\def`` adds no text, and that a macro defined in terms of itself stops expanding."""
    body = r"\def\k{K}A \k{} sentence. \newcommand{\loop}{\loop}Then \loop."
    chunks = read(tmp_path, body)
    assert texts(chunks) == ["A K{} sentence.", r"Then \loop."]


def test_equations(tmp_path):
    """Test labelled and unlabelled displays of every delimiter, one chunk per display, labels removed from the text."""
    body = lines(
        "Before.",
        r"\begin{equation} a = b \label{eq:one} \end{equation}",
        r"\begin{equation*} c = d \end{equation*}",
        r"\[ e = f \]",
        "$$ g = h $$",
        r"\begin{align} i &= j \label{eq:i} \\ k &= l \label{eq:k} \end{align}",
        "where $a$ is inline.",
    )
    chunks = read(tmp_path, body)
    assert [(chunk.unit, chunk.text, chunk.labels) for chunk in chunks] == [
        ("sentence", "Before.", ()),
        ("equation", r"\begin{equation} a = b \end{equation}", ("eq:one",)),
        ("equation", r"\begin{equation*} c = d \end{equation*}", ()),
        ("equation", r"\[ e = f \]", ()),
        ("equation", "$$ g = h $$", ()),
        ("equation", r"\begin{align} i &= j \\ k &= l \end{align}", ("eq:i", "eq:k")),
        ("sentence", "where $a$ is inline.", ()),
    ]


def test_sentence_splitting(tmp_path):
    """Test that sentences end at . ? ! and paragraph breaks, but not after abbreviations, initials or before lower case."""
    body = lines(
        "Dr. Smith and J. Doe found it, i.e. the bound, and it holds for $n = 1$. Does it hold",
        "for all $n$? Yes! See Fig. 2 and Eqs. 3 and 4, cf. the appendix. it continues here.",
        "% a comment between paragraphs",
        "",
        "A paragraph without a full stop",
        "",
        r"A sentence.\footnote{Its note.} Next sentence.",
    )
    chunks = read(tmp_path, body)
    assert [(chunk.unit, chunk.text) for chunk in chunks] == [
        ("sentence", "Dr. Smith and J. Doe found it, i.e. the bound, and it holds for $n = 1$."),
        ("sentence", "Does it hold for all $n$?"),
        ("sentence", "Yes!"),
        ("sentence", "See Fig. 2 and Eqs. 3 and 4, cf. the appendix. it continues here."),
        ("sentence", "A paragraph without a full stop"),
        ("sentence", "A sentence."),
        ("footnote", "Its note."),
        ("sentence", "Next sentence."),
    ]
    assert chunks[1].locator == "file=main.tex;lines=4-5"


def test_structure_is_not_text(tmp_path):
    """Test that headings, list items, comments, labels and tables end sentences and add no text of their own."""
    body = lines(
        r"\section{Intro}\label{sec:intro}",
        "First point % trailing comment",
        r"\begin{itemize}\item One item \item Two items\end{itemize}",
        r"\begin{tabular}{cc} 1 & 2 \end{tabular}",
        r"\begin{table}\caption[short]{A table caption.}\begin{tabular}{c} x \end{tabular}\end{table}",
    )
    chunks = read(tmp_path, body)
    assert [(chunk.unit, chunk.text) for chunk in chunks] == [
        ("sentence", "First point"),
        ("sentence", "One item"),
        ("sentence", "Two items"),
        ("caption", "A table caption."),
    ]


def test_thebibliography_and_citations(tmp_path):
    r"""Test a ``thebibliography`` in the document, citation keys from every cite command, and an unresolved key."""
    body = lines(
        r"One \cite{a}. Two \citet[p.~2]{b, a}. Three \cite{missing}.",
        r"\begin{thebibliography}{9}",
        r"\bibitem[A(2001)]{a} A. Author, \emph{Invented}, 2001.",
        r"\bibitem{b} B. Author,",
        r"\newblock Also invented, 2002.",
        r"\end{thebibliography}",
    )
    chunks = read(tmp_path, body)
    assert [(chunk.unit, chunk.text, chunk.cites, chunk.key) for chunk in chunks] == [
        ("sentence", r"One \cite{a}.", ("a",), None),
        ("sentence", r"Two \citet[p.~2]{b, a}.", ("b", "a"), None),
        ("sentence", r"Three \cite{missing}.", ("missing",), None),
        ("reference", r"A. Author, \emph{Invented}, 2001.", (), "a"),
        ("reference", "B. Author, Also invented, 2002.", (), "b"),
    ]
    graph, unresolved = chunks_to_graph(chunks, "doc")
    assert unresolved == ["missing"]
    assert [(r.id, r.type, r.source, r.target) for r in graph.relations.values()] == [
        ("s-1-cites-r-1", "cites", "s-1", "r-1"),
        ("s-2-cites-r-2", "cites", "s-2", "r-2"),
        ("s-2-cites-r-1", "cites", "s-2", "r-1"),
    ]


def test_bibliography_without_bbl(tmp_path):
    r"""Test that ``\bibliography`` without a ``.bbl`` beside the main file adds no reference, leaving citations unresolved."""
    chunks = read(tmp_path, "Cited \\cite{a}.\n\\bibliography{refs}")
    assert texts(chunks) == [r"Cited \cite{a}."]
    assert chunks_to_graph(chunks, "doc")[1] == ["a"]


def test_paper_graph():
    """Test the graph of the invented paper: ids, anchors, attributes and the cites relations."""
    graph, unresolved = chunks_to_graph(read_latex(PAPER), "paper.tex")
    assert unresolved == []
    assert list(graph.nodes) == [f"s-{i}" for i in range(1, 8)] + ["e-1", "s-8", "e-2", "s-9", "r-1", "r-2"]
    equation = graph.nodes["e-1"]
    assert dict(equation.attributes) == {"origin": "document", "unit": "equation", "label": "eq:mean"}
    assert dict(graph.nodes["e-2"].attributes) == {"origin": "document", "unit": "equation"}
    assert dict(graph.nodes["r-2"].attributes) == {"origin": "document", "unit": "reference", "key": "gauss1809"}
    anchor = equation.sources[0]
    assert (anchor.document, anchor.locator, anchor.quote) == (
        "paper.tex",
        "file=sections/method.tex;lines=3-6",
        equation.statement,
    )
    assert anchor.digest == text_digest(equation.statement)
    assert [(r.source, r.target) for r in graph.relations.values()] == [
        ("s-2", "r-2"),
        ("s-6", "r-1"),
        ("s-6", "r-2"),
        ("s-9", "r-1"),
    ]


def test_paper_passes_typed_checks():
    """Test that the paper's graph passes the ``typed`` checks but for untyped nodes, and with ``[initial]`` types."""
    chunks = read_latex(PAPER)
    graph, _ = chunks_to_graph(chunks, "paper.tex")
    assert check_graph(graph, RUBRIC, "anchored") == []
    findings = check_graph(graph, RUBRIC, "typed")
    assert [finding.id for finding in findings] == [f"untyped:{node}" for node in graph.nodes]
    typed, _ = chunks_to_graph(chunks, "paper.tex", RUBRIC.initial)
    assert [node.kind for node in typed if node.kind != "unassigned"] == ["reference", "reference"]
    assert {finding.id for finding in check_graph(typed, RUBRIC, "typed")} == {
        f"untyped:{node.id}" for node in typed if node.kind == "unassigned"
    }


def test_ids_are_padded_to_the_width_needed():
    """Test that ids are numbered per prefix and zero-padded to the width of that prefix's count."""
    chunks = [Chunk("sentence", f"Sentence {i}.", f"file=m.tex;lines={i}-{i}") for i in range(10)]
    chunks.insert(3, Chunk("equation", "$$ x $$", "file=m.tex;lines=3-3"))
    graph, _ = chunks_to_graph(chunks, "doc")
    assert list(graph.nodes)[:5] == ["s-01", "s-02", "s-03", "e-1", "s-04"]
    assert list(graph.nodes)[-1] == "s-10"


@pytest.mark.parametrize("quantity", ["0.6 K", "$0.6$ K", r"\(0.6\) K", r"0.6\,K", "0.6~K"])
def test_a_unit_after_a_numeral_ends_the_sentence(tmp_path, quantity):
    """Test that "0.6 K." ends a sentence however the numeral and its unit are written, but "C. Lee" does not."""
    body = f"Over 30 days the mean drift is {quantity}. Lee reports drifts of this size, as C. Lee found."
    assert texts(read(tmp_path, body)) == [
        f"Over 30 days the mean drift is {quantity}.",
        "Lee reports drifts of this size, as C. Lee found.",
    ]


def test_a_label_after_a_spacing_macro_ends_the_sentence(tmp_path):
    r"""Test that the word before a capital is read across a macro, so "Table\,I." is a label."""
    assert texts(read(tmp_path, r"Listed in Table\,I. The records follow.")) == [
        r"Listed in Table\,I.",
        "The records follow.",
    ]


@pytest.mark.parametrize("name", ["J. R. R. Tolkien", r"J.\,R.\,R. Tolkien", "J.~R.~R. Tolkien"])
def test_initials_separated_by_spacing_macros_do_not_end_the_sentence(tmp_path, name):
    """Test that initials stay in their sentence however the space between them is written."""
    assert texts(read(tmp_path, f"As {name} wrote, it holds. Next.")) == [f"As {name} wrote, it holds.", "Next."]
