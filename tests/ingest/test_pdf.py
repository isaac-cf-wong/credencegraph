"""The PDF reader, on the invented paper under ``tests/fixtures/dummy-paper`` and on PDFs written in the tests.

The expected values come from ``dummy-paper.tex``: revtex prints section titles in capitals, numbers
sections with roman numerals and subsections with letters, and labels tables with roman numerals.
"""

from __future__ import annotations

import re
import struct
import zlib
from pathlib import Path

import pytest
from _pdfs import fonts_pdf, form_pdf, page_pdf, standard_font, text_operators, to_unicode_pdf, unmapped_font_pdf

from credencegraph.ingest import chunks_to_graph
from credencegraph.ingest.pdf import REPLACEMENT, PdfError, analyse_pdf, read_pdf, render_region
from credencegraph.rubric import load_rubric
from credencegraph.rubric.validate import check_graph

ROOT = Path(__file__).parents[2]
PAPER = ROOT / "tests" / "fixtures" / "dummy-paper" / "dummy-paper.pdf"
RUBRIC = load_rubric(ROOT / "docs" / "examples" / "methods-paper.toml")
LOCATOR = re.compile(r"page=1;bbox=(-?\d+\.\d),(-?\d+\.\d),(-?\d+\.\d),(-?\d+\.\d)")

# Sentences of the paper, as the PDF prints them: TeX sets the apostrophe of "filter's" as U+2019.
SENTENCES = [
    "We describe a toy smoothing filter for invented tide records.",
    "The filter suppresses short spikes while keeping slow trends, and its bias vanishes for long windows.",
    "We test it on synthetic data only.",
    "Smoothing a noisy record is a familiar task [1].",
    "Earlier filters treat every sample alike [1\N{EN DASH}3], which blurs sharp but genuine features.",
    (
        "In this paper we follow the efficient scheme of H\N{LATIN SMALL LETTER E WITH ACUTE}l"
        "\N{LATIN SMALL LETTER E WITH GRAVE}ne Dupr\N{LATIN SMALL LETTER E WITH ACUTE} [2] and add one refinement."
    ),
    "Let xk be the record at step k and let w > 0 be the window length.",
    "The filter returns",
    "Its bias for a linear trend xk = a + bk is",
    "so the relative bias \N{GREEK SMALL LETTER BETA}/(bw) tends to 1/2 and the absolute bias grows with w.",
    "The refinement splits the window in two halves and weights them separately,",
    "The two halves combine as",
    "which equals Eq. (1) for every even w.",
    "Its variance is still Eq. (3).",
    "We ran the filter on three synthetic records, listed in Table I.",
    "The records are generated without any measurement, so the test says nothing about real tides [3].",
    "Figure 1 shows one of them.",
    (
        "Notwithstanding the simplicity of the records, the filter\N{RIGHT SINGLE QUOTATION MARK}s behaviour is"
        " unremarkable and uninteresting throughout."
    ),
    "No data were measured for this study.",
]
FIGURE_CAPTION = "The smoothed record against the step k for a window w = 8."
TABLE_CAPTION = "The synthetic records and their window lengths."
REFERENCES = [
    "A. Quill, Smoothing invented records, arXiv:9913.00001.",
    (
        "H. Dupr\N{LATIN SMALL LETTER E WITH ACUTE} and B. Ostrander, An efficient toy filter, Imaginary Journal of"
        " Filters 1, 1 (2032), doi:10.5555/toy.2032.001."
    ),
    "C. Nabokova, Synthetic tides, arXiv:9913.00002, doi:10.5555/toy.2033.002.",
]


@pytest.fixture(scope="module")
def paper():
    """The dummy paper, read once."""
    return analyse_pdf(PAPER)


def _box(locator: str) -> tuple[float, ...]:
    match = LOCATOR.fullmatch(locator)
    assert match, locator
    return tuple(float(value) for value in match.groups())


def _inside(inner: tuple[float, ...], outer: tuple[float, ...]) -> bool:
    return (
        outer[0] - 0.1 <= inner[0]
        and outer[1] - 0.1 <= inner[1]
        and inner[2] <= outer[2] + 0.1
        and inner[3] <= outer[3] + 0.1
    )


def _write(tmp_path: Path, data: bytes, name: str = "doc.pdf") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _png_pixels(png: bytes) -> tuple[int, int, bytes]:
    """The width, height and greyscale pixels of a PNG written by ``render_region``."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    length, kind = struct.unpack(">I4s", png[8:16])
    assert kind == b"IHDR"
    width, height, depth, colour = struct.unpack(">IIBB", png[16:26])
    assert (length, depth, colour) == (13, 8, 0)
    data = zlib.decompress(png[png.index(b"IDAT") + 4 :])
    rows = [data[row * (width + 1) : (row + 1) * (width + 1)] for row in range(height)]
    assert all(row[0] == 0 for row in rows)
    return width, height, b"".join(row[1:] for row in rows)


# The dummy paper


def test_title_and_headings(paper):
    """Test the title, the numbered and unnumbered headings and their levels."""
    assert paper.title == "A Toy Smoothing Filter for Invented Tide Records"
    assert [(heading.number, heading.text, heading.level) for heading in paper.headings] == [
        ("I", "INTRODUCTION", 1),
        ("II", "THE FILTER", 1),
        ("A", "Refinement", 2),
        ("III", "TESTS", 1),
        (None, "DATA AVAILABILITY", None),
    ]


def test_sentences_are_found_verbatim(paper):
    """Test every sentence of the paper: abstract, accents, a ligature, hyphenated words and a sentence a float splits."""
    assert [chunk.text for chunk in paper.chunks if chunk.unit == "sentence"] == SENTENCES


def test_words_hyphenated_at_line_ends(paper):
    """Test that the layout breaks words across lines, and that each one is read whole."""
    broken = [line.text.split()[-1] for line in paper.lines if line.glyphs and line.glyphs[-1].hyphen]
    assert broken == ["measure-", "Fig-", "fil-"]
    text = " ".join(chunk.text for chunk in paper.chunks)
    assert all(word in text for word in ("measurement", "Figure 1", "filter,"))


def test_chunk_order(paper):
    """Test the order of the chunks: a display ends the sentence before it; a float's chunks follow the sentence it splits."""
    units = "".join(chunk.unit[0] for chunk in paper.chunks)
    assert units == "ssssssssesessesesssssscescrrr"
    split = [chunk.text for chunk in paper.chunks].index(SENTENCES[17])
    assert [(chunk.unit, chunk.labels) for chunk in paper.chunks[split + 1 : split + 3]] == [
        ("caption", ()),
        ("equation", ("3",)),
    ]


def test_equation_numbers_by_position(paper):
    """Test the equation numbers: the align's two numbers form one display, and (3), set in a float, is found out of order."""
    assert [equation.numbers for equation in paper.equations] == [("1",), ("2",), ("4a", "4b"), ("5",), ("3",)]
    assert [chunk.labels for chunk in paper.chunks if chunk.unit == "equation"] == [
        ("1",),
        ("2",),
        ("4a", "4b"),
        ("5",),
        ("3",),
    ]
    assert paper.equations[4].text == "\N{GREEK SMALL LETTER SIGMA}2y = \N{GREEK SMALL LETTER SIGMA}2 w"
    # The summation sign's font maps it to no Unicode value.
    assert REPLACEMENT in paper.equations[0].text
    # The numbers are right-aligned in their columns: (1), (2), (4a), (4b) and (5) on the left, (3) on the right.
    rights = [round(equation.region[2]) for equation in paper.equations]
    assert rights == [298, 298, 298, 298, 561]


def test_captions(paper):
    """Test the figure and table captions, their numbers as printed, and the regions they caption."""
    figure, table = paper.captions
    assert (figure.kind, figure.number, figure.text) == ("figure", "1", FIGURE_CAPTION)
    assert (table.kind, table.number, table.text) == ("table", "I", TABLE_CAPTION)
    assert [chunk.text for chunk in paper.chunks if chunk.unit == "caption"] == [FIGURE_CAPTION, TABLE_CAPTION]
    # The figure is drawn above its caption, the table below; both in the right-hand column.
    assert figure.region[1] >= figure.box[3]
    assert table.region[3] <= table.box[1]
    assert all(caption.region[0] > 300 for caption in paper.captions)


def test_bibliography(paper):
    """Test the entries, their keys, and the fictitious arXiv identifiers and DOIs in them."""
    assert [reference.number for reference in paper.references] == ["1", "2", "3"]
    assert [reference.text for reference in paper.references] == REFERENCES
    assert [reference.arxiv for reference in paper.references] == [("9913.00001",), (), ("9913.00002",)]
    assert [reference.doi for reference in paper.references] == [
        (),
        ("10.5555/toy.2032.001",),
        ("10.5555/toy.2033.002",),
    ]
    assert [(chunk.key, chunk.text) for chunk in paper.chunks if chunk.unit == "reference"] == list(
        zip(["1", "2", "3"], REFERENCES, strict=True)
    )


def test_citations(paper):
    """Test each citation: its keys, ranges spelled out, and its position inside the sentence that makes it."""
    assert [(citation.text, citation.keys) for citation in paper.citations] == [
        ("[1]", ("1",)),
        ("[1\N{EN DASH}3]", ("1", "2", "3")),
        ("[2]", ("2",)),
        ("[3]", ("3",)),
    ]
    citing = [chunk for chunk in paper.chunks if chunk.cites]
    assert [(chunk.text, chunk.cites) for chunk in citing] == [
        (SENTENCES[3], ("1",)),
        (SENTENCES[4], ("1", "2", "3")),
        (SENTENCES[5], ("2",)),
        (SENTENCES[15], ("3",)),
    ]
    for citation, chunk in zip(paper.citations, citing, strict=True):
        assert citation.page == 1
        assert _inside(citation.box, _box(chunk.locator))


def test_locators(paper):
    """Test that every chunk is located by one page and a box on the A4 page, and blocks and lines cover the page."""
    for chunk in paper.chunks:
        x0, y0, x1, y1 = _box(chunk.locator)
        assert 0 <= x0 < x1 <= 595.3
        assert 0 <= y0 < y1 <= 841.9
    assert {line.page for line in paper.lines} == {1}
    assert sum(len(block.lines) for block in paper.blocks) == len(paper.lines)


def test_graph_of_the_paper_is_anchored(paper):
    """Test that the chunks make a graph that passes ``anchored``, with one ``cites`` relation per cited entry."""
    graph, unresolved = chunks_to_graph(read_pdf(PAPER), "dummy-paper")
    assert unresolved == []
    assert len(graph) == 29
    assert len(graph.relations) == 6
    assert [node.attributes["label"] for node in graph if node.attributes["unit"] == "equation"] == [
        "1",
        "2",
        ("4a", "4b"),
        "5",
        "3",
    ]
    assert check_graph(graph, RUBRIC, "anchored") == []


def test_render_equation_and_figure(paper):
    """Test that an equation's and a figure's regions render to greyscale PNG images of the region's size, with ink in them."""
    equation = paper.equations[0]
    width, height, pixels = _png_pixels(render_region(PAPER, equation.page, equation.region, scale=2))
    x0, y0, x1, y1 = equation.region
    assert abs(width - 2 * (x1 - x0 + 2)) <= 2
    assert abs(height - 2 * (y1 - y0 + 2)) <= 2
    assert min(pixels) < 64
    assert max(pixels) == 255
    figure = paper.captions[0]
    _, _, pixels = _png_pixels(render_region(PAPER, figure.page, figure.region))
    assert min(pixels) < 64


@pytest.mark.parametrize(
    ("media_box", "rotate"), [((0, 0, 400, 400), 0), ((0, 0, 400, 400), 90), ((-50, -50, 350, 350), 270)]
)
def test_render_a_rotated_or_offset_page(tmp_path, media_box, rotate):
    """Test that a line's box renders the line on a page with a /Rotate or a MediaBox whose origin is not (0, 0)."""
    content = text_operators([(b"Some text here to render", 40, 300, 0)])
    path = _write(tmp_path, page_pdf(content, media_box=media_box, rotate=rotate))
    line = analyse_pdf(path).lines[0]
    assert line.text == "Some text here to render"
    _, _, pixels = _png_pixels(render_region(path, line.page, line.box))
    assert min(pixels) < 64


def test_figure_with_nothing_above_reaches_the_top_of_the_page(tmp_path):
    """Test that a figure region with no text or drawing above its caption reaches the top of the crop box.

    The page is 400 wide and 300 high, turned by /Rotate: its top is at y = 300 whatever way it is shown.
    """
    content = text_operators([(b"Figure 1. A caption with nothing drawn above it.", 40, 100, 0)])
    path = _write(tmp_path, page_pdf(content, media_box=(0, 0, 400, 300), rotate=90))
    (caption,) = analyse_pdf(path).captions
    assert (caption.kind, caption.number, caption.text) == ("figure", "1", "A caption with nothing drawn above it.")
    assert caption.region == (caption.box[0], caption.box[3], caption.box[2], 300.0)


def test_render_rejects_a_missing_page_or_an_empty_region():
    """Test the errors of ``render_region``."""
    with pytest.raises(ValueError, match="page 2 does not exist"):
        render_region(PAPER, 2, (0, 0, 10, 10))
    with pytest.raises(ValueError, match="is empty"):
        render_region(PAPER, 1, (900, 900, 950, 950))


# PDFs written in the tests


def test_page_wrapped_in_a_form_xobject(tmp_path):
    """Test text inside a form XObject drawn at half size: it is read, and located on the page, not in the form."""
    content = text_operators(
        [
            (b"A paragraph set inside a form XObject, which", 40, 300, 0),
            (b"some submission systems wrap every page in.", 40, 288, 0),
            (b"Its second sentence is short.", 40, 276, 0),
        ]
    )
    chunks = read_pdf(_write(tmp_path, form_pdf(content, (0.5, 0, 0, 0.5, 100, 50))))
    assert [chunk.text for chunk in chunks] == [
        "A paragraph set inside a form XObject, which some submission systems wrap every page in.",
        "Its second sentence is short.",
    ]
    # Drawn at (100 + x / 2, 50 + y / 2): the first line starts near x = 120 and its baseline is at y = 200.
    x0, y0, x1, y1 = _box(chunks[0].locator)
    assert 119 < x0 < 121
    assert 192 < y0 < y1 < 205
    assert x1 < 100 + 400 / 2


def test_font_without_a_unicode_map(tmp_path):
    """Test that glyphs a font maps to no Unicode value read as U+FFFD, spaces staying spaces, and make no chunk."""
    document = analyse_pdf(_write(tmp_path, unmapped_font_pdf(text_operators([(b"Ok \x01\x02 ab", 50, 100, 0)]))))
    assert [line.text for line in document.lines] == [" ".join([REPLACEMENT * 2] * 3)]
    assert document.chunks == ()


def test_ligature_and_control_character(tmp_path):
    """Test that a ligature from a ToUnicode map is spelled out and a control character is dropped."""
    content = text_operators([(b"The \x01rst \x02line.", 50, 100, 0)])
    pdf = to_unicode_pdf(content, {1: "\N{LATIN SMALL LIGATURE FI}", 2: "\a"})
    assert [chunk.text for chunk in read_pdf(_write(tmp_path, pdf))] == ["The first line."]


def test_hyphenation_at_the_end_of_a_line(tmp_path):
    """Test that a word broken by a hyphen at the end of a line is joined, and one before a capital is not."""
    content = text_operators(
        [
            (b"The filter needs a calibra-", 40, 300, 0),
            (b"tion of the noise, as Quill-", 40, 288, 0),
            (b"Nabokova found.", 40, 276, 0),
        ]
    )
    assert [chunk.text for chunk in read_pdf(_write(tmp_path, page_pdf(content)))] == [
        "The filter needs a calibration of the noise, as Quill- Nabokova found."
    ]


def test_front_matter_wrapped_heading_and_unnumbered_display(tmp_path):
    """Test a title, an author line kept out of the chunks, a heading on two lines, and a display with no number.

    Courier sets every character equally wide, so lines of one length share both edges and make a column.
    """
    abstract = [
        b"This abstract is set in two lines of equal width.",
        b"Both are fifty characters long, so they align ok.",
    ]
    body = [
        b"The method runs in three steps that are each fast",
        b"and each one is checked against the one before it",
        b"in the order the steps are run, so the sum reads",
    ]
    content = b"\n".join(
        [
            text_operators([(b"A Generated Test Document", 60, 360, 0)], size=16, font=b"F2"),
            text_operators([(b"Ann Example", 150, 340, 0)]),
            text_operators([(abstract[0], 40, 320, 0), (abstract[1], 40, 308, 0)]),
            text_operators([(b"2.1 A heading that runs onto", 40, 290, 0), (b"a second line", 40, 278, 0)], font=b"F3"),
            text_operators([(body[0], 40, 260, 0), (body[1], 40, 248, 0), (body[2], 40, 236, 0)]),
            text_operators([(b"x = y + z", 150, 215, 0)]),
            text_operators([(b"which ends the paragraph.", 40, 195, 0)]),
        ]
    )
    fonts = [standard_font(b"Courier"), standard_font(b"Helvetica-Bold"), standard_font(b"Courier-Bold")]
    document = analyse_pdf(_write(tmp_path, fonts_pdf(content, fonts)))
    assert document.title == "A Generated Test Document"
    assert [(heading.number, heading.text, heading.level) for heading in document.headings] == [
        ("2.1", "A heading that runs onto a second line", 2)
    ]
    assert [(chunk.unit, chunk.text, chunk.labels) for chunk in document.chunks] == [
        ("sentence", abstract[0].decode(), ()),
        ("sentence", abstract[1].decode(), ()),
        ("sentence", b" ".join(body).decode(), ()),
        ("equation", "x = y + z", ()),
        ("sentence", "which ends the paragraph.", ()),
    ]
    assert document.equations[0].numbers == ()


def test_rotated_text(tmp_path):
    """Test text set at 90 degrees: read along its baseline, while an upright line on the same page is marginal."""
    content = text_operators(
        [
            (b"Rotated text is read", 100, 50, 90),
            (b"along its baseline.", 115, 50, 90),
            (b"stamp", 300, 20, 0),
        ]
    )
    document = analyse_pdf(_write(tmp_path, page_pdf(content)))
    assert [(line.text, line.angle) for line in document.lines] == [
        ("Rotated text is read", 90),
        ("along its baseline.", 90),
        ("stamp", 0),
    ]
    assert [chunk.text for chunk in document.chunks] == ["Rotated text is read along its baseline."]
    x0, y0, x1, y1 = _box(document.chunks[0].locator)
    # The lines' baselines are at x = 100 and x = 115, and their text runs up from y = 50.
    assert x0 < 100
    assert x1 > 115
    assert y0 < 60
    assert y1 > 120


def test_unreadable_documents(tmp_path):
    """Test a missing file and a file that is not a PDF."""
    with pytest.raises(FileNotFoundError):
        read_pdf(tmp_path / "absent.pdf")
    with pytest.raises(PdfError, match="PDFium cannot open it"):
        read_pdf(_write(tmp_path, b"not a PDF"))
