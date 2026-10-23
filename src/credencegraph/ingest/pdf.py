r"""The PDF reader: a ``.pdf`` document as chunks located by page and bounding box.

The reader is built on PDFium's per-character text API (through ``pypdfium2``): every character
comes with its Unicode value, its box, its font name and its size. Text inside form XObjects,
including pages that a submission system wraps whole, is read like any other text. From the
characters the reader builds

- **lines**: characters on one baseline, in content-stream order, cut where the text jumps by more
  than two font sizes along the baseline. Rotated text is read along its own baseline; lines at
  an angle other than the page's dominant one (such as a margin stamp) are kept as lines but read
  into no chunk.
- **blocks**: consecutive lines of one size, each below the last by at most 1.5 font sizes.

Cleanup happens on the way: ligatures (U+FB00 to U+FB06) are spelled out, an accent set as its own
glyph over or under a letter is combined with it, control characters are dropped, and a character
the font gives no Unicode value for becomes U+FFFD. A word hyphenated at the end of a line is
joined when the next line starts in lower case.

The structure is found from position and font, not from reading order:

- the **title**: the lines in the largest font at the top of the first page; the lines below it
  up to the abstract (the first block of at least two aligned lines) are front matter, and are
  read into no chunk;
- **headings**: short lines in a bold font (such as ``CMBX``), with their numbered label
  (``II.``, ``A.``, ``3.1``) split off;
- **captions**: blocks starting ``FIG. n``, ``Figure n``, ``TABLE n`` or ``Table n``;
- **equation numbers**: ``(n)`` or ``(na)`` set flush right in a text column, alone on a line or
  at the end of a display line, whatever their order in the content stream;
- the **bibliography**: from the last line starting ``[1]``, one entry per ``[n]``, with its arXiv
  identifiers and DOIs;
- **citations**: ``[n]``, ``[n, m]`` and ``[n-m]`` in running text.

Running text is the lines that start at the left edge of a text column (a column is a left and
right edge shared by at least three lines) and share their baseline with nothing else in it. The
other lines between two lines of running text form a display: an equation if it holds an equation
number or mathematics, a float if it holds a caption. A display ends the sentence before it; a
float does not, since floats land wherever the layout puts them, and its chunks follow the
sentence it interrupts.

Locators are ``page=<n>;bbox=<x0>,<y0>,<x1>,<y1>``, in PDF points with the origin at the bottom
left, one pair per page a chunk spans. ``render_region`` renders an equation's or a figure's
region to PNG, the anchor for mathematics the text cannot carry.
"""

from __future__ import annotations

import ctypes
import itertools
import math
import re
import struct
import unicodedata
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from credencegraph.ingest.chunks import CAPTION, EQUATION, REFERENCE, SENTENCE, Chunk, ends_sentence
from credencegraph.rubric.text import normalise_text

# A box in PDF points: left, bottom, right, top.
Box = tuple[float, float, float, float]

REPLACEMENT = "\N{REPLACEMENT CHARACTER}"

_LIGATURES = {
    "\N{LATIN SMALL LIGATURE FF}": "ff",
    "\N{LATIN SMALL LIGATURE FI}": "fi",
    "\N{LATIN SMALL LIGATURE FL}": "fl",
    "\N{LATIN SMALL LIGATURE FFI}": "ffi",
    "\N{LATIN SMALL LIGATURE FFL}": "ffl",
    "\N{LATIN SMALL LIGATURE LONG S T}": "st",
    "\N{LATIN SMALL LIGATURE ST}": "st",
}
# Accents that a TeX font sets as glyphs of their own, and the combining marks they stand for.
_ACCENTS = {
    "\N{ACUTE ACCENT}": "\N{COMBINING ACUTE ACCENT}",
    "\N{GRAVE ACCENT}": "\N{COMBINING GRAVE ACCENT}",
    "\N{DIAERESIS}": "\N{COMBINING DIAERESIS}",
    "\N{MODIFIER LETTER CIRCUMFLEX ACCENT}": "\N{COMBINING CIRCUMFLEX ACCENT}",
    "\N{CIRCUMFLEX ACCENT}": "\N{COMBINING CIRCUMFLEX ACCENT}",
    "\N{SMALL TILDE}": "\N{COMBINING TILDE}",
    "\N{TILDE}": "\N{COMBINING TILDE}",
    "\N{MACRON}": "\N{COMBINING MACRON}",
    "\N{MODIFIER LETTER MACRON}": "\N{COMBINING MACRON}",
    "\N{BREVE}": "\N{COMBINING BREVE}",
    "\N{DOT ABOVE}": "\N{COMBINING DOT ABOVE}",
    "\N{RING ABOVE}": "\N{COMBINING RING ABOVE}",
    "\N{DOUBLE ACUTE ACCENT}": "\N{COMBINING DOUBLE ACUTE ACCENT}",
    "\N{CARON}": "\N{COMBINING CARON}",
    "\N{CEDILLA}": "\N{COMBINING CEDILLA}",
    "\N{OGONEK}": "\N{COMBINING OGONEK}",
}
_SUBSET = re.compile(r"^[A-Z]{6}\+")
_BOLD = re.compile(r"bold|black|heavy|demi|BX", re.IGNORECASE)
_BOLD_WEIGHT = 600
_EQUATION_NUMBER = re.compile(r"(?:^|\s)\(([A-Z]?\d+[a-z]?)\)\s*$")
_CAPTION = re.compile(r"^(FIG\.|Fig\.|FIGURE|Figure|TABLE|Table)\s*(\d+|[IVXLC]+)\s*[.:]\s*")
_HEADING_LABEL = re.compile(r"^((?:\d+\.)*\d+|[IVXLC]+|[A-Z])\.?\s+(?=\S)")
_ENTRY = re.compile(r"^\[(\d+)\]\s*")
_DASH = "-\N{EN DASH}"
_CITATION = re.compile(rf"\[(\d+(?:\s*[{_DASH}]\s*\d+)?(?:\s*,\s*\d+(?:\s*[{_DASH}]\s*\d+)?)*)\]")
_ARXIV = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)")
_DOI = re.compile(r"\b(10\.\d{4,9}/\S+)")
_SENTENCE_END = re.compile("[.?!][')\\]\"\N{RIGHT SINGLE QUOTATION MARK}\N{RIGHT DOUBLE QUOTATION MARK}]*(?=\\s)")
_MATHEMATICS = re.compile(
    "[=<>+^/\N{MINUS SIGN}\N{PLUS-MINUS SIGN}\N{MULTIPLICATION SIGN}\N{N-ARY SUMMATION}\N{INTEGRAL}\N{LESS-THAN OR EQUAL TO}\N{GREATER-THAN OR EQUAL TO}\N{ALMOST EQUAL TO}]"
)

# Geometric tolerances, in font sizes unless named in points.
_BASELINE_SHIFT = 0.5  # a character this far off the line's baseline starts a new line
_LINE_GAP = 2.0  # a jump this far along the baseline starts a new line
_BLOCK_STEP = 1.5  # a block's next line is at most this far below the last
_INDENT = 2.5  # running text starts at most this far right of its column's left edge
_EDGE = 2.0  # points: lines this close to an edge are on it
_EQUATION_REACH = 3.0  # a display line belongs to the nearest equation number within this distance
_TITLE_SIZE = 1.15  # the title is set at least this much larger than running text
_HEADING_LENGTH = 100  # a heading is at most this many characters
_COLUMN_LINES = 3  # a column is an edge pair at least this many lines share
_COLUMN_LENGTH = 20  # only lines at least this long define columns
_PAD = 1.0  # points around a rendered region
_SIZE_TOLERANCE = 0.5  # points: font sizes this close are the same
_LARGER = 1.2  # a character this much larger than the first of its line sets the line's baseline
_LONGEST_RANGE = 500  # a citation range spanning more entries than this is read as its two ends
_HIGH_SURROGATES = range(0xD800, 0xDC00)
_LOW_SURROGATES = range(0xDC00, 0xE000)


class PdfError(Exception):
    """A PDF document cannot be read: it is not a PDF, it is damaged, or it needs a password."""


@dataclass(frozen=True, slots=True)
class Glyph:
    """A character as the reader keeps it, after cleanup.

    Attributes:
        text: The character's text: usually one character, two or three for a ligature, a letter
            with its combining accent, or U+FFFD when the font maps the glyph to no Unicode value.
        box: Its box on the page, or ``None`` for a space PDFium inserted between words.
        hyphen: Whether PDFium marks it as a hyphen that breaks a word at the end of a line.
    """

    text: str
    box: Box | None
    hyphen: bool = False


@dataclass(frozen=True, slots=True)
class Line:
    """Characters on one baseline, in content-stream order.

    Attributes:
        page: The page number, from 1.
        text: The line's text.
        glyphs: Its glyphs.
        box: The box around its glyphs.
        font: The font most of its characters are set in, without a subset prefix.
        size: The font size most of its characters are set in.
        bold: Whether that font is bold.
        angle: The baseline's angle, in degrees counterclockwise, rounded to a degree.
        start: Where the line starts along the baseline direction.
        end: Where it ends along the baseline direction.
        baseline: The baseline's position across the baseline direction, larger further up.
    """

    page: int
    text: str
    glyphs: tuple[Glyph, ...]
    box: Box
    font: str
    size: float
    bold: bool
    angle: int
    start: float
    end: float
    baseline: float


@dataclass(frozen=True, slots=True)
class Block:
    """Consecutive lines of one size, each just below the last.

    Attributes:
        page: The page number, from 1.
        lines: The lines, in order.
        box: The box around them.
    """

    page: int
    lines: tuple[Line, ...]
    box: Box


@dataclass(frozen=True, slots=True)
class Heading:
    """A section heading.

    Attributes:
        text: The heading without its label, such as ``INTRODUCTION``.
        number: Its label, such as ``II``, ``A`` or ``3.1``, or ``None`` for an unnumbered heading.
        level: 1 for a roman numeral, 2 for a capital letter, the number of parts for ``3.1``, or
            ``None`` for an unnumbered heading. A single letter that is also a roman numeral, such
            as ``C`` or ``V``, counts as a roman numeral when the heading is in capitals.
        page: The page number, from 1.
        box: The box around it.
    """

    text: str
    number: str | None
    level: int | None
    page: int
    box: Box


@dataclass(frozen=True, slots=True)
class Caption:
    """A figure or table caption.

    Attributes:
        kind: ``figure`` or ``table``.
        number: Its number as printed, such as ``1`` or ``IV``.
        text: The caption without its label.
        page: The page number, from 1.
        box: The box around the caption.
        region: The box of the figure or table it captions: the part of its column between the
            caption and the nearest text that is not part of the float, tightened to what is drawn
            there.
    """

    kind: str
    number: str
    text: str
    page: int
    box: Box
    region: Box


@dataclass(frozen=True, slots=True)
class Equation:
    """A display equation.

    Attributes:
        numbers: Its equation numbers, such as ``("4a", "4b")`` for a two-line display, or ``()``.
        text: The text of its lines, numbers left out; mathematics extracts only roughly.
        page: The page number, from 1.
        region: The box around its lines and numbers.
    """

    numbers: tuple[str, ...]
    text: str
    page: int
    region: Box


@dataclass(frozen=True, slots=True)
class Reference:
    """A bibliography entry.

    Attributes:
        number: Its number, the ``n`` of ``[n]``.
        text: The entry without its number.
        arxiv: The arXiv identifiers in it.
        doi: The DOIs in it.
        locator: Where it is.
    """

    number: str
    text: str
    arxiv: tuple[str, ...]
    doi: tuple[str, ...]
    locator: str


@dataclass(frozen=True, slots=True)
class Citation:
    """A citation in running text.

    Attributes:
        text: The citation as printed, such as ``[1-3]``.
        keys: The entry numbers it cites, ranges spelled out.
        page: The page number of its opening bracket, from 1.
        box: The box around it on that page.
    """

    text: str
    keys: tuple[str, ...]
    page: int
    box: Box


@dataclass(frozen=True, slots=True)
class PdfDocument:
    """What the reader found in a PDF document.

    Attributes:
        title: The title, or ``None`` if no line is set larger than running text.
        headings: The section headings, in reading order.
        captions: The captions, in reading order.
        equations: The display equations, in reading order.
        references: The bibliography entries, in order.
        citations: The citations in running text, in reading order.
        chunks: The chunks, in reading order.
        lines: Every line, page by page in content-stream order.
        blocks: Every block, in the same order.
    """

    title: str | None
    headings: tuple[Heading, ...]
    captions: tuple[Caption, ...]
    equations: tuple[Equation, ...]
    references: tuple[Reference, ...]
    citations: tuple[Citation, ...]
    chunks: tuple[Chunk, ...]
    lines: tuple[Line, ...]
    blocks: tuple[Block, ...]


# Characters and lines


@dataclass(slots=True)
class _Char:
    """A character as PDFium reports it, with what the line builder needs."""

    glyph: Glyph
    font: str
    size: float
    bold: bool
    angle: int
    start: float
    end: float
    baseline: float


def _union(boxes: list[Box]) -> Box:
    """The smallest box containing every box."""
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _font(textpage: pdfium_c.FPDF_TEXTPAGE, index: int, buffer: ctypes.Array) -> tuple[str, bool]:
    """The name of a character's font without its subset prefix, and whether the font is bold."""
    flags = ctypes.c_int()
    size = pdfium_c.FPDFText_GetFontInfo(textpage, index, buffer, len(buffer), ctypes.byref(flags))
    name = _SUBSET.sub("", buffer.value.decode("utf-8", "replace")) if size else ""
    bold = bool(_BOLD.search(name)) or pdfium_c.FPDFText_GetFontWeight(textpage, index) >= _BOLD_WEIGHT
    return name, bold


def _text(code: int, *, unmapped: bool, hyphen: bool) -> str:
    """The cleaned text of one character code: unmapped glyphs become U+FFFD, controls are dropped."""
    if hyphen:
        return "-"
    if unmapped and code != ord(" "):
        return REPLACEMENT
    char = chr(code)
    if unicodedata.category(char) == "Cc":
        return " " if char.isspace() else ""
    return _LIGATURES.get(char, char)


def _characters(page: pdfium.PdfPage) -> list[_Char | None]:
    """The characters of a page in content-stream order; ``None`` marks a space PDFium inserted."""
    textpage = page.get_textpage()
    raw = textpage.raw
    buffer = ctypes.create_string_buffer(256)
    chars: list[_Char | None] = []
    count = pdfium_c.FPDFText_CountChars(raw)
    index = 0
    while index < count:
        code = pdfium_c.FPDFText_GetUnicode(raw, index)
        if pdfium_c.FPDFText_IsGenerated(raw, index):
            if code == ord(" "):
                chars.append(None)
            index += 1
            continue
        if code in _HIGH_SURROGATES and index + 1 < count:
            low = pdfium_c.FPDFText_GetUnicode(raw, index + 1)
            if low in _LOW_SURROGATES:
                code = 0x10000 + ((code - _HIGH_SURROGATES.start) << 10) + (low - _LOW_SURROGATES.start)
                index += 1
        hyphen = bool(pdfium_c.FPDFText_IsHyphen(raw, index)) or code == ord("\N{SOFT HYPHEN}")
        text = _text(code, unmapped=bool(pdfium_c.FPDFText_HasUnicodeMapError(raw, index)), hyphen=hyphen)
        if not text:
            index += 1
            continue
        left, bottom, right, top = textpage.get_charbox(index)
        x, y = ctypes.c_double(), ctypes.c_double()
        pdfium_c.FPDFText_GetCharOrigin(raw, index, ctypes.byref(x), ctypes.byref(y))
        clockwise = pdfium_c.FPDFText_GetCharAngle(raw, index)
        angle = round(math.degrees(-clockwise)) % 360 if clockwise >= 0 else 0
        theta = math.radians(angle)
        cos, sin = math.cos(theta), math.sin(theta)
        along = [cx * cos + cy * sin for cx in (left, right) for cy in (bottom, top)]
        font, bold = _font(raw, index, buffer)
        chars.append(
            _Char(
                Glyph(text, (left, bottom, right, top), hyphen),
                font,
                pdfium_c.FPDFText_GetFontSize(raw, index),
                bold,
                angle,
                min(along),
                max(along),
                -x.value * sin + y.value * cos,
            )
        )
        index += 1
    textpage.close()
    return chars


def _combine_accents(chars: list[_Char | None]) -> list[_Char | None]:
    """Combine each accent set as a glyph of its own with the letter next to it that it sits over or under."""
    dropped: set[int] = set()
    for index, char in enumerate(chars):
        if char is None or char.glyph.text not in _ACCENTS or char.end <= char.start:
            continue
        for other_index in (index + 1, index - 1):
            other = chars[other_index] if 0 <= other_index < len(chars) and other_index not in dropped else None
            if other is None or not other.glyph.text.isalpha():
                continue
            if min(char.end, other.end) - max(char.start, other.start) >= (char.end - char.start) / 2:
                text = unicodedata.normalize("NFC", other.glyph.text + _ACCENTS[char.glyph.text])
                box = _union([char.glyph.box, other.glyph.box])  # type: ignore[list-item]
                other.glyph = Glyph(text, box, other.glyph.hyphen)
                dropped.add(index)
                break
    return [char for index, char in enumerate(chars) if index not in dropped]


def _make_line(page: int, chars: list[_Char | None]) -> Line | None:
    """A line of characters, or ``None`` if it has no visible character."""
    while chars and chars[-1] is None:
        chars.pop()
    visible = [char for char in chars if char is not None]
    if not visible:
        return None
    (font, bold), _ = Counter((char.font, char.bold) for char in visible).most_common(1)[0]
    size, _ = Counter(round(char.size, 2) for char in visible).most_common(1)[0]
    main = next(char for char in visible if round(char.size, 2) == size)
    glyphs = tuple(Glyph(" ", None) if char is None else char.glyph for char in chars)
    return Line(
        page,
        "".join(glyph.text for glyph in glyphs),
        glyphs,
        _union([char.glyph.box for char in visible]),  # type: ignore[misc]
        font,
        size,
        bold,
        main.angle,
        min(char.start for char in visible),
        max(char.end for char in visible),
        main.baseline,
    )


def _lines(page_number: int, chars: list[_Char | None]) -> list[Line]:
    """Cut a page's characters into lines."""
    lines: list[Line] = []
    current: list[_Char | None] = []
    reference: _Char | None = None
    end = 0.0
    for char in chars:
        if char is None:
            if current:
                current.append(None)
            continue
        if reference is not None:
            size = max(char.size, reference.size)
            if (
                char.angle != reference.angle
                or abs(char.baseline - reference.baseline) > _BASELINE_SHIFT * size
                or char.start - end > _LINE_GAP * size
                or end - char.start > _LINE_GAP * size
            ):
                line = _make_line(page_number, current)
                if line is not None:
                    lines.append(line)
                current, reference = [], None
        if reference is None:
            reference, end = char, char.end
        elif char.size > reference.size * _LARGER:
            reference = char
        current.append(char)
        end = max(end, char.end)
    line = _make_line(page_number, current)
    if line is not None:
        lines.append(line)
    return lines


def _blocks(lines: list[Line]) -> list[Block]:
    """Group consecutive lines of one size, each at most 1.5 font sizes below the last, into blocks."""
    blocks: list[list[Line]] = []
    for line in lines:
        previous = blocks[-1][-1] if blocks else None
        if (
            previous is not None
            and previous.page == line.page
            and previous.angle == line.angle
            and abs(previous.size - line.size) <= _SIZE_TOLERANCE
            and 0 < previous.baseline - line.baseline <= _BLOCK_STEP * line.size
            and min(previous.end, line.end) > max(previous.start, line.start)
        ):
            blocks[-1].append(line)
        else:
            blocks.append([line])
    return [Block(group[0].page, tuple(group), _union([line.box for line in group])) for group in blocks]


# Layout: columns, and what each line is

_TITLE = "title"
_FRONT = "front"
_MARGINAL = "marginal"
_PROSE = "prose"
_HEADING = "heading"
_CAPTION_LINE = "caption"
_CAPTION_MORE = "caption-continued"
_NUMBER = "number"
_DISPLAY = "display"
_ENTRY_LINE = "reference"


def _clusters(values: list[float]) -> list[list[float]]:
    """Group sorted values into runs whose members lie within ``_EDGE`` of the run's first value."""
    clusters: list[list[float]] = []
    for value in sorted(values):
        if clusters and value - clusters[-1][0] <= _EDGE:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return clusters


def _columns(lines: list[Line]) -> dict[int, list[tuple[float, float]]]:
    """The text columns of each baseline angle: left and right edges that at least three long lines share."""
    columns: dict[int, list[tuple[float, float]]] = {}
    for angle in sorted({line.angle for line in lines}):
        long = [line for line in lines if line.angle == angle and len(line.text.strip()) >= _COLUMN_LENGTH]
        starts = {
            value: index for index, cluster in enumerate(_clusters([line.start for line in long])) for value in cluster
        }
        ends = {
            value: index for index, cluster in enumerate(_clusters([line.end for line in long])) for value in cluster
        }
        members: dict[tuple[int, int], list[Line]] = {}
        for line in long:
            members.setdefault((starts[line.start], ends[line.end]), []).append(line)
        columns[angle] = sorted(
            (min(line.start for line in group), max(line.end for line in group))
            for group in members.values()
            if len(group) >= _COLUMN_LINES
        )
    return columns


def _column(line: Line, columns: dict[int, list[tuple[float, float]]]) -> tuple[float, float] | None:
    """The narrowest column a line lies within, or ``None``."""
    fitting = [
        column
        for column in columns.get(line.angle, [])
        if column[0] - _EDGE <= line.start and line.end <= column[1] + _EDGE
    ]
    return min(fitting, key=lambda column: column[1] - column[0], default=None)


def _alphanumeric(text: str) -> bool:
    """Whether a text reads as words: it has a letter, and at least 40% of its visible characters are letters or digits."""
    visible = [char for char in text if not char.isspace()]
    return any(char.isalpha() for char in visible) and sum(char.isalnum() for char in visible) >= 0.4 * len(visible)


@dataclass
class _Layout:
    """The lines of a document and what each one is."""

    lines: list[Line]
    blocks: list[Block]
    kinds: list[str]
    columns: dict[int, list[tuple[float, float]]]
    body: float
    title: str | None


def _front_matter(lines: list[Line], kinds: list[str], block_of: dict[int, Block], body: float) -> str | None:
    """Mark the title, and the front matter between it and the abstract or the first heading; return the title.

    The title is the first run of lines in the largest font of the first page, if that font is
    larger than running text. The abstract is the first block after it of at least two lines, at
    least half as wide as the widest line of the page, whose first two lines share an edge.
    """
    first = [index for index, line in enumerate(lines) if line.page == lines[0].page and not kinds[index]]
    largest = max((lines[index].size for index in first), default=0.0)
    if not body or largest < _TITLE_SIZE * body:
        return None
    title_lines: list[int] = []
    for index in first:
        if abs(lines[index].size - largest) <= _SIZE_TOLERANCE:
            title_lines.append(index)
        elif title_lines:
            break
    for index in title_lines:
        kinds[index] = _TITLE
    widest = max(lines[index].end - lines[index].start for index in first)
    for index in (index for index in first if index > title_lines[-1]):
        line = lines[index]
        if line.bold and len(line.text.strip()) <= _HEADING_LENGTH:
            break
        block = block_of[id(line)]
        if (
            len(block.lines) > 1
            and block.lines[0] is line
            and line.end - line.start >= widest / 2
            and (abs(block.lines[1].end - line.end) <= _EDGE or abs(block.lines[1].start - line.start) <= _EDGE)
        ):
            for offset in range(len(block.lines)):
                kinds[index + offset] = _PROSE
            break
        kinds[index] = _FRONT
    return normalise_text(" ".join(lines[index].text for index in title_lines))


def _kind(line: Line, column: tuple[float, float] | None, *, first_of_block: bool, shared: bool) -> str:
    """What a line of the body is: a heading, a caption, an equation number, running text or part of a display."""
    text = line.text.strip()
    if line.bold and len(text) <= _HEADING_LENGTH and any(char.isalpha() for char in text):
        return _HEADING
    if _CAPTION.match(text) and first_of_block:
        return _CAPTION_LINE
    if (
        _EQUATION_NUMBER.search(text)
        and column is not None
        and abs(line.end - column[1]) <= _EDGE
        and (text.startswith("(") or line.start > column[0] + _INDENT * line.size)
    ):
        return _NUMBER
    if _alphanumeric(text) and not shared and (column is None or line.start <= column[0] + _INDENT * line.size):
        return _PROSE
    return _DISPLAY


def _shared_baselines(lines: list[Line], kinds: list[str], columns: dict[int, list[tuple[float, float]]]) -> set[int]:
    """The unclassified lines that share a baseline with another in their column: table cells or pieces of a display."""
    by_place: dict[tuple[int, int, tuple[float, float] | None], list[int]] = {}
    for index, line in enumerate(lines):
        if not kinds[index]:
            by_place.setdefault((line.page, line.angle, _column(line, columns)), []).append(index)
    shared = set()
    for place in by_place.values():
        place.sort(key=lambda index: lines[index].baseline)
        for first, second in itertools.pairwise(place):
            if lines[second].baseline - lines[first].baseline < _SIZE_TOLERANCE:
                shared.update((first, second))
    return shared


def _layout(lines: list[Line], blocks: list[Block]) -> _Layout:
    """Classify every line: title, front matter, heading, caption, equation number, reference, prose or display."""
    kinds = [""] * len(lines)
    block_of = {id(line): block for block in blocks for line in block.lines}
    sizes: Counter[float] = Counter()
    for line in lines:
        sizes[line.size] += len(line.text.strip())
    body = sizes.most_common(1)[0][0] if sizes else 0.0

    # Lines at an angle other than their page's dominant one are marginal.
    angles: dict[int, Counter[int]] = {}
    for line in lines:
        angles.setdefault(line.page, Counter())[line.angle] += len(line.text.strip())
    for index, line in enumerate(lines):
        if line.angle != angles[line.page].most_common(1)[0][0]:
            kinds[index] = _MARGINAL
    upright = [line for line, kind in zip(lines, kinds, strict=True) if kind != _MARGINAL]
    columns = _columns(upright)

    title = _front_matter(lines, kinds, block_of, body)

    # The bibliography: everything from the last line that starts with [1].
    starts = [index for index, line in enumerate(lines) if not kinds[index] and line.text.lstrip().startswith("[1]")]
    if starts:
        for index in range(starts[-1], len(lines)):
            if not kinds[index]:
                kinds[index] = _ENTRY_LINE

    shared = _shared_baselines(lines, kinds, columns)

    for index, line in enumerate(lines):
        if kinds[index]:
            continue
        block = block_of[id(line)]
        kinds[index] = _kind(
            line, _column(line, columns), first_of_block=block.lines[0] is line, shared=index in shared
        )
        if kinds[index] == _CAPTION_LINE:
            for offset in range(1, len(block.lines)):
                kinds[index + offset] = _CAPTION_MORE
    return _Layout(lines, blocks, kinds, columns, body, title)


# Reading: running text, displays, floats and the bibliography


@dataclass(slots=True)
class _Piece:
    """A piece of running text: a glyph's text, and where the glyph is."""

    text: str
    page: int
    box: Box | None
    hyphen: bool = False


def _pieces(line: Line) -> list[_Piece]:
    """The pieces of a line's text."""
    return [_Piece(glyph.text, line.page, glyph.box, glyph.hyphen) for glyph in line.glyphs]


def _join(pieces: list[_Piece], line: Line) -> None:
    """Append a line's pieces to running text, joining a word hyphenated across the line break."""
    following = line.text.lstrip()
    while pieces and pieces[-1].box is None:
        pieces.pop()
    if pieces and following[:1].islower() and pieces[-1].text == "-":
        if pieces[-1].hyphen:
            pieces.pop()
    elif pieces:
        pieces.append(_Piece(" ", line.page, None))
    pieces.extend(_pieces(line))


def _locator(pieces: list[_Piece]) -> str:
    """The locator of some pieces: a page and the box around its pieces, for each page they are on."""
    boxes: dict[int, list[Box]] = {}
    for piece in pieces:
        if piece.box is not None:
            boxes.setdefault(piece.page, []).append(piece.box)
    return ";".join(f"page={page};bbox={_format_box(_union(page_boxes))}" for page, page_boxes in boxes.items())


def _format_box(box: Box) -> str:
    """A box as ``x0,y0,x1,y1`` to a tenth of a point."""
    return ",".join(f"{value:.1f}" for value in box)


def _owners(pieces: list[_Piece]) -> list[int]:
    """For each character of the pieces' joined text, the index of its piece."""
    return [index for index, piece in enumerate(pieces) for _ in piece.text]


def _keys(citation: str) -> tuple[str, ...]:
    """The entry numbers a citation names, ranges such as ``1-3`` spelled out."""
    keys: list[str] = []
    for part in citation.split(","):
        low, _, high = re.sub(rf"\s*[{_DASH}]\s*", "-", part.strip()).partition("-")
        first, last = int(low), int(high or low)
        numbers = range(first, last + 1) if 0 <= last - first <= _LONGEST_RANGE else (first, last)
        keys.extend(str(number) for number in numbers if str(number) not in keys)
    return tuple(keys)


@dataclass
class _Sentence:
    """A sentence, before its citations are checked against the bibliography."""

    text: str
    locator: str
    citations: list[Citation]


@dataclass
class _Reader:
    """Read a classified document into chunks and structure."""

    layout: _Layout
    page_objects: dict[int, list[Box]]
    page_boxes: dict[int, Box]
    items: list[_Sentence | Chunk] = field(default_factory=list)
    running: list[_Piece] = field(default_factory=list)
    pending: list[tuple[int, Chunk]] = field(default_factory=list)
    stretch: list[int] = field(default_factory=list)
    headings: list[Heading] = field(default_factory=list)
    captions: list[tuple[Caption, int, list[int]]] = field(default_factory=list)
    equations: list[Equation] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)

    def read(self) -> None:
        """Read every line in order."""
        kinds = self.layout.kinds
        entries: list[int] = []
        for index, kind in enumerate(kinds):
            if kind == _PROSE:
                self.close_stretch()
                _join(self.running, self.layout.lines[index])
                self.cut(final=False)
            elif kind == _HEADING:
                self.close_stretch()
                self.cut(final=True)
                self.heading(index)
            elif kind == _ENTRY_LINE:
                entries.append(index)
            elif kind in {_CAPTION_LINE, _CAPTION_MORE, _NUMBER, _DISPLAY}:
                self.stretch.append(index)
        self.close_stretch()
        self.cut(final=True)
        self.bibliography(entries)

    # Running text

    def cut(self, *, final: bool) -> None:
        """Cut the complete sentences off the running text; with ``final``, the rest too."""
        while True:
            text = "".join(piece.text for piece in self.running)
            end = None
            for match in _SENTENCE_END.finditer(text):
                rest = text[match.end() :].lstrip()
                if rest and ends_sentence(text[: match.start()], rest):
                    end = match.end()
                    break
            if end is None:
                if not final:
                    return
                end = len(text)
            count = _owners(self.running)[end - 1] + 1 if end else 0
            self.sentence(self.running[:count], text[:end])
            while count < len(self.running) and self.running[count].box is None:
                count += 1
            self.running = self.running[count:]
            # A float waiting at or before the cut follows the sentence just added.
            waiting = [(at - count, chunk) for at, chunk in self.pending]
            self.items.extend(chunk for at, chunk in waiting if at <= 0 or not self.running)
            self.pending = [(at, chunk) for at, chunk in waiting if at > 0 and self.running]
            if not self.running:
                return

    def sentence(self, pieces: list[_Piece], text: str) -> None:
        """Add a sentence, with the citations in it."""
        if not normalise_text(text) or not any(char.isalnum() for char in text):
            return
        owners = _owners(pieces)
        citations = []
        for match in _CITATION.finditer(text):
            span = [pieces[owner] for owner in owners[match.start() : match.end()]]
            boxes = [piece.box for piece in span if piece.box is not None and piece.page == span[0].page]
            citations.append(Citation(match[0], _keys(match[1]), span[0].page, _union(boxes)))  # type: ignore[arg-type]
        self.items.append(_Sentence(normalise_text(text), _locator(pieces), citations))

    def emit(self, chunk: Chunk, *, interrupts: bool) -> None:
        """Add a display's or a float's chunk; a float's waits for the sentence it interrupts to end."""
        open_sentence = "".join(piece.text for piece in self.running).strip()
        if interrupts and open_sentence:
            self.pending.append((len(self.running), chunk))
        else:
            self.items.append(chunk)

    # Headings

    def heading(self, index: int) -> None:
        """Add a heading, or extend the one just before it when it continues on this line."""
        line = self.layout.lines[index]
        previous = self.layout.lines[index - 1] if index else None
        if (
            self.headings
            and previous is not None
            and self.layout.kinds[index - 1] == _HEADING
            and previous.font == line.font
            and 0 < previous.baseline - line.baseline <= _BLOCK_STEP * line.size
            and not _HEADING_LABEL.match(line.text.strip())
        ):
            last = self.headings[-1]
            self.headings[-1] = Heading(
                normalise_text(f"{last.text} {line.text}"),
                last.number,
                last.level,
                last.page,
                _union([last.box, line.box]),
            )
            return
        text = normalise_text(line.text)
        match = _HEADING_LABEL.match(text)
        number = level = None
        if match:
            number = match[1]
            text = text[match.end() :]
            if number[0].isdigit():
                level = number.count(".") + 1
            elif set(number) <= set("IVXLC") and (len(number) > 1 or text.isupper()):
                level = 1
            else:
                level = 2
        self.headings.append(Heading(text, number, level, line.page, line.box))

    # Displays and floats

    def same_column(self, first: int, second: int) -> bool:
        """Whether two lines are on one page and in one column."""
        lines, columns = self.layout.lines, self.layout.columns
        return lines[first].page == lines[second].page and _column(lines[first], columns) == _column(
            lines[second], columns
        )

    def number_groups(self, stretch: list[int]) -> list[list[int]]:
        """Group a stretch's equation-number lines: numbers in one column with no caption between are one display."""
        kinds = self.layout.kinds
        groups: list[list[int]] = []
        for index in (index for index in stretch if kinds[index] == _NUMBER):
            previous = groups[-1][-1] if groups else None
            if (
                previous is not None
                and self.same_column(previous, index)
                and not any(kinds[other] in {_CAPTION_LINE, _CAPTION_MORE} for other in range(previous + 1, index))
            ):
                groups[-1].append(index)
            else:
                groups.append([index])
        return groups

    def close_stretch(self) -> None:
        """Read the lines between two lines of running text: equations, captions and float content.

        Each display line goes to the equation whose number is nearest across the baseline in its
        column; in a float, only within ``_EQUATION_REACH`` font sizes, the rest being the float's
        content. Outside a float, display lines with mathematics but no number are one equation.
        """
        stretch, self.stretch = self.stretch, []
        if not stretch:
            return
        lines, kinds = self.layout.lines, self.layout.kinds
        is_float = any(kinds[index] == _CAPTION_LINE for index in stretch)
        if not is_float:
            self.cut(final=True)
        groups = self.number_groups(stretch)
        members: list[list[int]] = [list(group) for group in groups]
        leftover = []
        for index in (index for index in stretch if kinds[index] == _DISPLAY):
            nearest = min(
                (
                    (abs(lines[number].baseline - lines[index].baseline), position)
                    for position, group in enumerate(groups)
                    for number in group
                    if self.same_column(number, index)
                ),
                default=None,
            )
            if nearest is not None and (not is_float or nearest[0] <= _EQUATION_REACH * self.layout.body):
                members[nearest[1]].append(index)
            else:
                leftover.append(index)
        # Equations and captions are added in the order their first lines come in.
        parts: list[tuple[list[int], list[int] | None]] = [
            (sorted(member), group) for member, group in zip(members, groups, strict=True)
        ]
        if not is_float and leftover and _MATHEMATICS.search(" ".join(lines[index].text for index in leftover)):
            parts.append((leftover, []))
        parts.extend(([index], None) for index in stretch if kinds[index] == _CAPTION_LINE)
        for indices, numbers in sorted(parts, key=lambda part: part[0][0]):
            if numbers is None:
                self.caption(indices[0], leftover)
            else:
                self.equation(indices, numbers, is_float=is_float)

    def equation(self, indices: list[int], numbers: list[int], *, is_float: bool) -> None:
        """Add an equation made of some display lines and the lines that hold its numbers."""
        lines = self.layout.lines
        labels = []
        texts = []
        for index in indices:
            text = lines[index].text
            if index in numbers:
                match = _EQUATION_NUMBER.search(text)
                labels.append(match[1])  # type: ignore[index]
                text = text[: match.start()]  # type: ignore[union-attr]
            texts.append(text)
        page = lines[numbers[0] if numbers else indices[0]].page
        region = _union([lines[index].box for index in indices if lines[index].page == page])
        text = normalise_text(" ".join(texts))
        self.equations.append(Equation(tuple(labels), text, page, region))
        if text:
            chunk = Chunk(EQUATION, text, f"page={page};bbox={_format_box(region)}", labels=tuple(labels))
            self.emit(chunk, interrupts=is_float)

    def caption(self, index: int, content: list[int]) -> None:
        """Add a caption: the caption line and the lines of its block after it, with the float's other content."""
        lines, kinds = self.layout.lines, self.layout.kinds
        own = [index]
        for other in range(index + 1, len(lines)):
            if kinds[other] != _CAPTION_MORE:
                break
            own.append(other)
        pieces: list[_Piece] = []
        for other in own:
            _join(pieces, lines[other])
        text = "".join(piece.text for piece in pieces)
        match = _CAPTION.match(text.lstrip())
        offset = len(text) - len(text.lstrip()) + match.end()  # type: ignore[union-attr]
        body = normalise_text(text[offset:])
        kind = "figure" if match[1].upper().startswith("FIG") else "table"  # type: ignore[index]
        box = _union([lines[other].box for other in own])
        caption = Caption(kind, match[2], body, lines[index].page, box, box)  # type: ignore[index]
        self.captions.append((caption, index, content))
        owners = _owners(pieces)
        chunk_pieces = pieces[owners[offset] :] if offset < len(owners) else []
        if body:
            self.emit(Chunk(CAPTION, body, _locator(chunk_pieces)), interrupts=True)

    def regions(self) -> list[Caption]:
        """The captions with their regions, now that every line is read."""
        lines = self.layout.lines
        result = []
        for caption, index, content in self.captions:
            line = lines[index]
            column = _column(line, self.layout.columns) or (line.box[0], line.box[2])
            _, page_bottom, _, page_top = self.page_boxes[caption.page]
            blocking = [
                other
                for other, kind in enumerate(self.layout.kinds)
                if lines[other].page == caption.page
                and kind not in {_MARGINAL, _FRONT}
                and other not in content
                and other != index
                and lines[other].box[2] > column[0]
                and lines[other].box[0] < column[1]
            ]
            if caption.kind == "figure":
                above = [lines[other].box[1] for other in blocking if lines[other].box[1] >= caption.box[3]]
                zone = (column[0], caption.box[3], column[1], min(above, default=page_top))
            else:
                below = [lines[other].box[3] for other in blocking if lines[other].box[3] <= caption.box[1]]
                zone = (column[0], max(below, default=page_bottom), column[1], caption.box[1])
            drawn = [lines[other].box for other in content] + self.page_objects.get(caption.page, [])
            inside = [
                (max(box[0], zone[0]), max(box[1], zone[1]), min(box[2], zone[2]), min(box[3], zone[3]))
                for box in drawn
                if zone[0] <= (box[0] + box[2]) / 2 <= zone[2] and zone[1] <= (box[1] + box[3]) / 2 <= zone[3]
            ]
            region = _union(inside) if inside else zone
            result.append(Caption(caption.kind, caption.number, caption.text, caption.page, caption.box, region))
        return result

    # The bibliography

    def bibliography(self, indices: list[int]) -> None:
        """Add a reference per ``[n]`` entry of the bibliography."""
        lines = self.layout.lines
        entries: list[list[int]] = []
        for index in indices:
            if _ENTRY.match(lines[index].text.lstrip()) or not entries:
                entries.append([index])
            else:
                entries[-1].append(index)
        for entry in entries:
            pieces: list[_Piece] = []
            for index in entry:
                _join(pieces, lines[index])
            text = "".join(piece.text for piece in pieces)
            match = _ENTRY.match(text.lstrip())
            if match is None:
                continue
            offset = len(text) - len(text.lstrip()) + match.end()
            body = normalise_text(text[offset:])
            locator = _locator(pieces[_owners(pieces)[offset] :]) if offset < len(text) else _locator(pieces)
            arxiv = tuple(found[1] for found in _ARXIV.finditer(body))
            doi = tuple(found[1].rstrip(".,;)") for found in _DOI.finditer(body))
            self.references.append(Reference(match[1], body, arxiv, doi, locator))
            if body:
                self.items.append(Chunk(REFERENCE, body, locator, key=match[1]))

    def chunks(self) -> tuple[list[Chunk], list[Citation]]:
        """The chunks in order, and the citations; a bracket naming a number with no entry is no citation."""
        numbers = {reference.number for reference in self.references}
        chunks: list[Chunk] = []
        citations: list[Citation] = []
        for item in self.items:
            if isinstance(item, Chunk):
                chunks.append(item)
                continue
            kept = [c for c in item.citations if not numbers or set(c.keys) <= numbers]
            citations.extend(kept)
            cites = tuple(dict.fromkeys(key for citation in kept for key in citation.keys))
            chunks.append(Chunk(SENTENCE, item.text, item.locator, cites=cites))
        return chunks, citations


# Reading a document


def _open(path: Path) -> pdfium.PdfDocument:
    """Open a PDF document.

    Raises:
        FileNotFoundError: If the file does not exist.
        PdfError: If it is not a PDF PDFium can open.
    """
    if not path.is_file():
        msg = f"no such file: {str(path)!r}"
        raise FileNotFoundError(msg)
    try:
        return pdfium.PdfDocument(path)
    except pdfium.PdfiumError as error:
        msg = f"PDFium cannot open it: {error}"
        raise PdfError(msg) from None


def analyse_pdf(path: Path | str) -> PdfDocument:
    """Read a PDF document: its lines and blocks, its structure, and its chunks.

    Args:
        path: The PDF file.

    Returns:
        What the reader found.

    Raises:
        FileNotFoundError: If the file does not exist.
        PdfError: If it is not a PDF PDFium can open.
    """
    document = _open(Path(path))
    try:
        lines: list[Line] = []
        page_objects: dict[int, list[Box]] = {}
        page_boxes: dict[int, Box] = {}
        for page_index in range(len(document)):
            page = document[page_index]
            number = page_index + 1
            page_boxes[number] = page.get_cropbox()
            page_objects[number] = [
                obj.get_bounds() for obj in page.get_objects(max_depth=0) if obj.type != pdfium_c.FPDF_PAGEOBJ_TEXT
            ]
            lines.extend(_lines(number, _combine_accents(_characters(page))))
            page.close()
    finally:
        document.close()
    blocks = _blocks(lines)
    layout = _layout(lines, blocks)
    reader = _Reader(layout, page_objects, page_boxes)
    reader.read()
    chunks, citations = reader.chunks()
    return PdfDocument(
        title=layout.title,
        headings=tuple(reader.headings),
        captions=tuple(reader.regions()),
        equations=tuple(reader.equations),
        references=tuple(reader.references),
        citations=tuple(citations),
        chunks=tuple(chunks),
        lines=tuple(lines),
        blocks=tuple(blocks),
    )


def read_pdf(path: Path | str) -> list[Chunk]:
    """Read a PDF document into chunks, in reading order.

    Args:
        path: The PDF file.

    Returns:
        The chunks: sentences, equations, captions and bibliography entries.

    Raises:
        FileNotFoundError: If the file does not exist.
        PdfError: If it is not a PDF PDFium can open.
    """
    return list(analyse_pdf(path).chunks)


# Rendering


def _png(pixels: object) -> bytes:
    """Encode an 8-bit greyscale image, an array of rows, as PNG."""
    height, width = pixels.shape  # type: ignore[attr-defined]
    rows = b"".join(b"\x00" + bytes(row) for row in pixels)  # type: ignore[attr-defined]

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b"")


def render_region(path: Path | str, page: int, box: Box, scale: float = 2.0) -> bytes:
    """Render a region of a page to PNG, in greyscale.

    The region is the box with a point of margin, at ``scale`` pixels per point, clipped to the
    page. The page is rendered without its ``/Rotate``, in the coordinates the reader's boxes use.

    Args:
        path: The PDF file.
        page: The page number, from 1.
        box: The region, in PDF points as the reader reports them.
        scale: Pixels per point.

    Returns:
        The PNG image.

    Raises:
        FileNotFoundError: If the file does not exist.
        PdfError: If it is not a PDF PDFium can open.
        ValueError: If the page does not exist or the region is empty.
    """
    document = _open(Path(path))
    try:
        if not 1 <= page <= len(document):
            msg = f"page {page} does not exist: the document has {len(document)} pages"
            raise ValueError(msg)
        pdf_page = document[page - 1]
        # Boxes are in the page's own coordinates, so the page is rendered as if it had no /Rotate.
        pdf_page.set_rotation(0)
        origin_x, _, _, origin_top = pdf_page.get_cropbox()
        pixels = pdf_page.render(scale=scale, grayscale=True).to_numpy()
        pdf_page.close()
    finally:
        document.close()
    left = max(0, math.floor((box[0] - origin_x - _PAD) * scale))
    right = min(pixels.shape[1], math.ceil((box[2] - origin_x + _PAD) * scale))
    top = max(0, math.floor((origin_top - box[3] - _PAD) * scale))
    bottom = min(pixels.shape[0], math.ceil((origin_top - box[1] + _PAD) * scale))
    if right <= left or bottom <= top:
        msg = f"the region {_format_box(box)} is empty on page {page}"
        raise ValueError(msg)
    return _png(pixels[top:bottom, left:right])
