"""Small PDF documents written byte by byte, for the cases a LaTeX document does not produce easily.

Each builder returns the bytes of a one-page PDF. The text is set with ``Tm``, so its position and
angle are exact.
"""

from __future__ import annotations

import math

WIDTH, HEIGHT = 400, 400


HELVETICA = b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica/Encoding/WinAnsiEncoding>>"


def _string(text: bytes) -> bytes:
    """A PDF literal string."""
    return b"(" + text.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)") + b")"


def text_operators(runs: list[tuple[bytes, float, float, float]], size: float = 10, font: bytes = b"F1") -> bytes:
    """Show each run of ``(text, x, y, angle in degrees counterclockwise)`` in one font."""
    parts = []
    for text, x, y, angle in runs:
        cos, sin = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        matrix = b" ".join(b"%.4f" % value for value in (cos, sin, -sin, cos, x, y))
        parts.append(b"BT /%s %g Tf %s Tm %s Tj ET" % (font, size, matrix, _string(text)))
    return b"\n".join(parts)


def _stream(dictionary: bytes, data: bytes) -> bytes:
    return b"<<%s/Length %d>>stream\n%s\nendstream" % (dictionary, len(data), data)


def _document(objects: list[bytes]) -> bytes:
    """Number the objects from 1, the first being the catalog, and write the file with its cross-reference table."""
    out = b"%PDF-1.7\n"
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return out


def standard_font(name: bytes) -> bytes:
    """One of the 14 standard fonts, such as ``Courier-Bold``."""
    return b"<</Type/Font/Subtype/Type1/BaseFont/%s/Encoding/WinAnsiEncoding>>" % name


def page_pdf(content: bytes, font: bytes = HELVETICA, extra: list[bytes] | None = None, **page: object) -> bytes:
    """A page showing ``content`` with ``font`` as ``/F1``; ``extra`` objects are numbered from 6."""
    return fonts_pdf(content, [font], extra, **page)  # type: ignore[arg-type]


def fonts_pdf(
    content: bytes,
    fonts: list[bytes],
    extra: list[bytes] | None = None,
    media_box: tuple[int, int, int, int] = (0, 0, WIDTH, HEIGHT),
    rotate: int = 0,
) -> bytes:
    """A page showing ``content`` with the fonts as ``/F1``, ``/F2``, ...; ``extra`` objects follow the fonts."""
    names = b"".join(b"/F%d %d 0 R" % (number, number + 4) for number in range(1, len(fonts) + 1))
    return _document(
        [
            b"<</Type/Catalog/Pages 2 0 R>>",
            b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
            b"<</Type/Page/Parent 2 0 R/MediaBox[%d %d %d %d]/Rotate %d/Contents 4 0 R/Resources<</Font<<%s>>>>>>"
            % (*media_box, rotate, names),
            _stream(b"", content),
            *fonts,
            *(extra or []),
        ]
    )


def form_pdf(content: bytes, matrix: tuple[float, ...]) -> bytes:
    """A page whose only content is a form XObject holding ``content``, drawn through ``matrix``."""
    page = b"q %s cm /Fm1 Do Q" % b" ".join(b"%g" % value for value in matrix)
    return _document(
        [
            b"<</Type/Catalog/Pages 2 0 R>>",
            b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
            b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 %d %d]/Contents 4 0 R/Resources<</XObject<</Fm1 5 0 R>>>>>>"
            % (WIDTH, HEIGHT),
            _stream(b"", page),
            _stream(
                b"/Type/XObject/Subtype/Form/BBox[0 0 %d %d]/Resources<</Font<</F1 6 0 R>>>>" % (WIDTH, HEIGHT),
                content,
            ),
            HELVETICA,
        ]
    )


def unmapped_font_pdf(content: bytes) -> bytes:
    """A page showing ``content`` in a Type 3 font whose glyphs have made-up names and no Unicode map."""
    glyph = b"0 0 0 0 500 700 d1 0 0 500 700 re f"
    names = b" ".join(b"/g%d" % code for code in range(256))
    font = (
        b"<</Type/Font/Subtype/Type3/FontBBox[0 0 500 700]/FontMatrix[0.001 0 0 0.001 0 0]"
        b"/CharProcs 6 0 R/Encoding<</Type/Encoding/Differences[0 %s]>>/FirstChar 0/LastChar 255"
        b"/Widths[%s]/Resources<<>>>>" % (names, b" ".join([b"500"] * 256))
    )
    procs = b"<<%s>>" % b"".join(b"/g%d 7 0 R" % code for code in range(256))
    return page_pdf(content, font, [procs, _stream(b"", glyph)])


def to_unicode_pdf(content: bytes, mapping: dict[int, str]) -> bytes:
    """A page showing ``content`` in Helvetica with a ``ToUnicode`` map that sends each code in ``mapping`` to its text."""
    entries = b"\n".join(
        b"<%02X> <%s>" % (code, text.encode("utf-16-be").hex().upper().encode()) for code, text in mapping.items()
    )
    cmap = (
        b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
        b"/CIDSystemInfo <</Registry (Adobe) /Ordering (UCS) /Supplement 0>> def\n"
        b"/CMapName /Adobe-Identity-UCS def /CMapType 2 def\n"
        b"1 begincodespacerange <00> <FF> endcodespacerange\n"
        b"%d beginbfchar\n%s\nendbfchar\n"
        b"endcmap CMapName currentdict /CMap defineresource pop end end" % (len(mapping), entries)
    )
    font = b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica/Encoding/WinAnsiEncoding/ToUnicode 6 0 R>>"
    return page_pdf(content, font, [_stream(b"", cmap)])
