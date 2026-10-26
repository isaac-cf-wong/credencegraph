r"""The LaTeX reader: a ``.tex`` document, with the files it inputs, as chunks located by file and line.

The reader parses with ``pylatexenc`` and keeps the LaTeX source of each chunk, so inline
mathematics, ``\ref`` and ``\cite`` stay in the text as written. It

- follows ``\input`` and ``\include``, relative to the main file's directory;
- expands the macros defined with ``\newcommand``, ``\renewcommand``, ``\providecommand`` and
  ``\def`` (with ``#1``-style parameters), wherever they are defined; a macro still undefined, or
  nested deeper than ``MAX_EXPANSION_DEPTH``, stays as written;
- splits running prose into sentences, at ``.``, ``?`` or ``!`` followed by a space and a word not
  starting in lower case, except after a common abbreviation or an initial, and at paragraph
  breaks, headings, list items and displays;
- makes each display (``\[ \]``, ``$$ $$``, ``equation``, ``align``, ...) an equation chunk with
  its ``\label`` values, each ``\caption`` a caption chunk, each ``\footnote`` a footnote chunk and
  each ``\bibitem`` of a ``thebibliography`` a reference chunk; the bibliography is read from the
  document, or from ``<main>.bbl`` beside the main file when the document uses ``\bibliography``;
- drops comments, ``\label`` and headings from the text.

Locators are ``file=<path relative to the main file's directory>;lines=<first>-<last>``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from pylatexenc.latexwalker import (
    LatexCharsNode,
    LatexCommentNode,
    LatexEnvironmentNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexMathNode,
    LatexNode,
    LatexWalker,
    get_default_latex_context_db,
)
from pylatexenc.macrospec import MacroSpec

from credencegraph.ingest.chunks import CAPTION, EQUATION, FOOTNOTE, REFERENCE, SENTENCE, Chunk, ends_sentence
from credencegraph.rubric.text import normalise_text

MAX_EXPANSION_DEPTH = 32

_DEFINING = frozenset({"newcommand", "renewcommand", "providecommand"})
_INPUTS = frozenset({"input", "include"})
_CITES = frozenset(
    {
        "cite",
        "citep",
        "citet",
        "citealp",
        "citealt",
        "citeauthor",
        "citeyear",
        "parencite",
        "textcite",
        "autocite",
        "nocite",
    }
)
# typos:off - "multline" is the amsmath environment's name
_DISPLAYS = frozenset(
    name + star
    for name in ("equation", "align", "gather", "multline", "flalign", "alignat", "eqnarray", "displaymath")
    for star in ("", "*")
)
# typos:on
_FLOATS = frozenset({"figure", "figure*", "table", "table*"})
# Environments whose content is neither prose nor a chunk.
_SKIPPED = frozenset({"tabular", "tabular*", "tikzpicture", "verbatim", "lstlisting", "comment"})
# Macros that end the sentence before them and are dropped with their arguments.
_BREAKING = frozenset(
    {
        "part",
        "chapter",
        "section",
        "subsection",
        "subsubsection",
        "paragraph",
        "subparagraph",
        "item",
        "par",
        "title",
        "author",
        "date",
        "maketitle",
        "appendix",
        "bibliographystyle",
        "newpage",
        "clearpage",
        "tableofcontents",
    }
)
# Macros dropped from the text.
_DROPPED = frozenset({"label", "newblock", "thanks"})
# The argument specs pylatexenc lacks for the macros the reader handles.
_SPECS = (
    MacroSpec("caption", "*[{"),
    MacroSpec("bibitem", "[{"),
    MacroSpec("thanks", "{"),
    MacroSpec("bibliographystyle", "{"),
    MacroSpec("nocite", "{"),
    MacroSpec("parencite", "*[[{"),
    MacroSpec("textcite", "*[[{"),
    MacroSpec("autocite", "*[[{"),
)

_BOUNDARY = re.compile(r"(?P<par>\n[ \t]*\n\s*)|(?P<end>[.?!][\'\")\]]*)(?P<space>\s+|$)")
_DEF = re.compile(r"\\def\s*\\([A-Za-z@]+)\s*((?:#\d\s*)*)\{")
_PARAMETER = re.compile(r"#(\d)")


class LatexError(Exception):
    r"""A LaTeX document cannot be read: it inputs itself, or a ``\def`` body is never closed."""


@dataclass(frozen=True)
class _Definition:
    """A user macro: its parameter count, the default of an optional first parameter, and its body."""

    parameters: int
    default: str | None
    body: str


def _group_body(text: str, start: int) -> str:
    """Read the brace group that opens at ``text[start]``.

    Returns:
        The group's content.

    Raises:
        LatexError: If the group is never closed.
    """
    depth = 0
    index = start
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
        index += 1
    msg = "a \\def body is never closed"
    raise LatexError(msg)


def _argument_text(node: LatexNode | None) -> str:
    """The source of a macro argument, without the braces or brackets around it."""
    if node is None:
        return ""
    if isinstance(node, LatexGroupNode):
        return "".join(child.latex_verbatim() for child in node.nodelist)
    return node.latex_verbatim()


def _walk(nodes: list[LatexNode] | None) -> Iterator[LatexNode]:
    """Every node of a node list and, depth first, of the lists and arguments inside it."""
    for node in nodes or ():
        if node is None:
            continue
        yield node
        yield from _walk(getattr(node, "nodelist", None))
        arguments = getattr(node, "nodeargd", None)
        if arguments is not None:
            yield from _walk(arguments.argnlist)


@dataclass
class _Sentence:
    """The sentence being read: its text pieces, the span of source they come from, and its citations."""

    pieces: list[str] = field(default_factory=list)
    start: int | None = None
    end: int | None = None
    cites: list[str] = field(default_factory=list)

    def add(self, text: str, start: int, end: int, cites: tuple[str, ...] | list[str] = ()) -> None:
        """Append a piece of text read from ``source[start:end]``."""
        self.pieces.append(text)
        if text.strip():
            self.start = start if self.start is None else self.start
            self.end = end
        self.cites.extend(key for key in cites if key not in self.cites)


class _Reader:
    """Read a main ``.tex`` file and every file it inputs into chunks."""

    def __init__(self, main: Path) -> None:
        self.main = main
        self.root = main.parent
        self.definitions: dict[str, _Definition] = {}
        self.chunks: list[Chunk] = []
        self.pending: list[Chunk] = []
        self.sentence = _Sentence()
        self.may_end = False
        self.after_comment = False
        self.texts: dict[Path, str] = {}
        self.stack: list[Path] = []
        self.walker: LatexWalker | None = None
        self.expanding: list[LatexMacroNode] = []
        self.file = ""
        self.context = get_default_latex_context_db()

    # Files

    def text(self, path: Path) -> str:
        """The content of a file, read once."""
        if path not in self.texts:
            self.texts[path] = path.read_text(encoding="utf-8")
        return self.texts[path]

    def resolve(self, name: str) -> Path:
        r"""The file an ``\input`` or ``\include`` argument names, relative to the main file's directory.

        Raises:
            FileNotFoundError: If neither the name nor the name with ``.tex`` is a file.
        """
        path = self.root / name.strip()
        candidates = [path] if path.suffix == ".tex" else [path, path.with_name(path.name + ".tex")]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        msg = f"\\input{{{name}}} names no file: tried {', '.join(str(c) for c in candidates)}"
        raise FileNotFoundError(msg)

    def parse(self, path: Path) -> tuple[LatexWalker, list[LatexNode]]:
        """Parse a file with the macros defined so far."""
        walker = LatexWalker(self.text(path), latex_context=self.context, tolerant_parsing=True)
        nodes, _, _ = walker.get_latex_nodes()
        return walker, nodes

    # Macro definitions

    def collect_definitions(self, path: Path) -> None:
        """Record the macros a file defines, and those of every file it inputs, in reading order."""
        if path in self.stack:
            chain = " -> ".join(str(item) for item in [*self.stack, path])
            msg = f"the document inputs itself: {chain}"
            raise LatexError(msg)
        self.stack.append(path)
        _, nodes = self.parse(path)
        # A \def is found in the text with its comments blanked out, so a commented-out one defines nothing.
        text = list(self.text(path))
        for comment in _walk(nodes):
            if isinstance(comment, LatexCommentNode):
                text[comment.pos : comment.pos + comment.len] = " " * comment.len
        uncommented = "".join(text)
        for match in _DEF.finditer(uncommented):
            body = _group_body(uncommented, match.end() - 1)
            self.definitions[match[1]] = _Definition(len(_PARAMETER.findall(match[2])), None, body)
        for node in _walk(nodes):
            if not isinstance(node, LatexMacroNode) or node.nodeargd is None:
                continue
            if node.macroname in _DEFINING:
                _, name, count, default, body = node.nodeargd.argnlist
                macro = _argument_text(name).strip().lstrip("\\")
                if node.macroname == "providecommand" and macro in self.definitions:
                    continue
                self.definitions[macro] = _Definition(
                    int(_argument_text(count) or 0),
                    None if default is None else _argument_text(default),
                    _argument_text(body),
                )
            elif node.macroname in _INPUTS:
                self.collect_definitions(self.resolve(_argument_text(node.nodeargd.argnlist[-1])))
        self.stack.pop()

    def install_definitions(self) -> None:
        r"""Teach the parser the arguments of every user macro, so ``\vec{x}`` parses as one macro."""
        specs = []
        for name, definition in self.definitions.items():
            optional = "[" if definition.default is not None else ""
            specs.append(MacroSpec(name, optional + "{" * (definition.parameters - len(optional))))
        self.context.add_context_category("credencegraph-user", macros=[*_SPECS, *specs], prepend=True)

    def expand(self, node: LatexMacroNode, cites: list[str], labels: list[str], depth: int) -> str:
        """The text of a user macro with its arguments substituted and its body expanded in turn."""
        definition = self.definitions[node.macroname]
        arguments = list(node.nodeargd.argnlist) if node.nodeargd is not None else []
        values = []
        for index, argument in enumerate(arguments):
            if argument is None:
                values.append(definition.default if index == 0 else "")
            else:
                values.append(self.render_nodes(getattr(argument, "nodelist", [argument]), cites, labels, depth))
        body = _PARAMETER.sub(lambda m: values[int(m[1]) - 1] if int(m[1]) <= len(values) else m[0], definition.body)
        walker = LatexWalker(body, latex_context=self.context, tolerant_parsing=True)
        nodes, _, _ = walker.get_latex_nodes()
        self.expanding.append(node)
        try:
            text = self.render_nodes(nodes, cites, labels, depth + 1)
        finally:
            self.expanding.pop()
        return text + ("" if arguments else node.macro_post_space or "")

    # Rendering: the LaTeX text of a node, macros expanded, comments and labels dropped

    def render_nodes(self, nodes: list[LatexNode], cites: list[str], labels: list[str], depth: int = 0) -> str:
        """The text of a node list."""
        return "".join(self.render(node, cites, labels, depth) for node in nodes if node is not None)

    def render(self, node: LatexNode, cites: list[str], labels: list[str], depth: int = 0) -> str:
        """The text of a node, collecting the citation keys and labels inside it."""
        if isinstance(node, LatexCommentNode):
            return ""
        if isinstance(node, LatexGroupNode):
            opening, closing = node.delimiters
            return opening + self.render_nodes(node.nodelist, cites, labels, depth) + closing
        if isinstance(node, LatexMathNode):
            opening, closing = node.delimiters
            return opening + self.render_nodes(node.nodelist, cites, labels, depth) + closing
        if isinstance(node, LatexEnvironmentNode):
            arguments = node.nodeargd.argnlist if node.nodeargd is not None else []
            head = "".join(argument.latex_verbatim() for argument in arguments if argument is not None)
            body = self.render_nodes(node.nodelist, cites, labels, depth)
            return f"\\begin{{{node.environmentname}}}{head}{body}\\end{{{node.environmentname}}}"
        if isinstance(node, LatexMacroNode):
            return self.render_macro(node, cites, labels, depth)
        return node.latex_verbatim()

    def render_macro(self, node: LatexMacroNode, cites: list[str], labels: list[str], depth: int) -> str:
        """The text of a macro: a label or footnote taken out, a user macro expanded, arguments read."""
        name = node.macroname
        if name == "label" and node.nodeargd is not None:
            labels.append(_argument_text(node.nodeargd.argnlist[-1]).strip())
            return ""
        if name == "footnote":
            self.footnote(node)
            return ""
        if name in _DROPPED:
            return ""
        if name in _CITES and node.nodeargd is not None:
            for key in _argument_text(node.nodeargd.argnlist[-1]).split(","):
                if key.strip() and key.strip() not in cites:
                    cites.append(key.strip())
        if name in self.definitions and depth < MAX_EXPANSION_DEPTH:
            return self.expand(node, cites, labels, depth)
        if node.nodeargd is not None:
            return self.render_arguments(node, cites, labels, depth)
        return node.latex_verbatim()

    def render_arguments(self, node: LatexMacroNode, cites: list[str], labels: list[str], depth: int) -> str:
        """The source of a macro with each argument replaced by its text, so what is inside them is read too."""
        source = node.latex_verbatim()
        parts = []
        cursor = 0
        for argument in sorted((a for a in node.nodeargd.argnlist if a is not None), key=lambda a: a.pos):
            start = argument.pos - node.pos
            parts.extend((source[cursor:start], self.render(argument, cites, labels, depth)))
            cursor = start + argument.len
        parts.append(source[cursor:])
        return "".join(parts)

    def footnote(self, node: LatexMacroNode) -> None:
        r"""Add a footnote chunk once the current sentence ends, wherever the ``\footnote`` is.

        Inside a macro expansion the footnote is located at the macro the source holds.
        """
        if node.nodeargd is None or not node.nodeargd.argnlist:
            return
        cites: list[str] = []
        argument = node.nodeargd.argnlist[-1]
        text = self.render_nodes(getattr(argument, "nodelist", [argument]), cites, [])
        located = self.expanding[0] if self.expanding else node
        self.emit(self.chunk(FOOTNOTE, text, located, cites), after_sentence=True)

    # Chunks

    def locator(self, start: int, end: int) -> str:
        """The locator of ``source[start:end]`` in the current file, from its first to its last visible character."""
        source = self.walker.s  # type: ignore[union-attr]
        while end > start + 1 and source[end - 1].isspace():
            end -= 1
        while start < end - 1 and source[start].isspace():
            start += 1
        first, _ = self.walker.pos_to_lineno_colno(start)  # type: ignore[union-attr]
        last, _ = self.walker.pos_to_lineno_colno(end - 1)  # type: ignore[union-attr]
        return f"file={self.file};lines={first}-{last}"

    def chunk(self, unit: str, text: str, node: LatexNode, cites=(), labels=(), key=None) -> Chunk | None:  # noqa: PLR0913, PLR0917
        """A chunk of a node's text, or ``None`` if the text is empty."""
        text = normalise_text(text)
        if not text:
            return None
        return Chunk(unit, text, self.locator(node.pos, node.pos + node.len), tuple(cites), tuple(labels), key)

    def emit(self, chunk: Chunk | None, *, after_sentence: bool = False) -> None:
        """Add a chunk now, or once the current sentence ends."""
        if chunk is not None:
            (self.pending if after_sentence else self.chunks).append(chunk)

    def end_sentence(self) -> None:
        """Close the current sentence, then add the chunks waiting for it."""
        sentence, self.sentence, self.may_end = self.sentence, _Sentence(), False
        text = normalise_text("".join(sentence.pieces))
        if text and any(char.isalnum() for char in text) and sentence.start is not None:
            self.chunks.append(Chunk(SENTENCE, text, self.locator(sentence.start, sentence.end), tuple(sentence.cites)))
        self.chunks.extend(self.pending)
        self.pending.clear()

    # Prose

    def read_file(self, path: Path, *, body_only: bool) -> None:
        """Read a file's prose: the ``document`` environment of the main file, or a whole input file."""
        if path in self.stack:
            chain = " -> ".join(str(item) for item in [*self.stack, path])
            msg = f"the document inputs itself: {chain}"
            raise LatexError(msg)
        self.end_sentence()
        saved = self.walker, self.file
        self.stack.append(path)
        self.walker, nodes = self.parse(path)
        self.file = Path(os.path.relpath(path, self.root)).as_posix()
        if body_only:
            document = next(
                (n for n in nodes if isinstance(n, LatexEnvironmentNode) and n.environmentname == "document"), None
            )
            if document is not None:
                nodes = document.nodelist
        self.prose(nodes)
        self.end_sentence()
        self.stack.pop()
        self.walker, self.file = saved

    def prose(self, nodes: list[LatexNode]) -> None:
        """Read a list of nodes of running prose."""
        skip_def = 0
        for node in nodes:
            if node is None:
                continue
            if skip_def:
                skip_def = self.skip_def(node, skip_def)
                continue
            after_comment, self.after_comment = self.after_comment, False
            if isinstance(node, LatexCharsNode):
                self.chars(node, after_comment)
            elif isinstance(node, LatexCommentNode):
                self.after_comment = node.latex_verbatim().endswith("\n") or after_comment
            elif isinstance(node, LatexMathNode) and node.displaytype == "display":
                self.display(node)
            elif isinstance(node, LatexEnvironmentNode):
                self.environment(node)
            elif isinstance(node, LatexMacroNode):
                skip_def = self.macro(node)
            else:
                self.piece(node)

    def skip_def(self, node: LatexNode, state: int) -> int:
        r"""Skip the name, the parameters and the body of a ``\def`` in prose; return the state for the next node."""
        if state == 1:  # the macro name
            return 2
        if isinstance(node, LatexGroupNode):  # the body
            return 0
        return 2  # the parameter text

    def piece(self, node: LatexNode) -> None:
        """Add an inline node to the current sentence."""
        cites: list[str] = []
        labels: list[str] = []
        text = self.render(node, cites, labels)
        if text:
            self.may_end = False
        self.sentence.add(text, node.pos, node.pos + node.len, cites)

    def chars(self, node: LatexCharsNode, after_comment: bool) -> None:
        """Add running text, ending sentences at paragraph breaks and sentence ends."""
        text = node.chars
        if after_comment and re.match(r"[ \t]*\n", text):
            self.end_sentence()
        if self.may_end:
            match = re.match(r"\s+(?=\S)", text)
            if match and not text[match.end()].islower():
                self.sentence.add(text[: match.end()], node.pos, node.pos + match.end())
                self.end_sentence()
                text, offset = text[match.end() :], match.end()
            else:
                offset = 0
            self.may_end = False
        else:
            offset = 0
        start = 0
        for match in _BOUNDARY.finditer(text):
            if match["par"]:
                self.sentence.add(
                    text[start : match.start()], node.pos + offset + start, node.pos + offset + match.start()
                )
                self.end_sentence()
                start = match.end()
                continue
            # The sentence so far, with the math and macros read before this run of text.
            before = "".join(self.sentence.pieces) + text[start : match.start()]
            if not ends_sentence(before, text[match.end() :]):
                continue
            stop = match.end("end")
            self.sentence.add(text[start:stop], node.pos + offset + start, node.pos + offset + stop)
            if match["space"]:
                self.end_sentence()
            else:
                self.may_end = True
            start = match.end()
        if start < len(text):
            self.sentence.add(text[start:], node.pos + offset + start, node.pos + offset + len(text))
            if text[start:].strip():
                self.may_end = False

    def display(self, node: LatexNode, labels: list[str] | None = None) -> None:
        """Add an equation chunk for a display; it ends the sentence before it."""
        self.end_sentence()
        cites: list[str] = []
        labels = [] if labels is None else labels
        text = self.render(node, cites, labels)
        self.emit(self.chunk(EQUATION, text, node, cites, labels))

    def environment(self, node: LatexEnvironmentNode) -> None:
        """Read an environment: a display, a float, a bibliography, skipped content, or more prose."""
        name = node.environmentname
        if name in _DISPLAYS:
            self.display(node)
        elif name in _FLOATS:
            self.end_sentence()
            for inner in _walk(node.nodelist):
                if isinstance(inner, LatexMacroNode) and inner.macroname == "caption" and inner.nodeargd is not None:
                    cites: list[str] = []
                    text = self.render_nodes(getattr(inner.nodeargd.argnlist[-1], "nodelist", []), cites, [])
                    self.emit(self.chunk(CAPTION, text, inner, cites))
        elif name == "thebibliography":
            self.end_sentence()
            self.bibliography(node.nodelist)
        elif name in _SKIPPED:
            self.end_sentence()
        else:
            self.end_sentence()
            self.prose(node.nodelist)
            self.end_sentence()

    def macro(self, node: LatexMacroNode) -> int:
        r"""Read a macro in prose; return 1 if it starts a ``\def`` whose parts must be skipped, else 0."""
        name = node.macroname
        arguments = node.nodeargd.argnlist if node.nodeargd is not None else []
        if name == "def":
            return 1
        if name in _DEFINING:
            return 0
        if name in _INPUTS and arguments:
            self.read_file(self.resolve(_argument_text(arguments[-1])), body_only=False)
        elif name == "bibliography":
            self.end_sentence()
            bbl = self.main.with_suffix(".bbl")
            if bbl.is_file():
                self.read_bbl(bbl)
        elif name == "footnote":
            self.footnote(node)
        elif name in _BREAKING:
            self.end_sentence()
        elif name in _DROPPED:
            pass
        else:
            self.piece(node)
        return 0

    def read_bbl(self, path: Path) -> None:
        """Read the ``thebibliography`` of a BibTeX ``.bbl`` file."""
        saved = self.walker, self.file
        self.walker, nodes = self.parse(path)
        self.file = Path(os.path.relpath(path, self.root)).as_posix()
        for node in nodes:
            if isinstance(node, LatexEnvironmentNode) and node.environmentname == "thebibliography":
                self.bibliography(node.nodelist)
        self.walker, self.file = saved

    def bibliography(self, nodes: list[LatexNode]) -> None:
        r"""Add a reference chunk per ``\bibitem``: its key, and its text up to the next one."""
        entries: list[tuple[LatexMacroNode, list[LatexNode]]] = []
        for node in nodes:
            if isinstance(node, LatexMacroNode) and node.macroname == "bibitem" and node.nodeargd is not None:
                entries.append((node, []))
            elif entries and node is not None:
                entries[-1][1].append(node)
        for item, content in entries:
            key = _argument_text(item.nodeargd.argnlist[-1]).strip()
            cites: list[str] = []
            text = normalise_text(self.render_nodes(content, cites, []))
            if not text:
                continue
            locator = self.locator(item.pos, content[-1].pos + content[-1].len)
            self.chunks.append(Chunk(REFERENCE, text, locator, (), (), key))


def read_latex(path: Path) -> list[Chunk]:
    """Read a LaTeX document into chunks, in document order.

    Args:
        path: The main ``.tex`` file.

    Returns:
        The chunks.

    Raises:
        FileNotFoundError: If the main file, or a file it inputs, does not exist.
        LatexError: If the document inputs itself.
    """
    reader = _Reader(Path(path))
    reader.collect_definitions(reader.main)
    reader.install_definitions()
    reader.read_file(reader.main, body_only=True)
    return reader.chunks
