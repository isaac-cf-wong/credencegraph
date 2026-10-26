"""Chunks of a document, and the graph of document nodes built from them.

A reader turns a document into ``Chunk`` objects in document order; ``chunks_to_graph`` numbers
them, anchors each verbatim and links each citation to the bibliography entry it names. The result
passes ``check --level anchored``. ``ends_sentence`` is the sentence-boundary rule the readers share.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from credencegraph.core.anchor import SourceAnchor
from credencegraph.core.graph import Graph
from credencegraph.core.node import Node
from credencegraph.core.relation import Relation
from credencegraph.rubric.faithfulness import _NUMERAL
from credencegraph.rubric.model import DOCUMENT, UNASSIGNED
from credencegraph.rubric.text import normalise_text, text_digest
from credencegraph.rubric.validate import CITES

SENTENCE = "sentence"
EQUATION = "equation"
CAPTION = "caption"
FOOTNOTE = "footnote"
REFERENCE = "reference"
UNITS = (SENTENCE, EQUATION, CAPTION, FOOTNOTE, REFERENCE)

# A word after which a full stop does not end a sentence: a common abbreviation or an initial.
_ABBREVIATIONS = frozenset(
    {
        "e.g",
        "i.e",
        "al",
        "cf",
        "vs",
        "fig",
        "figs",
        "eq",
        "eqs",
        "ref",
        "refs",
        "sec",
        "secs",
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
    }
)
_INITIAL = re.compile(r"[A-Z]")
# Words after which a single capital is a numbered label, as in "Table I.", not an initial.
_LABELLED = frozenset({"table", "tab", "section", "sec", "appendix", "app", "chapter", "part"})

# The id prefix of each unit. Sentences, footnotes and captions share one numbering.
_PREFIXES = {SENTENCE: "s", FOOTNOTE: "s", CAPTION: "s", EQUATION: "e", REFERENCE: "r"}


@dataclass(frozen=True, slots=True)
class Chunk:
    r"""A chunk of a document, as a reader found it.

    Attributes:
        unit: One of ``UNITS``.
        text: The chunk's text, normalised with ``normalise_text``.
        locator: Where it is, such as ``file=sec/method.tex;lines=41-42``.
        cites: The bibliography keys the chunk cites, in order of first citation.
        labels: An equation's labels, in order: its ``\label`` values in LaTeX, its printed numbers in a PDF.
        key: A bibliography entry's key.
    """

    unit: str
    text: str
    locator: str
    cites: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    key: str | None = None


def ends_sentence(before: str, after: str) -> bool:
    """Tell whether a full stop, question mark or exclamation mark followed by whitespace ends a sentence.

    It does unless the word before it is a common abbreviation or an initial, or the text after it
    starts in lower case. A single capital after a word such as "Table" or "Sec." is a label, as in
    "Table I.", and one after a numeral is a unit, as in "0.6 K."; both end the sentence. An initial
    after a year, as in "In 2019 A. Smith showed", is therefore read as a unit and ends one too.

    Args:
        before: The text up to the punctuation.
        after: The text after the punctuation and the whitespace that follows it.

    Returns:
        Whether the sentence ends there.
    """
    words = before.split()
    if words:
        stem = words[-1].lstrip("(\"'`~")
        if stem.lower() in _ABBREVIATIONS:
            return False
        if _INITIAL.fullmatch(stem) and not (len(words) > 1 and _labels_or_counts(words[-2])):
            return False
    return not (after and after[0].islower())


def _labels_or_counts(word: str) -> bool:
    """Tell whether a single capital after this word is a label or a unit rather than an initial."""
    return word.rstrip(".").lower() in _LABELLED or _NUMERAL.fullmatch(word) is not None


def _ids(chunks: list[Chunk]) -> list[str]:
    """Number the chunks per id prefix, in document order, zero-padded to the width each prefix needs."""
    totals: dict[str, int] = {}
    for chunk in chunks:
        prefix = _PREFIXES[chunk.unit]
        totals[prefix] = totals.get(prefix, 0) + 1
    counters = dict.fromkeys(totals, 0)
    ids = []
    for chunk in chunks:
        prefix = _PREFIXES[chunk.unit]
        counters[prefix] += 1
        ids.append(f"{prefix}-{counters[prefix]:0{len(str(totals[prefix]))}d}")
    return ids


def chunks_to_graph(
    chunks: Iterable[Chunk], document: str, initial: Mapping[str, str] | None = None
) -> tuple[Graph, list[str]]:
    """Build the graph of document nodes for a document's chunks.

    Every chunk becomes a node anchored verbatim: its statement and its anchor's quote are the
    chunk's text, and the anchor's digest is the text's digest. An equation with labels carries
    ``attributes.label``: its label, or the list of them when it has several. A bibliography entry
    carries ``attributes.key``. Each citation becomes a ``cites`` relation from the citing chunk to
    the first entry with that key.

    Args:
        chunks: The chunks, in document order.
        document: The document id of every anchor.
        initial: The type each unit starts with, by unit, as a rubric's ``initial`` gives it; any
            other unit starts ``unassigned``.

    Returns:
        The graph, and the cited keys that name no bibliography entry, in order of first citation.
    """
    chunks = list(chunks)
    initial = initial or {}
    ids = _ids(chunks)
    graph = Graph()
    references: dict[str, str] = {}
    for node_id, chunk in zip(ids, chunks, strict=True):
        text = normalise_text(chunk.text)
        attributes: dict[str, object] = {"origin": DOCUMENT, "unit": chunk.unit}
        if chunk.labels:
            attributes["label"] = chunk.labels[0] if len(chunk.labels) == 1 else list(chunk.labels)
        if chunk.key is not None:
            attributes["key"] = chunk.key
            references.setdefault(chunk.key, node_id)
        anchor = SourceAnchor(document, chunk.locator, text, text_digest(text))
        graph.add_node(Node(node_id, initial.get(chunk.unit, UNASSIGNED), text, (anchor,), attributes=attributes))
    unresolved: list[str] = []
    for node_id, chunk in zip(ids, chunks, strict=True):
        for key in chunk.cites:
            target = references.get(key)
            if target is None:
                if key not in unresolved:
                    unresolved.append(key)
                continue
            relation_id = f"{node_id}-{CITES}-{target}"
            if relation_id not in graph.relations:
                graph.add_relation(Relation(relation_id, CITES, node_id, target))
    return graph, unresolved
