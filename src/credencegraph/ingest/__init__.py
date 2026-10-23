"""The ingestion layer: readers that turn a document into chunks, and the graph of document nodes built from them.

This package is installed with the ``ingest`` extra. The core never imports it. ``chunks`` needs no
extra dependency; ``latex``, the LaTeX reader, needs ``pylatexenc``; ``pdf``, the PDF reader, needs
``pypdfium2``.
"""

from __future__ import annotations

from credencegraph.ingest.chunks import Chunk, chunks_to_graph, ends_sentence

__all__ = ["Chunk", "chunks_to_graph", "ends_sentence"]
