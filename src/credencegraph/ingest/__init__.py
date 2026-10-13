"""The ingestion layer: readers that turn a document into chunks, and the graph of document nodes built from them.

This package is installed with the ``ingest`` extra. The core never imports it. ``chunks`` needs no
extra dependency; ``latex``, the LaTeX reader, needs ``pylatexenc``.
"""

from __future__ import annotations

from credencegraph.ingest.chunks import Chunk, chunks_to_graph

__all__ = ["Chunk", "chunks_to_graph"]
