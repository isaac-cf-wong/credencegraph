"""Fixtures for the command-line tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from _cli import Cli, example_graph

from credencegraph.core import dump


@pytest.fixture
def cli() -> Cli:
    """Provide a command-line driver.

    Returns:
        The driver.
    """
    return Cli()


@pytest.fixture
def graph_file(tmp_path: Path) -> Path:
    """Write the example graph to a file.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The file.
    """
    path = tmp_path / "graph.json"
    dump(example_graph(), path)
    return path
