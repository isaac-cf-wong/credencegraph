"""Helpers for driving the command line in tests."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from credencegraph.cli.main import app
from credencegraph.core import Beta, Graph, Node, Relation, SourceAnchor


class Cli:
    """Run the ``credencegraph`` app in-process and decode its JSON output."""

    def __init__(self) -> None:
        """Create the runner."""
        self.runner = CliRunner()

    def run(self, *args: str):
        """Run the app with the given arguments.

        Args:
            *args: The command-line arguments.

        Returns:
            The click result, with separate stdout and stderr.
        """
        return self.runner.invoke(app, [str(arg) for arg in args])

    def json(self, *args: str, exit_code: int = 0) -> dict:
        """Run a command with ``--json`` and return the decoded object on stdout.

        Args:
            *args: The command-line arguments, without ``--json``.
            exit_code: The expected exit status.

        Returns:
            The decoded JSON object.
        """
        result = self.run(*args, "--json")
        assert result.exit_code == exit_code, (result.exit_code, result.stdout, result.stderr, result.exception)
        return json.loads(result.stdout)

    def error(self, *args: str) -> dict:
        """Run a command expected to fail and return its ``error`` object.

        Args:
            *args: The command-line arguments, without ``--json``.

        Returns:
            The ``error`` object, with ``code``, ``message``, ``hint`` and ``details``.
        """
        output = self.json(*args, exit_code=1)
        assert set(output) == {"command", "error"}
        assert set(output["error"]) == {"code", "message", "hint", "details"}
        return output["error"]


def example_graph() -> Graph:
    """The README example: ``claim`` rests on ``signal``, which requires ``calibrated``.

    P(signal) = 0.9 * 0.05 = 0.045, and P(claim) = 1 - (1 - 0.1) * (1 - 0.8 * 0.045) = 0.1324,
    where 0.8 is the mean of Beta(8, 2).

    Returns:
        The graph.
    """
    graph = Graph()
    graph.add_node(Node("calibrated", statement="The instrument is calibrated.", base=0.9))
    graph.add_node(Node("signal", base=0.05, sources=[SourceAnchor("doi:10.0000/example", "p4")]))
    graph.add_node(Node("claim", base=0.1, stated=0.5))
    graph.add_node(Node("alice", kind="person"))
    graph.add_relation(Relation("r1", "requires", "calibrated", "signal", strength=1.0))
    graph.add_relation(Relation("r2", "supports", "signal", "claim", strength=Beta(8, 2)))
    graph.add_relation(Relation("a1", "authored_by", "claim", "alice"))
    return graph
