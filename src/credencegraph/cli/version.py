"""The ``version`` command."""

from __future__ import annotations

from credencegraph.cli.common import JsonOption, Result, respond


def version_command(as_json: JsonOption = False) -> None:
    """Print the installed credencegraph version."""
    from credencegraph.version import __version__  # noqa: PLC0415

    respond("version", as_json, lambda: Result({"version": __version__}, [__version__]))
