"""The ``version`` command."""

from __future__ import annotations

import typer


def version_command() -> None:
    """Print the installed credencegraph version."""
    from credencegraph.version import __version__  # noqa: PLC0415

    typer.echo(__version__)
