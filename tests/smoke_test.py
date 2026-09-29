"""Smoke tests before publishing to verify the wheel and source distribution."""

from __future__ import annotations

import subprocess
import sys

import credencegraph


def test_basic_import() -> None:
    """Test basic import."""
    print(f"Python version: {sys.version}")
    print(f"Package version: {credencegraph.__version__}")

    # Ensure it's not importing the local folder
    if "site-packages" not in credencegraph.__file__ and "dist" not in credencegraph.__file__:
        print(f"Warning: Package imported from unexpected location: {credencegraph.__file__}")


def test_cli_help() -> None:
    """Test CLI help."""
    # Ensure the 'my-tool' command was registered and runs
    result = subprocess.run(["credencegraph", "--help"], capture_output=True, text=True, check=False)  # noqa: S607
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


if __name__ == "__main__":
    test_basic_import()
    print("Smoke test passed: Package is importable.")

    test_cli_help()
    print("Smoke test passed: The CLI is executable.")
