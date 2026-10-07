"""The helpers that write a finding's message."""

from __future__ import annotations

from credencegraph.diagnostics.records import render_evidence, render_item


def test_render_item():
    """Test that one value is written the way it is typed on the command line."""
    assert render_item("a", True) == "a=true"
    assert render_item("b", False) == "b=false"


def test_render_evidence():
    """Test that evidence is its items, in order, separated by commas."""
    assert render_evidence({"b": False, "a": True}) == "b=false, a=true"
    assert render_evidence({}) == ""
