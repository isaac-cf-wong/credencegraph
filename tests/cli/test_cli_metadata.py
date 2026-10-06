"""The shape of the command line, read from the Typer app rather than from rendered help."""

from __future__ import annotations

import pytest
import typer.main

from credencegraph.cli.main import app
from credencegraph.core.node import DEFAULT_KIND
from credencegraph.diagnostics import DEFAULT_CLAIM_THRESHOLD, DEFAULT_FAILURE_THRESHOLD
from credencegraph.inference import DEFAULT_DRAWS

COMMANDS = ("version", "init", "add-node", "relate", "query", "diagnose", "check")


@pytest.fixture(scope="module")
def group():
    """The command group Typer builds from the app.

    Returns:
        The group.
    """
    return typer.main.get_command(app)


def params(command) -> dict:
    """Index a command's parameters by name.

    Args:
        command: The command.

    Returns:
        The parameters by name.
    """
    return {param.name: param for param in command.params}


def test_commands(group):
    """Test that exactly the documented commands are registered."""
    assert set(group.commands) == set(COMMANDS)


@pytest.mark.parametrize("name", COMMANDS)
def test_every_command_takes_json(group, name):
    """Test that every command has a ``--json`` flag that is off by default."""
    option = params(group.commands[name])["as_json"]
    assert option.param_type_name == "option"
    assert option.opts == ["--json"]
    assert option.is_flag
    assert option.default is False


@pytest.mark.parametrize("name", [name for name in COMMANDS if name != "version"])
def test_graph_path_is_the_first_argument(group, name):
    """Test that every graph command takes the graph file as its first, required argument."""
    first = group.commands[name].params[0]
    assert first.param_type_name == "argument"
    assert first.name == "path"
    assert first.required


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("init", ["path"]),
        ("add-node", ["path"]),
        ("relate", ["path", "source", "target"]),
        ("query", ["path", "kind", "targets"]),
        ("diagnose", ["path"]),
        ("check", ["path"]),
        ("version", []),
    ],
)
def test_positional_arguments(group, name, arguments):
    """Test the positional arguments of each command, in order."""
    assert [p.name for p in group.commands[name].params if p.param_type_name == "argument"] == arguments


@pytest.mark.parametrize(
    ("name", "option", "opts", "flags"),
    [
        ("init", "force", ["--force"], (False, False)),
        ("add-node", "node_id", ["--id"], (True, False)),
        ("add-node", "statement", ["--statement"], (False, False)),
        ("add-node", "kind", ["--kind"], (False, False)),
        ("add-node", "base", ["--base"], (False, False)),
        ("add-node", "stated", ["--stated"], (False, False)),
        ("add-node", "source", ["--source"], (False, True)),
        ("relate", "relation_type", ["--type"], (True, False)),
        ("relate", "strength", ["--strength"], (False, False)),
        ("relate", "relation_id", ["--id"], (False, False)),
        ("query", "given", ["--given"], (False, True)),
        ("query", "setting", ["--set"], (False, True)),
        ("query", "draws", ["--draws"], (False, False)),
        ("query", "seed", ["--seed"], (False, False)),
        ("diagnose", "target", ["--target"], (False, True)),
        ("diagnose", "targets", ["--targets"], (False, False)),
        ("diagnose", "claim_threshold", ["--claim-threshold"], (False, False)),
        ("diagnose", "failure_threshold", ["--failure-threshold"], (False, False)),
    ],
)
def test_options(group, name, option, opts, flags):
    """Test each option's flag and its ``(required, multiple)`` flags."""
    param = params(group.commands[name])[option]
    assert param.param_type_name == "option"
    assert param.opts == opts
    assert (param.required, param.multiple) == flags


def test_query_targets_are_variadic(group):
    """Test that ``query`` takes one or more target nodes after the kind."""
    targets = params(group.commands["query"])["targets"]
    assert targets.nargs == -1
    assert targets.required


def test_defaults_are_the_library_defaults(group):
    """Test that the defaults that change results are taken from the library, not restated."""
    assert params(group.commands["add-node"])["kind"].default == DEFAULT_KIND
    assert params(group.commands["query"])["draws"].default == DEFAULT_DRAWS
    assert params(group.commands["diagnose"])["claim_threshold"].default == DEFAULT_CLAIM_THRESHOLD
    assert params(group.commands["diagnose"])["failure_threshold"].default == DEFAULT_FAILURE_THRESHOLD
