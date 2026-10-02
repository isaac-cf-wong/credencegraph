"""Validation of the open ``attributes`` metadata carried by nodes and relations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from credencegraph.core.errors import ValidationError

# A JSON value.
type JSON = bool | int | float | str | list[JSON] | dict[str, JSON] | None

# A read-only JSON value: arrays are tuples and objects are read-only mappings.
type FrozenJSON = bool | int | float | str | tuple[FrozenJSON, ...] | Mapping[str, FrozenJSON] | None


def copy_json(value: Any, path: str) -> Any:
    """Return a deep copy of ``value`` after checking that it is JSON-representable.

    A ``tuple`` is accepted as an array and any ``Mapping`` as an object, so a frozen copy made by
    ``freeze_attributes`` can be copied back.

    Args:
        value: The candidate JSON value.
        path: Location of the value, used in the error message.

    Returns:
        A detached deep copy made of ``dict``, ``list``, ``str``, ``int``, ``float``, ``bool`` and ``None``.

    Raises:
        ValidationError: If the value holds a non-finite float, a non-string key or any other type.
    """
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            msg = f"{path} must be finite, got {value!r}"
            raise ValidationError(msg)
        return value
    if isinstance(value, list | tuple):
        return [copy_json(item, f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                msg = f"{path} has a non-string key {key!r}"
                raise ValidationError(msg)
            out[key] = copy_json(item, f"{path}.{key}")
        return out
    msg = f"{path} is not JSON-representable: {type(value).__name__}"
    raise ValidationError(msg)


def _freeze(value: Any) -> FrozenJSON:
    """Return a read-only view of a validated JSON value, turning arrays into tuples."""
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    return value


def freeze_attributes(value: Any, name: str) -> Mapping[str, FrozenJSON]:
    """Validate an ``attributes`` mapping and return a detached, read-only copy.

    The copy cannot be changed in place at any depth, so it stays as valid as it was when checked.

    Args:
        value: The candidate mapping, a ``dict`` or a copy already frozen by this function.
        name: Description of the mapping, used in the error message.

    Returns:
        The deep copy, with objects as read-only mappings and arrays as tuples.

    Raises:
        ValidationError: If ``value`` is not a mapping with string keys and JSON values.
    """
    if not isinstance(value, dict | MappingProxyType):
        msg = f"{name} must be a dict, got {type(value).__name__}"
        raise ValidationError(msg)
    return _freeze(copy_json(value, name))
