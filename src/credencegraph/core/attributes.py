"""Validation of the open ``attributes`` metadata carried by nodes and relations."""

from __future__ import annotations

import math
from typing import Any

from credencegraph.core.errors import ValidationError

# A JSON value.
type JSON = bool | int | float | str | list[JSON] | dict[str, JSON] | None


def copy_json(value: Any, path: str) -> Any:
    """Return a deep copy of ``value`` after checking that it is JSON-representable.

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
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                msg = f"{path} has a non-string key {key!r}"
                raise ValidationError(msg)
            out[key] = copy_json(item, f"{path}.{key}")
        return out
    msg = f"{path} is not JSON-representable: {type(value).__name__}"
    raise ValidationError(msg)


def copy_attributes(value: Any, name: str) -> dict[str, JSON]:
    """Validate an ``attributes`` mapping and return a detached copy.

    Args:
        value: The candidate mapping.
        name: Description of the mapping, used in the error message.

    Returns:
        The deep copy.

    Raises:
        ValidationError: If ``value`` is not a mapping with string keys and JSON values.
    """
    if not isinstance(value, dict):
        msg = f"{name} must be a dict, got {type(value).__name__}"
        raise ValidationError(msg)
    return copy_json(value, name)
