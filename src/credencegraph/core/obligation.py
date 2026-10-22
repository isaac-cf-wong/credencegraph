"""Obligations on inferential relations, and the discharges that record how they were met.

An obligation states what must be true for an inferential step to hold. A discharge records how
it was met, in one of five kinds:

- ``formal``: a named theorem of a proof assistant. No checker ships with this package, so it is
  recorded but not re-checked.
- ``symbolic``: a Python script asserting an identity, such as one checked with sympy.
- ``numerical``: a Python script that fails when a computed quantity misses its tolerance.
- ``citation``: a quote anchored to the cited source.
- ``judgement``: an elicited credence, the relation's strength, and nothing more. A step discharged
  by judgement, or not discharged at all, remains a *sorry*.

A script discharge records the run it was checked by: the ``digest`` of the script's bytes and the
exit ``status`` it returned, both or neither. A script with neither was never run.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from credencegraph.core.anchor import SourceAnchor, require_text
from credencegraph.core.errors import ValidationError

FORMAL = "formal"
SYMBOLIC = "symbolic"
NUMERICAL = "numerical"
CITATION = "citation"
JUDGEMENT = "judgement"

DISCHARGE_KINDS = (FORMAL, SYMBOLIC, NUMERICAL, CITATION, JUDGEMENT)

# Kinds discharged by running a script.
SCRIPT_KINDS = frozenset({SYMBOLIC, NUMERICAL})

_FILE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")

# The fields each kind requires, and the ones it may carry besides.
_REQUIRED = {
    FORMAL: {"theorem"},
    SYMBOLIC: {"script"},
    NUMERICAL: {"script", "tolerance"},
    CITATION: {"anchor"},
    JUDGEMENT: set(),
}
_OPTIONAL = {SYMBOLIC: {"digest", "status"}, NUMERICAL: {"digest", "status"}}
_FIELDS = ("theorem", "script", "digest", "status", "tolerance", "anchor")


def file_digest(data: bytes) -> str:
    """Return the digest of a script: ``sha256:`` and the hex SHA-256 of its bytes.

    Args:
        data: The file's contents.

    Returns:
        The digest, such as ``sha256:9f86d0...``.
    """
    return "sha256:" + hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class Discharge:
    """How an obligation was met.

    Each kind takes only its own fields: ``theorem`` for ``formal``; ``script``, and the optional
    ``digest`` and ``status`` of its recorded run, for ``symbolic`` and ``numerical``, which also
    needs a ``tolerance``; ``anchor``, with a quote, for ``citation``; nothing for ``judgement``.

    Attributes:
        kind: One of ``formal``, ``symbolic``, ``numerical``, ``citation`` and ``judgement``.
        theorem: The name of the theorem that proves the step.
        script: Path of the Python script, relative to the graph file's directory.
        digest: ``sha256:`` and the hex SHA-256 of the script's bytes when it was run.
        status: The exit status of that run; 0 means the check passed.
        tolerance: The tolerance the numerical check holds the computed quantity to.
        anchor: The cited source and the quote from it.
    """

    kind: str
    theorem: str | None = None
    script: str | None = None
    digest: str | None = None
    status: int | None = None
    tolerance: float | None = None
    anchor: SourceAnchor | None = None

    def __post_init__(self) -> None:
        """Validate the fields against the kind."""
        if self.kind not in DISCHARGE_KINDS:
            msg = f"Discharge kind must be one of {', '.join(DISCHARGE_KINDS)}, got {self.kind!r}"
            raise ValidationError(msg)
        given = {name for name in _FIELDS if getattr(self, name) is not None}
        missing = sorted(_REQUIRED[self.kind] - given)
        if missing:
            msg = f"a {self.kind} discharge needs {', '.join(missing)}"
            raise ValidationError(msg)
        extra = sorted(given - _REQUIRED[self.kind] - _OPTIONAL.get(self.kind, set()))
        if extra:
            msg = f"a {self.kind} discharge takes no {', '.join(extra)}"
            raise ValidationError(msg)
        require_text(self.theorem, "Discharge.theorem", optional=True)
        require_text(self.script, "Discharge.script", optional=True)
        if self.script is not None and not self.script.endswith(".py"):
            msg = f"Discharge.script must be a Python script ending in .py, got {self.script!r}"
            raise ValidationError(msg)
        if (self.digest is None) != (self.status is None):
            msg = "a script discharge records both the digest and the status of its run, or neither"
            raise ValidationError(msg)
        if self.digest is not None and not (isinstance(self.digest, str) and _FILE_DIGEST.fullmatch(self.digest)):
            msg = f"Discharge.digest must be 'sha256:' and 64 lowercase hex digits, got {self.digest!r}"
            raise ValidationError(msg)
        if self.status is not None and (isinstance(self.status, bool) or not isinstance(self.status, int)):
            msg = f"Discharge.status must be an integer exit status, got {self.status!r}"
            raise ValidationError(msg)
        if self.tolerance is not None:
            tolerance = self.tolerance
            if isinstance(tolerance, bool) or not isinstance(tolerance, int | float) or not 0 < tolerance < math.inf:
                msg = f"Discharge.tolerance must be a finite number above 0, got {tolerance!r}"
                raise ValidationError(msg)
            object.__setattr__(self, "tolerance", float(tolerance))
        if self.anchor is not None:
            if not isinstance(self.anchor, SourceAnchor):
                msg = f"Discharge.anchor must be a SourceAnchor, got {self.anchor!r}"
                raise ValidationError(msg)
            if self.anchor.quote is None:
                msg = "a citation discharge needs an anchor with a quote"
                raise ValidationError(msg)


@dataclass(frozen=True, slots=True)
class Obligation:
    """What must be true for an inferential step to hold, and how it was met, if it was.

    Attributes:
        statement: The condition, in words.
        discharge: How it was met, or ``None`` while it is open.
    """

    statement: str
    discharge: Discharge | None = None

    def __post_init__(self) -> None:
        """Validate the fields."""
        require_text(self.statement, "Obligation.statement")
        if self.discharge is not None and not isinstance(self.discharge, Discharge):
            msg = f"Obligation.discharge must be a Discharge, got {self.discharge!r}"
            raise ValidationError(msg)
