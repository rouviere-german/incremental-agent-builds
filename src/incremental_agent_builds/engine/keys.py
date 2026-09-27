"""Build keys: typed, immutable identifiers of rule results.

A key type is a frozen pydantic model. Its fields are the rule's argument and its ``kind``
names the rule (by default the class name in snake_case). The canonical key id is
``<kind>:<canonical JSON of the fields>``, so equal keys always have equal ids and the id alone is
enough to rebuild the key object when a trace is verified.
"""

from __future__ import annotations

import json
import re
from functools import cached_property
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from incremental_agent_builds.core.canonical import canonical_json

__all__ = ["KIND_PATTERN", "BuildKey", "split_key_id"]

KIND_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_CAMEL = re.compile(r"(?<!^)(?=[A-Z])")


class BuildKey(BaseModel):
    """Base class of every key type. Subclasses declare fields and optionally ``kind``."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    kind: ClassVar[str] = ""

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        if "kind" not in cls.__dict__ or not cls.__dict__["kind"]:
            cls.kind = _CAMEL.sub("_", cls.__name__).lower()
        if not KIND_PATTERN.fullmatch(cls.kind):
            raise ValueError(f"key kind {cls.kind!r} must match {KIND_PATTERN.pattern}")

    @cached_property
    def key_id(self) -> str:
        return f"{self.kind}:{canonical_json(self.model_dump(mode='json'))}"

    def __str__(self) -> str:
        return self.key_id


def split_key_id(key_id: str) -> tuple[str, dict[str, Any]]:
    """Split a key id into its kind and its argument object."""
    kind, sep, raw = key_id.partition(":")
    if not sep or not KIND_PATTERN.fullmatch(kind):
        raise ValueError(f"malformed key id {key_id!r}")
    args = json.loads(raw)
    if not isinstance(args, dict):
        # A malformed id is bad data, not a programming error, so ValueError is the right type:
        # callers treat it the same as a JSON decode failure.
        raise ValueError(f"malformed key id {key_id!r}")  # noqa: TRY004
    return kind, args
