"""Narrowing helpers for walking ``JsonValue`` in tests.

A bundle, a plan or a build result is JSON: ``dict[str, JsonValue]``. Indexing into it twice is not
type-safe, and sprinkling ``# type: ignore`` over the navigation hides the real assertions. These
helpers narrow one level at a time and fail with a readable message when the shape is wrong, which
is itself something worth asserting: a test that silently walked past a missing key would pass for
the wrong reason.
"""

from __future__ import annotations

from pydantic import JsonValue

__all__ = ["arr", "at", "num", "obj", "text"]


def obj(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict), f"expected an object, got {type(value).__name__}: {value!r}"
    return value


def arr(value: JsonValue) -> list[JsonValue]:
    assert isinstance(value, list), f"expected an array, got {type(value).__name__}: {value!r}"
    return value


def text(value: JsonValue) -> str:
    assert isinstance(value, str), f"expected a string, got {type(value).__name__}: {value!r}"
    return value


def num(value: JsonValue) -> float:
    assert isinstance(value, int | float), f"expected a number, got {value!r}"
    return float(value)


def at(value: JsonValue, *path: str | int) -> JsonValue:
    """Walk ``path`` through nested objects and arrays, naming the step that failed."""
    current = value
    for step in path:
        if isinstance(step, int):
            items = arr(current)
            assert step < len(items), f"index {step} out of range at {path}"
            current = items[step]
        else:
            mapping = obj(current)
            assert step in mapping, f"key {step!r} missing at {path} (have {sorted(mapping)})"
            current = mapping[step]
    return current
