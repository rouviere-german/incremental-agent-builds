"""Build keys, rule declarations and value hashing: the vocabulary traces are written in."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import JsonValue, ValidationError

from incremental_agent_builds.engine.context import BuildContext
from incremental_agent_builds.engine.errors import RuleDefinitionError
from incremental_agent_builds.engine.hashing import (
    MISSING,
    canonical_bytes,
    is_sha256_hex,
    value_hash,
)
from incremental_agent_builds.engine.keys import BuildKey, split_key_id
from incremental_agent_builds.engine.rules import RuleSet, rule

pytestmark = pytest.mark.unit


class Simple(BuildKey):
    pass


class WithFields(BuildKey):
    path: str
    depth: int = 0


class CustomKind(BuildKey):
    kind = "renamed"
    value: str


# --- keys ---------------------------------------------------------------------------------------


def test_a_key_id_is_the_kind_plus_canonical_arguments() -> None:
    assert Simple().key_id == "simple:{}"
    assert WithFields(path="a.py", depth=2).key_id == 'with_fields:{"depth":2,"path":"a.py"}'
    assert CustomKind(value="x").key_id == 'renamed:{"value":"x"}'


def test_the_default_kind_is_the_class_name_in_snake_case() -> None:
    class ModuleOutlineLike(BuildKey):
        pass

    assert ModuleOutlineLike.kind == "module_outline_like"


def test_equal_keys_have_equal_ids_whatever_the_argument_order() -> None:
    assert WithFields(depth=1, path="a").key_id == WithFields(path="a", depth=1).key_id


def test_keys_are_frozen_strict_and_reject_unknown_fields() -> None:
    key = WithFields(path="a")
    with pytest.raises(ValidationError):
        key.path = "b"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        WithFields(path="a", extra=1)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        WithFields(path="a", depth="2")  # type: ignore[arg-type]


def test_an_invalid_kind_is_refused_at_class_definition() -> None:
    with pytest.raises(ValueError, match="must match"):

        class Bad(BuildKey):
            kind = "Not Valid"


def test_str_of_a_key_is_its_id() -> None:
    assert str(WithFields(path="a")) == WithFields(path="a").key_id


def test_split_key_id_recovers_kind_and_arguments() -> None:
    assert split_key_id('with_fields:{"depth":2,"path":"a"}') == (
        "with_fields",
        {"depth": 2, "path": "a"},
    )


@pytest.mark.parametrize(
    "key_id", ["", "nocolon", "Bad:{}", "with_fields:[1]", "with_fields:notjson", ":{}"]
)
def test_malformed_key_ids_are_refused(key_id: str) -> None:
    with pytest.raises(ValueError, match=r"malformed|Expecting"):
        split_key_id(key_id)


# --- rules --------------------------------------------------------------------------------------


@rule(WithFields, version=2)
async def with_fields_rule(ctx: BuildContext, key: WithFields) -> JsonValue:
    return key.path


@rule(Simple)
async def simple_rule(ctx: BuildContext, key: Simple) -> JsonValue:
    return None


def test_a_rule_carries_its_key_type_name_and_version() -> None:
    assert with_fields_rule.name == "with_fields"
    assert with_fields_rule.version == 2
    assert simple_rule.version == 1


def test_a_synchronous_function_cannot_be_a_rule() -> None:
    with pytest.raises(RuleDefinitionError, match="async"):

        @rule(Simple)  # type: ignore[arg-type]
        def not_async(ctx: BuildContext, key: Simple) -> JsonValue:
            return None


def test_rule_requires_a_build_key_subclass_and_a_valid_version() -> None:
    with pytest.raises(RuleDefinitionError, match="BuildKey"):
        rule(str)  # type: ignore[type-var]
    with pytest.raises(RuleDefinitionError, match="version"):
        rule(Simple, version=-1)


def test_a_rule_set_rejects_duplicate_kinds() -> None:
    with pytest.raises(RuleDefinitionError, match="duplicate"):
        RuleSet([simple_rule, simple_rule])


def test_a_rule_set_looks_rules_up_by_key_and_kind() -> None:
    rules = RuleSet([simple_rule, with_fields_rule])
    assert rules.kinds == ("simple", "with_fields")
    assert "simple" in rules
    assert rules.rule_for(Simple()) is simple_rule
    assert rules.get("with_fields") is with_fields_rule
    assert rules.get("absent") is None


def test_an_unknown_key_has_no_rule() -> None:
    rules = RuleSet([simple_rule])
    with pytest.raises(RuleDefinitionError, match="no rule"):
        rules.rule_for(WithFields(path="a"))


def test_key_from_id_round_trips_and_rejects_foreign_ids() -> None:
    rules = RuleSet([simple_rule, with_fields_rule])
    key = WithFields(path="a.py", depth=3)
    assert rules.key_from_id(key.key_id) == key
    assert rules.key_from_id("unknown:{}") is None
    assert rules.key_from_id("with_fields:{}") is None, "missing required fields must not validate"
    assert rules.key_from_id('with_fields:{"path":"a","extra":1}') is None
    assert rules.key_from_id("garbage") is None


def test_key_from_id_rejects_a_non_canonical_id() -> None:
    """Accepting a non-canonical id would give one key two trace rows."""
    rules = RuleSet([with_fields_rule])
    assert rules.key_from_id('with_fields:{"path":"a","depth":0}') is None


# --- hashing ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right"),
    [({"a": 1, "b": 2}, {"b": 2, "a": 1}), ([1, 2], [1, 2]), ("x", "x"), (None, None)],
)
def test_equal_json_values_hash_equally(left: Any, right: Any) -> None:
    assert value_hash(left) == value_hash(right)


@pytest.mark.parametrize(
    ("left", "right"),
    [(1, "1"), (1, 1.5), ({"a": 1}, {"a": 2}), ([1, 2], [2, 1]), (None, False), (0, False)],
)
def test_different_values_hash_differently(left: Any, right: Any) -> None:
    assert value_hash(left) != value_hash(right)


def test_missing_is_not_a_valid_digest_so_it_matches_no_content() -> None:
    assert not is_sha256_hex(MISSING)
    assert value_hash(None) != MISSING
    assert value_hash("") != MISSING


def test_canonical_bytes_are_compact_utf8() -> None:
    assert canonical_bytes({"k": "é"}) == '{"k":"é"}'.encode()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a" * 64, True),
        ("A" * 64, False),
        ("a" * 63, False),
        ("a" * 65, False),
        ("g" * 64, False),
        ("", False),
    ],
)
def test_is_sha256_hex_accepts_only_lowercase_64_hex(text: str, expected: bool) -> None:
    assert is_sha256_hex(text) is expected
