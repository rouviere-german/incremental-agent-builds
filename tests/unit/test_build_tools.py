"""Build tools: read-only by construction, memoized by fingerprint, volatile ones on a TTL.

Rules call tools; the model never does. The registry enforces that by refusing any tool that
declares a side-effecting capability, which is what keeps every build step free of effects.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, JsonValue

from incremental_agent_builds.agentic.errors import ToolArgumentError, ToolNotAllowedError
from incremental_agent_builds.agentic.tools import Capability
from incremental_agent_builds.engine.context import BuildContext
from incremental_agent_builds.engine.engine import EngineConfig
from incremental_agent_builds.engine.keys import BuildKey
from incremental_agent_builds.engine.rules import RuleSet, rule
from incremental_agent_builds.engine.tools import BuildTool, BuildToolRegistry, tool_fingerprint
from tests.engine_helpers import Upper, counting_rules, make_engine, upper_tool, workspace_with

pytestmark = pytest.mark.unit


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    depth: int = 1


class LaxArgs(BaseModel):
    path: str


def _tool(**overrides: Any) -> BuildTool[Any]:
    base: dict[str, Any] = {
        "name": "probe",
        "description": "A probe.",
        "args_model": Args,
        "handler": lambda args, inputs: {"path": args.path},
        "inputs": lambda args: (args.path,),
    }
    base.update(overrides)
    return BuildTool(**base)


# --- registration -------------------------------------------------------------------------------


def test_a_read_only_tool_registers() -> None:
    registry = BuildToolRegistry([_tool()])
    assert registry.names == ("probe",)
    assert registry.get("probe").name == "probe"


@pytest.mark.parametrize(
    "capabilities",
    [
        frozenset({Capability.WRITE_DATA}),
        frozenset({Capability.READ_DATA, Capability.NETWORK_EGRESS}),
        frozenset({Capability.READ_DATA, Capability.SEND_MESSAGE}),
        frozenset(),
    ],
)
def test_a_tool_with_any_side_effecting_capability_is_refused(capabilities: frozenset[Any]) -> None:
    with pytest.raises(ToolNotAllowedError, match="read-only"):
        BuildToolRegistry([_tool(capabilities=capabilities)])


def test_registering_a_name_twice_is_refused() -> None:
    registry = BuildToolRegistry([_tool()])
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_tool())


def test_an_unregistered_tool_is_not_available() -> None:
    with pytest.raises(ToolNotAllowedError, match="not registered"):
        BuildToolRegistry().get("probe")


@pytest.mark.parametrize("name", ["", "has space", "a" * 65, "bad/name"])
def test_invalid_tool_names_are_refused(name: str) -> None:
    with pytest.raises(ValueError, match="invalid tool name"):
        _tool(name=name)


def test_a_lax_arguments_model_is_refused() -> None:
    with pytest.raises(ValueError, match="extra='forbid'"):
        _tool(args_model=LaxArgs)


def test_a_non_positive_ttl_is_refused() -> None:
    with pytest.raises(ValueError, match="ttl_seconds must be positive"):
        _tool(ttl_seconds=0)


def test_volatile_is_derived_from_the_ttl() -> None:
    assert not _tool().volatile
    assert _tool(ttl_seconds=5).volatile


# --- argument validation ------------------------------------------------------------------------


def test_arguments_are_validated_strictly_and_canonicalized() -> None:
    args, canonical = _tool().validate({"path": "a.py", "depth": 2})
    assert args.path == "a.py"
    assert canonical == {"path": "a.py", "depth": 2}


@pytest.mark.parametrize(
    "raw",
    [{}, {"path": 1}, {"path": "a", "extra": 1}, {"path": "a", "depth": "2"}],
)
def test_invalid_arguments_report_their_location_only(raw: dict[str, Any]) -> None:
    with pytest.raises(ToolArgumentError, match="invalid arguments for probe at"):
        _tool().validate(raw)


# --- fingerprints -------------------------------------------------------------------------------


def test_a_fingerprint_covers_name_version_arguments_and_input_hashes() -> None:
    base = tool_fingerprint("probe", 1, {"path": "a"}, {"a": "h1"})
    assert base == tool_fingerprint("probe", 1, {"path": "a"}, {"a": "h1"})
    assert base != tool_fingerprint("other", 1, {"path": "a"}, {"a": "h1"})
    assert base != tool_fingerprint("probe", 2, {"path": "a"}, {"a": "h1"})
    assert base != tool_fingerprint("probe", 1, {"path": "b"}, {"a": "h1"})
    assert base != tool_fingerprint("probe", 1, {"path": "a"}, {"a": "h2"})
    assert base != tool_fingerprint("probe", 1, {"path": "a"}, {})


def test_a_tool_version_bump_invalidates_its_memoized_results() -> None:
    workspace = workspace_with()
    rules, _ = counting_rules()
    first = make_engine(workspace, rules, tools=BuildToolRegistry([upper_tool(version=1)]))
    first.build_sync([Upper(path="in/a.txt")])
    second = make_engine(
        workspace,
        rules,
        tools=BuildToolRegistry([upper_tool(version=2)]),
        store=first.store,
        cas=first.cas,
    )
    result = second.build_sync([Upper(path="in/a.txt")])
    assert result.stats.tool_runs == 1, "a new tool version must not reuse the old step"


# --- volatile tools -----------------------------------------------------------------------------


class Volatile(BuildKey):
    pass


def _volatile_rules(counter: list[int]) -> tuple[RuleSet, BuildToolRegistry]:
    class NoArgs(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

    def handler(args: NoArgs, inputs: dict[str, bytes | None] | Any) -> JsonValue:
        counter.append(1)
        return {"calls": len(counter)}

    tool = BuildTool(
        name="clock",
        description="A volatile source outside the workspace.",
        args_model=NoArgs,
        handler=handler,
        ttl_seconds=60.0,
    )

    @rule(Volatile)
    async def volatile(ctx: BuildContext, key: Volatile) -> JsonValue:
        return await ctx.tool("clock", {})

    return RuleSet([volatile]), BuildToolRegistry([tool])


def test_a_volatile_result_is_reused_inside_its_ttl() -> None:
    counter: list[int] = []
    rules, tools = _volatile_rules(counter)
    engine = make_engine(workspace_with(), rules, tools=tools, clock=lambda: 1000.0)
    engine.build_sync([Volatile()])
    assert len(counter) == 1
    engine.build_sync([Volatile()])
    assert len(counter) == 1, "inside the TTL the recorded result stands"


def test_an_expired_volatile_result_is_re_run_and_still_cuts_off() -> None:
    """Re-running a volatile tool that returns the same value must not invalidate dependents."""
    counter: list[int] = []

    class NoArgs(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

    def handler(args: NoArgs, inputs: dict[str, bytes | None] | Any) -> JsonValue:
        counter.append(1)
        return {"stable": True}

    tool = BuildTool(
        name="clock",
        description="A volatile source outside the workspace.",
        args_model=NoArgs,
        handler=handler,
        ttl_seconds=10.0,
    )

    @rule(Volatile)
    async def volatile(ctx: BuildContext, key: Volatile) -> JsonValue:
        return await ctx.tool("clock", {})

    now = [1000.0]
    engine = make_engine(
        workspace_with(),
        RuleSet([volatile]),
        tools=BuildToolRegistry([tool]),
        clock=lambda: now[0],
        config=EngineConfig(jobs=1, llm_concurrency=1),
    )
    engine.build_sync([Volatile()])
    assert len(counter) == 1

    now[0] += 100.0  # past the TTL
    result = engine.build_sync([Volatile()])
    assert len(counter) == 2, "an expired volatile step must be re-run"
    assert result.stats.executed == 0, "an unchanged volatile result still lets the key be reused"


def test_a_tool_that_raises_fails_its_rule_without_leaking_details() -> None:
    class NoArgs(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

    def handler(args: NoArgs, inputs: dict[str, bytes | None] | Any) -> JsonValue:
        raise RuntimeError("secret internal detail")

    @rule(Volatile)
    async def volatile(ctx: BuildContext, key: Volatile) -> JsonValue:
        return await ctx.tool("boom", {})

    tool = BuildTool(name="boom", description="Fails.", args_model=NoArgs, handler=handler)
    engine = make_engine(workspace_with(), RuleSet([volatile]), tools=BuildToolRegistry([tool]))
    result = engine.build_sync([Volatile()])
    assert not result.ok
    message = result.failures[Volatile().key_id]
    assert "RuntimeError" in message
    assert "secret internal detail" not in message


def test_calling_an_unknown_tool_fails_the_rule() -> None:
    @rule(Volatile)
    async def volatile(ctx: BuildContext, key: Volatile) -> JsonValue:
        return await ctx.tool("absent", {})

    engine = make_engine(workspace_with(), RuleSet([volatile]), tools=BuildToolRegistry())
    result = engine.build_sync([Volatile()])
    assert not result.ok
    assert "ToolStepError" in result.failures[Volatile().key_id]


def test_tool_inputs_are_recorded_as_file_dependencies() -> None:
    """A tool's declared inputs must be trace dependencies, or an edit would not invalidate it."""
    workspace = workspace_with()
    rules, _ = counting_rules()
    engine = make_engine(workspace, rules)
    engine.build_sync([Upper(path="in/a.txt")])
    trace = engine.trace(Upper(path="in/a.txt"))
    assert trace is not None
    refs = [dep.ref for group in trace.deps for dep in group]
    assert "in/a.txt" in refs

    now = time.monotonic()
    workspace.write("in/a.txt", "changed")
    assert engine.build_sync([Upper(path="in/a.txt")]).stats.tool_runs == 1
    assert time.monotonic() - now < 5
