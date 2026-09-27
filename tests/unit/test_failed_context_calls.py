"""A rule that survives a failed ``BuildContext`` call must not cache the degraded result forever.

Every dependency is recorded *after* the call that read it returned, because only then is the hash
of what came back known. A rule that catches the exception and returns a fallback therefore used to
produce a trace that never mentioned the call at all: the next build verified that trace, found
every recorded input unchanged, and served the degraded value with the provider or the tool healthy
again -- a silent stale artifact, which is the exact failure this engine exists to prevent.

The fix records the failed call with a sentinel hash that no real digest can equal, so verification
mismatches on it and the key re-runs until the call succeeds. The contract (a rule *should* let the
exception propagate) is in docs/SEMANTICS.md; these tests cover what happens when it does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import JsonValue

from incremental_agent_builds.agentic.errors import ProviderUnavailableError
from incremental_agent_builds.agentic.providers.base import LLMRequest, LLMResponse
from incremental_agent_builds.agentic.providers.fake import FakeProvider, PolicyResult
from incremental_agent_builds.engine.context import BuildContext
from incremental_agent_builds.engine.errors import (
    LLMStepError,
    RuleFailedError,
    ToolStepError,
    WorkspaceError,
)
from incremental_agent_builds.engine.keys import BuildKey
from incremental_agent_builds.engine.plan import make_plan
from incremental_agent_builds.engine.rules import Rule, RuleSet, rule
from incremental_agent_builds.engine.store import DepKind
from incremental_agent_builds.engine.tools import BuildTool, BuildToolRegistry
from tests.engine_helpers import CONFIG_TEXT, UpperOut, make_engine, workspace_with

pytestmark = pytest.mark.unit

_DEGRADED = "DEGRADED"


@dataclass
class _Health:
    """Whether the environment a rule reaches through its context is working."""

    healthy: bool = False
    runs: int = 0


class Tolerant(BuildKey):
    """Catches whatever its context raises and returns a fallback value."""

    path: str


class Broken(BuildKey):
    """Fails while the environment is unhealthy."""

    path: str


class Guarded(BuildKey):
    """Needs :class:`Broken` but survives its failure."""

    path: str


def _flaky_provider(health: _Health) -> FakeProvider:
    def policy(request: LLMRequest, rng: Any, turn: int) -> PolicyResult:
        del rng, turn
        if not health.healthy:
            return ProviderUnavailableError("the provider is down")
        payload = {"shouted": request.sha256()[:8].upper()}
        return LLMResponse(
            model=request.model or "fake-a",
            stop_reason="end_turn",
            text=f'{{"shouted":"{payload["shouted"]}"}}',
            parsed=payload,
        )

    return FakeProvider(policy)


def _flaky_tool(health: _Health) -> BuildToolRegistry:
    from pydantic import BaseModel, ConfigDict

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)
        path: str

    def handler(args: Args, inputs: dict[str, bytes | None] | Any) -> JsonValue:
        if not health.healthy:
            raise ToolStepError("the tool is unavailable")
        data = inputs.get(args.path)
        return {"upper": (data or b"").decode("utf-8").upper()}

    return BuildToolRegistry(
        [
            BuildTool(
                name="upper",
                description="Upper-case a workspace file.",
                args_model=Args,
                handler=handler,
                inputs=lambda args: (args.path,),
            )
        ]
    )


def _llm_rules(health: _Health) -> RuleSet:
    @rule(Tolerant)
    async def tolerant(ctx: BuildContext, key: Tolerant) -> JsonValue:
        health.runs += 1
        text = await ctx.read(key.path) or ""
        try:
            loud = await ctx.llm("shout", {"word": text}, model="fake-a", schema=UpperOut)
        except LLMStepError:
            return {"shouted": _DEGRADED}
        return {"shouted": loud.shouted}

    rules: tuple[Rule[Any], ...] = (tolerant,)
    return RuleSet(rules)


def _tool_rules(health: _Health) -> RuleSet:
    @rule(Tolerant)
    async def tolerant(ctx: BuildContext, key: Tolerant) -> JsonValue:
        health.runs += 1
        try:
            return await ctx.tool("upper", {"path": key.path})
        except ToolStepError:
            return {"upper": _DEGRADED}

    rules: tuple[Rule[Any], ...] = (tolerant,)
    return RuleSet(rules)


def _need_rules(health: _Health) -> RuleSet:
    @rule(Broken)
    async def broken(ctx: BuildContext, key: Broken) -> JsonValue:
        await ctx.read(key.path)
        if not health.healthy:
            raise RuntimeError("the world is on fire")
        return "recovered"

    @rule(Guarded)
    async def guarded(ctx: BuildContext, key: Guarded) -> JsonValue:
        health.runs += 1
        try:
            return await ctx.need(Broken(path=key.path))
        except RuleFailedError:
            return _DEGRADED

    rules: tuple[Rule[Any], ...] = (broken, guarded)
    return RuleSet(rules)


def _config_rules(health: _Health) -> RuleSet:
    @rule(Tolerant)
    async def tolerant(ctx: BuildContext, key: Tolerant) -> JsonValue:
        health.runs += 1
        try:
            return await ctx.config("demo.factor")
        except WorkspaceError:
            return _DEGRADED

    rules: tuple[Rule[Any], ...] = (tolerant,)
    return RuleSet(rules)


def _glob_rules(health: _Health) -> RuleSet:
    @rule(Tolerant)
    async def tolerant(ctx: BuildContext, key: Tolerant) -> JsonValue:
        health.runs += 1
        try:
            found: list[JsonValue] = list(await ctx.glob("../*.py"))
        except WorkspaceError:
            return _DEGRADED
        return found

    rules: tuple[Rule[Any], ...] = (tolerant,)
    return RuleSet(rules)


def test_a_degraded_llm_result_is_not_reused_once_the_provider_recovers() -> None:
    health = _Health()
    engine = make_engine(workspace_with(), _llm_rules(health), provider=_flaky_provider(health))
    target = Tolerant(path="in/a.txt")

    first = engine.build_sync([target])
    assert first.ok, "the rule swallowed the provider error"
    assert engine.value(target) == {"shouted": _DEGRADED}

    health.healthy = True
    second = engine.build_sync([target])
    assert second.ok
    assert second.stats.executed == 1, "the failed LLM step must force a re-run"
    assert second.stats.reused == 0
    assert engine.value(target) != {"shouted": _DEGRADED}
    assert health.runs == 2


def test_a_degraded_tool_result_is_not_reused_once_the_tool_recovers() -> None:
    health = _Health()
    engine = make_engine(
        workspace_with(), _tool_rules(health), tools=_flaky_tool(health), provider=None
    )
    target = Tolerant(path="in/a.txt")

    assert engine.build_sync([target]).ok
    assert engine.value(target) == {"upper": _DEGRADED}

    health.healthy = True
    second = engine.build_sync([target])
    assert second.stats.executed == 1
    assert engine.value(target) == {"upper": "1"}


def test_a_swallowed_dependency_failure_is_not_reused_once_the_dependency_recovers() -> None:
    health = _Health()
    engine = make_engine(workspace_with(), _need_rules(health), provider=None)
    target = Guarded(path="in/a.txt")

    first = engine.build_sync([target])
    assert first.target_hashes, "the guarded key itself succeeded"
    assert engine.value(target) == _DEGRADED

    health.healthy = True
    second = engine.build_sync([target])
    assert second.ok
    assert second.stats.executed == 2, "both the recovered key and its guard must re-run"
    assert engine.value(target) == "recovered"


def test_a_swallowed_config_error_is_not_reused_once_the_file_parses() -> None:
    health = _Health()
    workspace = workspace_with({"iab.toml": "this is not toml = = ="})
    engine = make_engine(workspace, _config_rules(health), provider=None)
    target = Tolerant(path="in/a.txt")

    assert engine.build_sync([target]).ok
    assert engine.value(target) == _DEGRADED

    workspace.write("iab.toml", CONFIG_TEXT)
    second = engine.build_sync([target])
    assert second.stats.executed == 1
    assert engine.value(target) == 2


def test_a_swallowed_invalid_glob_keeps_the_key_dirty_on_every_build() -> None:
    """The price of ignoring the contract, stated as a test.

    This failure is deterministic — the pattern is invalid whatever the workspace holds — so the key
    can never be shown up to date and re-executes on every build. That is the honest cost of
    catching the exception instead of letting it propagate, and why the contract says to propagate.
    """
    health = _Health()
    engine = make_engine(workspace_with(), _glob_rules(health), provider=None)
    target = Tolerant(path="in/a.txt")

    assert engine.build_sync([target]).ok
    assert engine.value(target) == _DEGRADED

    second = engine.build_sync([target])
    assert second.stats.executed == 1
    assert second.stats.reused == 0
    assert health.runs == 2


def test_a_failed_call_makes_the_planner_report_the_key_as_dirty() -> None:
    health = _Health()
    engine = make_engine(workspace_with(), _llm_rules(health), provider=_flaky_provider(health))
    target = Tolerant(path="in/a.txt")
    engine.build_sync([target])

    health.healthy = True
    plan = make_plan(engine, [target])
    assert [item.key_id for item in plan.items] == [target.key_id]
    assert "failed" in plan.items[0].reason


def test_a_rule_whose_calls_all_succeed_records_no_failure() -> None:
    """The sentinel must appear only for calls that actually raised."""
    health = _Health(healthy=True)
    engine = make_engine(workspace_with(), _llm_rules(health), provider=_flaky_provider(health))
    target = Tolerant(path="in/a.txt")
    engine.build_sync([target]).raise_for_failures()

    trace = engine.trace(target)
    assert trace is not None
    kinds = [dep.kind for group in trace.deps for dep in group]
    assert DepKind.ERROR not in kinds
    assert engine.build_sync([target]).stats.reused == 1
