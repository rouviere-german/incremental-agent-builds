"""Small engines and rule sets shared by the engine unit tests.

The helpers here build an engine over an in-memory workspace, trace store and CAS, so a test can
make an edit and rebuild in microseconds. ``counting_rules`` records how often each rule body ran,
which is how the cutoff and reuse tests distinguish "reused" from "re-executed with the same value".
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, JsonValue

from incremental_agent_builds.agentic.providers.base import LLMRequest, LLMResponse
from incremental_agent_builds.agentic.providers.fake import FakeProvider
from incremental_agent_builds.engine.cas import ContentStore, MemoryContentStore
from incremental_agent_builds.engine.context import BuildContext
from incremental_agent_builds.engine.engine import BuildEngine, EngineConfig
from incremental_agent_builds.engine.keys import BuildKey
from incremental_agent_builds.engine.rules import Rule, RuleSet, rule
from incremental_agent_builds.engine.store import TraceStore
from incremental_agent_builds.engine.tools import BuildTool, BuildToolRegistry
from incremental_agent_builds.engine.workspace import MemoryWorkspace

__all__ = [
    "CONFIG_TEXT",
    "TEMPLATE_TEXT",
    "Counted",
    "Doubled",
    "Echo",
    "Shout",
    "Sum",
    "Upper",
    "UpperOut",
    "counting_rules",
    "make_engine",
    "static_provider",
    "step_rules",
    "upper_tool",
]

CONFIG_TEXT = '[demo]\nfactor = 2\nmodel = "fake-a"\n'
TEMPLATE_TEXT = "Shout {{word}}.\n"


class Echo(BuildKey):
    """Reads one file and returns its text (or None)."""

    path: str


class Doubled(BuildKey):
    """Needs Echo and repeats it ``factor`` times from the config."""

    path: str


class Sum(BuildKey):
    """Globs the inputs and needs every Echo it finds: a dynamic dependency set."""

    pattern: str


class Upper(BuildKey):
    """Uses the ``upper`` build tool on a file."""

    path: str


class Counted(BuildKey):
    """Returns ``value % modulus``, so re-executions often produce the same value."""

    modulus: int


class Shout(BuildKey):
    """One LLM step and one tool step, plus a config value that neither of them depends on.

    The config value is what makes this key useful for step-level tests: editing it invalidates the
    key's trace without changing either step fingerprint, so the rule re-runs and has to decide
    whether it may reuse the recorded LLM output and tool result. It is read *after* both steps, so
    verification walks past them before it finds the mismatch.
    """

    path: str


class UpperOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    shouted: str


@dataclass
class _Counts:
    runs: Counter[str] = field(default_factory=Counter)


def upper_tool(*, version: int = 1, ttl_seconds: float | None = None) -> BuildTool[Any]:
    """A read-only tool that upper-cases a declared input file."""

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)
        path: str

    def handler(args: Args, inputs: dict[str, bytes | None] | Any) -> JsonValue:
        data = inputs.get(args.path)
        return {"upper": (data or b"").decode("utf-8").upper()}

    return BuildTool(
        name="upper",
        description="Upper-case a workspace file.",
        args_model=Args,
        handler=handler,
        inputs=lambda args: (args.path,),
        version=version,
        ttl_seconds=ttl_seconds,
    )


def static_provider() -> FakeProvider:
    """A provider whose output is a pure function of the request hash."""

    def policy(request: LLMRequest, rng: random.Random, turn: int) -> LLMResponse:
        del rng, turn
        payload = {"shouted": request.sha256()[:8].upper()}
        return LLMResponse(
            model=request.model or "fake-a",
            stop_reason="end_turn",
            text=json.dumps(payload),
            parsed=payload,
        )

    return FakeProvider(policy)


def counting_rules() -> tuple[RuleSet, _Counts]:
    """The rule set used by the scheduler tests, plus a per-rule execution counter."""
    counts = _Counts()

    @rule(Echo)
    async def echo(ctx: BuildContext, key: Echo) -> JsonValue:
        counts.runs["echo"] += 1
        return await ctx.read(key.path)

    @rule(Doubled)
    async def doubled(ctx: BuildContext, key: Doubled) -> JsonValue:
        counts.runs["doubled"] += 1
        text = await ctx.need(Echo(path=key.path))
        factor = await ctx.config("demo.factor")
        times = factor if isinstance(factor, int) else 1
        return (str(text) if text is not None else "") * times

    @rule(Sum)
    async def total(ctx: BuildContext, key: Sum) -> JsonValue:
        counts.runs["sum"] += 1
        paths = await ctx.glob(key.pattern)
        values = await ctx.need_all([Echo(path=path) for path in paths])
        found: list[JsonValue] = [str(value) for value in values if value is not None]
        found.sort(key=str)
        return found

    @rule(Upper)
    async def upper(ctx: BuildContext, key: Upper) -> JsonValue:
        counts.runs["upper"] += 1
        return await ctx.tool("upper", {"path": key.path})

    @rule(Counted)
    async def counted(ctx: BuildContext, key: Counted) -> JsonValue:
        counts.runs["counted"] += 1
        text = await ctx.read("in/a.txt")
        number = int(text.strip()) if text and text.strip().lstrip("-").isdigit() else 0
        return number % key.modulus

    rules: tuple[Rule[Any], ...] = (echo, doubled, total, upper, counted)
    return RuleSet(rules), counts


def step_rules() -> tuple[RuleSet, _Counts]:
    """A rule set whose single rule makes one LLM step and one tool step."""
    counts = _Counts()

    @rule(Shout)
    async def shout(ctx: BuildContext, key: Shout) -> JsonValue:
        counts.runs["shout"] += 1
        text = await ctx.read(key.path) or ""
        loud = await ctx.llm("shout", {"word": text}, model="fake-a", schema=UpperOut)
        tooled = await ctx.tool("upper", {"path": key.path})
        # Read last on purpose: verification then reaches the LLM and tool steps first, so a test
        # that edits this value exercises both the verify-time and the execute-time step lookup.
        await ctx.config("demo.factor")
        return {"shouted": loud.shouted, "tool": tooled}

    rules: tuple[Rule[Any], ...] = (shout,)
    return RuleSet(rules), counts


def make_engine(
    workspace: MemoryWorkspace,
    rules: RuleSet,
    *,
    config: EngineConfig | None = None,
    tools: BuildToolRegistry | None = None,
    provider: object | None = None,
    store: TraceStore | None = None,
    cas: ContentStore | None = None,
    **kwargs: Any,
) -> BuildEngine:
    return BuildEngine(
        rules=rules,
        workspace=workspace,
        store=store or TraceStore(":memory:", mac_key=os.urandom(32)),
        cas=cas or MemoryContentStore(),
        provider=provider if provider is not None else static_provider(),  # type: ignore[arg-type]
        tools=tools or BuildToolRegistry([upper_tool()]),
        config=config or EngineConfig(jobs=2, llm_concurrency=2, guard="strict"),
        **kwargs,
    )


def workspace_with(files: dict[str, str] | None = None) -> MemoryWorkspace:
    base: dict[str, bytes | str] = {
        "in/a.txt": "1",
        "in/b.txt": "2",
        "iab.toml": CONFIG_TEXT,
        "templates/shout.txt": TEMPLATE_TEXT,
    }
    base.update(files or {})
    return MemoryWorkspace(base)
