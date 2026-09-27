"""Random rule programs with value-dependent (dynamic) dependencies, for property tests and benches.

A program has ``n_files`` integer input files and a list of nodes. Node ``i`` reads some files,
needs a fixed group of lower-numbered nodes, and then needs *one* more node chosen by the value
of a selector file (a dynamic dependency). It may read a config value and make an LLM step. Its
value is a hash-mix of everything it read, reduced modulo a small number for some nodes, so
re-executions often produce the same value and early cutoff has something to do.

Because static and dynamic dependencies point to lower-numbered nodes, programs are acyclic;
:func:`with_cycle` and ``back_edges`` inject cycles for the cycle-detection properties.
"""

from __future__ import annotations

import os
import random
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from pydantic import BaseModel, ConfigDict, JsonValue

from incremental_agent_builds.agentic.providers.base import LLMRequest, LLMResponse
from incremental_agent_builds.agentic.providers.fake import FakeProvider, PolicyResult
from incremental_agent_builds.engine.cas import MemoryContentStore
from incremental_agent_builds.engine.context import BuildContext
from incremental_agent_builds.engine.engine import BuildEngine, BuildResult, EngineConfig
from incremental_agent_builds.engine.keys import BuildKey
from incremental_agent_builds.engine.oracle import compare_live, unnecessary_reruns
from incremental_agent_builds.engine.rules import RuleSet, rule
from incremental_agent_builds.engine.store import TraceStore
from incremental_agent_builds.engine.workspace import MemoryWorkspace

__all__ = [
    "CONFIG_TEXT",
    "TEMPLATE_TEXT",
    "CaseReport",
    "Edit",
    "Node",
    "NodeSpec",
    "Program",
    "apply_edit",
    "initial_workspace",
    "make_synthetic_provider",
    "program_rules",
    "random_program",
    "run_case",
    "with_cycle",
]

TEMPLATE_TEXT = "Derive a number from {{acc}}.\n"
CONFIG_TEXT = '[synthetic]\nsalt = 3\nmodel = "fake-model-a"\n'
_MOD = 1_000_003


class Node(BuildKey):
    i: int


class SyntheticOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


@dataclass(frozen=True, slots=True)
class NodeSpec:
    reads: tuple[int, ...] = ()
    static_deps: tuple[int, ...] = ()
    selector: int | None = None
    choices: tuple[int, ...] = ()
    modulus: int = _MOD
    uses_config: bool = False
    uses_llm: bool = False
    back_edges: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Program:
    n_files: int
    nodes: tuple[NodeSpec, ...]

    @staticmethod
    def file(index: int) -> str:
        return f"in/f{index}.txt"


@dataclass(frozen=True, slots=True)
class Edit:
    """``set`` writes a value, ``delete`` removes a file, ``config``/``template`` edit those."""

    kind: str
    target: int = 0
    value: int = 0


@dataclass
class _Mixer:
    acc: int

    def add(self, value: JsonValue) -> None:
        number = value if isinstance(value, int) else len(str(value))
        self.acc = (self.acc * 31 + number + 7) % _MOD


def _parse(text: str | None) -> int:
    if text is None:
        return -1
    try:
        return int(text.strip())
    except ValueError:
        return len(text)


def program_rules(program: Program) -> RuleSet:
    @rule(Node)
    async def node(ctx: BuildContext, key: Node) -> JsonValue:
        spec = program.nodes[key.i]
        mix = _Mixer(key.i)
        for index in spec.reads:
            mix.add(_parse(await ctx.read(program.file(index))))
        if spec.static_deps:
            for value in await ctx.need_all([Node(i=j) for j in spec.static_deps]):
                mix.add(value)
        if spec.back_edges:
            for value in await ctx.need_all([Node(i=j) for j in spec.back_edges]):
                mix.add(value)
        if spec.selector is not None and spec.choices:
            selected = _parse(await ctx.read(program.file(spec.selector)))
            mix.add(await ctx.need(Node(i=spec.choices[selected % len(spec.choices)])))
        if spec.uses_config:
            mix.add(await ctx.config("synthetic.salt"))
        if spec.uses_llm:
            model = await ctx.config("synthetic.model")
            out = await ctx.llm(
                "synthetic", {"acc": mix.acc}, model=str(model), schema=SyntheticOut, max_tokens=16
            )
            mix.add(out.value)
        return mix.acc % spec.modulus

    return RuleSet([node])


class _SyntheticPolicy:
    def __call__(self, request: LLMRequest, rng: random.Random, turn: int) -> PolicyResult:
        del rng, turn
        value = int(request.sha256()[:8], 16) % 97
        return LLMResponse(
            model=request.model or "fake-model",
            stop_reason="end_turn",
            text=f'{{"value": {value}}}',
            parsed={"value": value},
        )


def make_synthetic_provider() -> FakeProvider:
    return FakeProvider(_SyntheticPolicy())


def initial_workspace(program: Program, values: Sequence[int]) -> MemoryWorkspace:
    files: dict[str, bytes | str] = {
        program.file(i): str(values[i % len(values)] if values else i)
        for i in range(program.n_files)
    }
    files["iab.toml"] = CONFIG_TEXT
    files["templates/synthetic.txt"] = TEMPLATE_TEXT
    return MemoryWorkspace(files)


def apply_edit(workspace: MemoryWorkspace, program: Program, edit: Edit) -> None:
    if edit.kind == "set":
        workspace.write(program.file(edit.target % program.n_files), str(edit.value))
    elif edit.kind == "delete":
        workspace.delete(program.file(edit.target % program.n_files))
    elif edit.kind == "config":
        model = "fake-model-a" if edit.value % 2 == 0 else "fake-model-b"
        workspace.write("iab.toml", f'[synthetic]\nsalt = {edit.value % 5}\nmodel = "{model}"\n')
    elif edit.kind == "template":
        workspace.write("templates/synthetic.txt", TEMPLATE_TEXT + "#" * (edit.value % 4))
    else:
        raise ValueError(f"unknown edit kind {edit.kind!r}")


def random_program(
    rng: random.Random,
    *,
    n_nodes: int,
    n_files: int,
    window: int = 50,
    llm_fraction: float = 0.1,
    max_static: int = 3,
) -> Program:
    """A random acyclic program (used by benches; tests use hypothesis strategies)."""
    nodes: list[NodeSpec] = []
    for i in range(n_nodes):
        low = max(0, i - window)
        earlier = list(range(low, i))
        static = tuple(sorted(rng.sample(earlier, min(len(earlier), rng.randint(0, max_static)))))
        choices = tuple(rng.sample(earlier, min(len(earlier), rng.randint(0, 3))))
        nodes.append(
            NodeSpec(
                reads=tuple(rng.sample(range(n_files), min(n_files, rng.randint(0, 2)))),
                static_deps=static,
                selector=rng.randrange(n_files) if choices else None,
                choices=choices,
                modulus=rng.choice([2, 3, 5, _MOD, _MOD]),
                uses_config=rng.random() < 0.1,
                uses_llm=rng.random() < llm_fraction,
            )
        )
    return Program(n_files=n_files, nodes=tuple(nodes))


def with_cycle(program: Program, members: Sequence[int]) -> Program:
    """Add edges members[k] -> members[k-1] and members[0] -> members[-1]: a guaranteed cycle."""
    if not members:
        raise ValueError("a cycle needs at least one member")
    nodes = list(program.nodes)
    ring = list(members)
    for position, member in enumerate(ring):
        target = ring[position - 1]
        spec = nodes[member]
        nodes[member] = replace(spec, back_edges=(*spec.back_edges, target))
    return replace(program, nodes=tuple(nodes))


@dataclass
class CaseReport:
    """Outcome of one program x edit-sequence case (see :func:`run_case`)."""

    builds: int = 0
    edits: int = 0
    mismatches: list[str] = field(default_factory=list)
    unnecessary: list[str] = field(default_factory=list)
    executed: int = 0
    reused: int = 0
    early_cutoffs: int = 0
    cycles: int = 0
    failures: int = 0


def _engine(program: Program, workspace: MemoryWorkspace, config: EngineConfig) -> BuildEngine:
    return BuildEngine(
        rules=program_rules(program),
        workspace=workspace,
        store=TraceStore(":memory:", mac_key=os.urandom(32)),
        cas=MemoryContentStore(),
        provider=make_synthetic_provider(),
        config=config,
    )


def run_case(
    program: Program,
    values: Sequence[int],
    edits: Sequence[Edit],
    *,
    config: EngineConfig | None = None,
    targets: Sequence[int] | None = None,
) -> CaseReport:
    """Build, then after every edit compare the incremental build with a clean rebuild.

    Records live-set/value mismatches (clean equivalence) and executed keys none of whose
    recorded dependencies changed between the two clean builds (minimality).
    """
    cfg = config or EngineConfig(guard="strict")
    workspace = initial_workspace(program, values)
    incremental = _engine(program, workspace, cfg)
    wanted = [Node(i=i) for i in (targets if targets is not None else range(len(program.nodes)))]
    report = CaseReport()

    def account(result: BuildResult) -> None:
        report.builds += 1
        report.cycles += len(result.cycles)
        report.failures += len(result.failures)

    account(incremental.build_sync(wanted))
    for edit in edits:
        apply_edit(workspace, program, edit)
        report.edits += 1
        before = incremental.store.load_traces()
        result = incremental.build_sync(wanted)
        account(result)
        report.executed += result.stats.executed
        report.reused += result.stats.reused
        report.early_cutoffs += result.stats.early_cutoffs
        snapshot = MemoryWorkspace(dict(workspace.files))
        clean = _engine(program, snapshot, cfg)
        clean_result = clean.build_sync(wanted)
        account(clean_result)
        report.mismatches += compare_live(incremental, result.live, clean, clean_result.live)
        if result.ok and clean_result.ok:
            report.unnecessary += unnecessary_reruns(
                incremental, before, result.executed, clean_result.live
            )
    return report
