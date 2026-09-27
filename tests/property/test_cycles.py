"""Every injected dependency cycle must raise CycleError instead of deadlocking.

Cycles are discovered at runtime, because the dependency set of a rule is not known before it runs.
The scheduler therefore keeps a waits-for graph over running keys and checks it before every
suspension. These properties inject cycles of every length from 1 (a rule that needs itself) upwards
and assert the error names a real cycle, that the build still terminates, and that unrelated keys
in the same build are unaffected.

The tests carry a hypothesis deadline via the profile, so a deadlock shows up as pytest-timeout-free
failure: :func:`asyncio.run` would otherwise never return.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from incremental_agent_builds.engine.engine import BuildEngine, EngineConfig
from incremental_agent_builds.engine.errors import CycleError
from incremental_agent_builds.workload.synthetic import (
    Node,
    Program,
    initial_workspace,
    make_synthetic_provider,
    program_rules,
    with_cycle,
)

pytestmark = pytest.mark.property


def _chain(n_nodes: int) -> Program:
    """A plain chain: node i needs node i-1, so any added back edge creates a cycle."""
    from incremental_agent_builds.workload.synthetic import NodeSpec

    nodes = [
        NodeSpec(reads=(0,), static_deps=(index - 1,) if index else ()) for index in range(n_nodes)
    ]
    return Program(n_files=1, nodes=tuple(nodes))


def _engine(program: Program, *, jobs: int) -> BuildEngine:
    import os

    from incremental_agent_builds.engine.cas import MemoryContentStore
    from incremental_agent_builds.engine.store import TraceStore

    return BuildEngine(
        rules=program_rules(program),
        workspace=initial_workspace(program, [1]),
        store=TraceStore(":memory:", mac_key=os.urandom(32)),
        cas=MemoryContentStore(),
        provider=make_synthetic_provider(),
        config=EngineConfig(jobs=jobs, llm_concurrency=jobs, guard="strict"),
    )


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n_nodes=st.integers(min_value=1, max_value=8),
    cycle_length=st.integers(min_value=1, max_value=8),
    jobs=st.integers(min_value=1, max_value=4),
)
def test_every_injected_cycle_raises_cycle_error(
    n_nodes: int, cycle_length: int, jobs: int
) -> None:
    length = min(cycle_length, n_nodes)
    program = with_cycle(_chain(n_nodes), list(range(length)))
    engine = _engine(program, jobs=jobs)
    result = engine.build_sync([Node(i=i) for i in range(n_nodes)])
    assert not result.ok
    assert result.cycles, "a cycle was injected but none was reported"
    for cycle in result.cycles:
        assert len(cycle) >= 2, cycle
        assert cycle[0] == cycle[-1], f"a reported cycle must close on itself: {cycle}"
    with pytest.raises(CycleError):
        result.raise_for_failures()


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(n_nodes=st.integers(min_value=3, max_value=8), jobs=st.integers(min_value=1, max_value=4))
def test_a_self_dependency_is_a_cycle_of_length_one(n_nodes: int, jobs: int) -> None:
    """``ctx.need`` on the rule's own key must raise immediately, not wait for itself."""
    program = with_cycle(_chain(n_nodes), [n_nodes - 1])
    engine = _engine(program, jobs=jobs)
    result = engine.build_sync([Node(i=n_nodes - 1)])
    assert not result.ok
    assert result.cycles
    assert result.cycles[0][0] == Node(i=n_nodes - 1).key_id


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(n_nodes=st.integers(min_value=4, max_value=8))
def test_keys_outside_the_cycle_still_build(n_nodes: int) -> None:
    """A cycle among the highest nodes must not stop the acyclic part of the same build."""
    program = with_cycle(_chain(n_nodes), [n_nodes - 2, n_nodes - 1])
    engine = _engine(program, jobs=2)
    result = engine.build_sync([Node(i=i) for i in range(n_nodes)])
    assert result.cycles
    healthy = Node(i=0).key_id
    assert healthy in result.target_hashes, "an independent target must still be built"
    assert healthy not in result.failures
