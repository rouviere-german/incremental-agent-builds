"""The two properties the whole repo exists to guarantee.

**Clean equivalence.** For a random rule program (whose dependencies are chosen at runtime by the
values it reads) and a random sequence of source edits, the incremental store after every edit must
hold exactly the keys a build from an empty store produces, with equal values. This is checked by
rebuilding from scratch after each edit and comparing live sets value by value
(:func:`compare_live`), so a stale value or a missing key fails here.

**Minimality.** Every rule the incremental build executed must have had at least one *direct*
recorded dependency whose value differs from what it saw last time, judged against the clean
rebuild rather than against the engine's own bookkeeping (:func:`unnecessary_reruns`). A build that
re-runs something it did not need to fails here even though it is still correct.

The strategies deliberately generate the shapes that break naive caches: nodes whose dependency set
depends on a value (``selector`` / ``choices``), small moduli so re-executions often produce the
same value (early cutoff), deleted files, edits to the prompt template and to the model id, and
LLM steps whose output is a value like any other.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from incremental_agent_builds.engine.engine import EngineConfig
from incremental_agent_builds.workload.synthetic import (
    Edit,
    NodeSpec,
    Program,
    run_case,
)

pytestmark = pytest.mark.property

_MAX_NODES = 12
_MAX_FILES = 5


@st.composite
def programs(draw: st.DrawFn, *, max_nodes: int = _MAX_NODES) -> Program:
    """A random acyclic program: node ``i`` may only depend on nodes below ``i``."""
    n_nodes = draw(st.integers(min_value=1, max_value=max_nodes))
    n_files = draw(st.integers(min_value=1, max_value=_MAX_FILES))
    nodes: list[NodeSpec] = []
    for index in range(n_nodes):
        earlier = list(range(index))
        reads = draw(
            st.lists(st.integers(min_value=0, max_value=n_files - 1), max_size=2, unique=True)
        )
        static = (
            draw(st.lists(st.sampled_from(earlier), max_size=3, unique=True)) if earlier else []
        )
        choices = (
            draw(st.lists(st.sampled_from(earlier), max_size=3, unique=True)) if earlier else []
        )
        nodes.append(
            NodeSpec(
                reads=tuple(sorted(reads)),
                static_deps=tuple(sorted(static)),
                selector=draw(st.integers(min_value=0, max_value=n_files - 1)) if choices else None,
                choices=tuple(choices),
                # Small moduli make re-executions collide, which is what early cutoff exploits.
                modulus=draw(st.sampled_from([2, 3, 5, 1_000_003])),
                uses_config=draw(st.booleans()),
                uses_llm=draw(st.booleans()),
            )
        )
    return Program(n_files=n_files, nodes=tuple(nodes))


edits = st.builds(
    Edit,
    kind=st.sampled_from(["set", "set", "set", "delete", "config", "template"]),
    target=st.integers(min_value=0, max_value=_MAX_FILES - 1),
    value=st.integers(min_value=-4, max_value=12),
)
values = st.lists(st.integers(min_value=-5, max_value=20), min_size=1, max_size=6)


# No max_examples here on purpose: this test follows the active hypothesis profile, so the same
# code runs 60 cases locally (dev), 200 in the standard CI job and 1,000 in the soak job.
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
@given(program=programs(), values=values, edit_list=st.lists(edits, min_size=1, max_size=4))
def test_incremental_build_equals_a_clean_rebuild_after_every_edit(
    program: Program, values: list[int], edit_list: list[Edit]
) -> None:
    """Clean equivalence and minimality, checked after each edit of a random edit sequence."""
    report = run_case(program, values, edit_list)
    assert report.mismatches == [], f"incremental build diverged from a clean rebuild: {report}"
    assert report.unnecessary == [], f"rules re-ran with no changed dependency: {report}"
    assert report.cycles == 0
    assert report.failures == 0


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    program=programs(max_nodes=8), values=values, edit_list=st.lists(edits, min_size=1, max_size=3)
)
def test_equivalence_holds_for_a_single_target_and_one_job(
    program: Program, values: list[int], edit_list: list[Edit]
) -> None:
    """The same properties with one worker and only the top node as the target.

    A single target exercises the part of the graph reachable from it; with ``jobs=1`` the
    suspending scheduler must still make progress (a blocking scheduler would deadlock here).
    """
    report = run_case(
        program,
        values,
        edit_list,
        config=EngineConfig(jobs=1, llm_concurrency=1, guard="strict"),
        targets=[len(program.nodes) - 1],
    )
    assert report.mismatches == []
    assert report.unnecessary == []
    assert report.failures == 0


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    program=programs(max_nodes=8), values=values, edit_list=st.lists(edits, min_size=1, max_size=3)
)
def test_disabling_early_cutoff_keeps_values_correct(
    program: Program, values: list[int], edit_list: list[Edit]
) -> None:
    """The early-cutoff ablation must cost more work, never change a value.

    Minimality is not asserted here: without early cutoff, a re-executed key counts as changed even
    when its value did not, so dependents re-run by design.
    """
    report = run_case(
        program,
        values,
        edit_list,
        config=EngineConfig(jobs=2, llm_concurrency=2, early_cutoff=False, guard="strict"),
    )
    assert report.mismatches == []
    assert report.failures == 0


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    program=programs(max_nodes=8), values=values, edit_list=st.lists(edits, min_size=1, max_size=3)
)
def test_disabling_step_memoization_keeps_values_correct(
    program: Program, values: list[int], edit_list: list[Edit]
) -> None:
    """Without LLM/tool step memoization every re-run calls the provider again, with equal results.

    This is the make-style baseline: it proves the deterministic provider is a function of the
    request, so reusing a recorded output can never change a value.
    """
    report = run_case(
        program,
        values,
        edit_list,
        config=EngineConfig(
            jobs=2, llm_concurrency=2, early_cutoff=False, step_memo=False, guard="strict"
        ),
    )
    assert report.mismatches == []
    assert report.failures == 0
