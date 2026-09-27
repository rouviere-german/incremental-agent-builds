"""The hermeticity guard: I/O inside a rule body must go through BuildContext.

Correctness rests on every input being recorded, so the guard is the test-mode check that a rule
did not read something behind the engine's back. ``strict`` raises, ``warn`` records, ``off``
ignores. It is a guard for honest mistakes, not a sandbox (see docs/adr/0004): these tests pin the
behaviour it does have and name the limits it does not cover.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import JsonValue

from incremental_agent_builds.engine import hermetic
from incremental_agent_builds.engine.context import BuildContext
from incremental_agent_builds.engine.engine import EngineConfig
from incremental_agent_builds.engine.errors import HermeticityError
from incremental_agent_builds.engine.keys import BuildKey
from incremental_agent_builds.engine.rules import RuleSet, rule
from tests.engine_helpers import make_engine, workspace_with

pytestmark = pytest.mark.unit


class Sneaky(BuildKey):
    path: str


@pytest.fixture(autouse=True)
def _clear_violations() -> None:
    hermetic.clear_violations()


def _sneaky_rules(target: Path) -> RuleSet:
    @rule(Sneaky)
    async def sneaky(ctx: BuildContext, key: Sneaky) -> JsonValue:
        # Reads the file directly instead of through ctx.read: exactly the mistake the guard exists
        # to catch, because nothing about this read reaches the trace.
        return target.read_text(encoding="utf-8")

    return RuleSet([sneaky])


def test_strict_mode_fails_the_rule_that_read_behind_the_engines_back(tmp_path: Path) -> None:
    secret = tmp_path / "hidden.txt"
    secret.write_text("hidden", encoding="utf-8")
    engine = make_engine(
        workspace_with(),
        _sneaky_rules(secret),
        config=EngineConfig(jobs=1, llm_concurrency=1, guard="strict"),
    )
    result = engine.build_sync([Sneaky(path="x")])
    assert not result.ok
    failure = result.failures[Sneaky(path="x").key_id]
    assert "HermeticityError" in failure
    assert any(v.event == "open" for v in hermetic.violations())


def test_warn_mode_records_the_violation_and_completes_the_build(tmp_path: Path) -> None:
    secret = tmp_path / "hidden.txt"
    secret.write_text("hidden", encoding="utf-8")
    engine = make_engine(
        workspace_with(),
        _sneaky_rules(secret),
        config=EngineConfig(jobs=1, llm_concurrency=1, guard="warn"),
    )
    result = engine.build_sync([Sneaky(path="x")])
    assert result.ok, "warn mode must not fail the build"
    assert result.stats.hermetic_violations >= 1
    assert engine.value(Sneaky(path="x")) == "hidden"


def test_off_mode_records_nothing(tmp_path: Path) -> None:
    secret = tmp_path / "hidden.txt"
    secret.write_text("hidden", encoding="utf-8")
    engine = make_engine(
        workspace_with(),
        _sneaky_rules(secret),
        config=EngineConfig(jobs=1, llm_concurrency=1, guard="off"),
    )
    result = engine.build_sync([Sneaky(path="x")])
    assert result.ok
    assert result.stats.hermetic_violations == 0
    assert hermetic.violations() == []


def test_context_reads_are_permitted_in_strict_mode() -> None:
    """The engine's own reads happen under an I/O permit, so they are not violations."""
    from tests.engine_helpers import Sum, counting_rules

    rules, _ = counting_rules()
    engine = make_engine(
        workspace_with(), rules, config=EngineConfig(jobs=2, llm_concurrency=2, guard="strict")
    )
    result = engine.build_sync([Sum(pattern="in/*.txt")])
    assert result.ok
    assert hermetic.violations() == []
    assert result.stats.hermetic_violations == 0


def test_the_rule_marker_is_scoped_to_the_rule_body() -> None:
    assert hermetic.active_rule() is None
    with hermetic.rule_scope("echo:{}", "strict"):
        assert hermetic.active_rule() == "echo:{}"
        with hermetic.io_permit():
            assert hermetic.active_rule() == "echo:{}"
    assert hermetic.active_rule() is None


def test_the_guard_raises_only_inside_a_rule_scope(tmp_path: Path) -> None:
    hermetic.install_guard()
    probe = tmp_path / "probe.txt"
    probe.write_text("x", encoding="utf-8")
    probe.read_text(encoding="utf-8")  # outside any rule: fine
    with pytest.raises(HermeticityError), hermetic.rule_scope("k:{}", "strict"):
        probe.read_text(encoding="utf-8")
    with hermetic.rule_scope("k:{}", "strict"), hermetic.io_permit():
        assert probe.read_text(encoding="utf-8") == "x"


def test_clear_rule_marker_lets_engine_tasks_run_outside_the_scope(tmp_path: Path) -> None:
    """New scheduler tasks copy the context and clear the marker, so they are not rule bodies."""
    hermetic.install_guard()
    probe = tmp_path / "probe.txt"
    probe.write_text("x", encoding="utf-8")
    with hermetic.rule_scope("k:{}", "strict"):
        hermetic.clear_rule_marker()
        assert probe.read_text(encoding="utf-8") == "x"


def test_installing_the_guard_twice_is_harmless() -> None:
    hermetic.install_guard()
    hermetic.install_guard()
    assert hermetic.active_rule() is None
