"""The hermeticity guard: I/O inside a rule body must go through BuildContext.

Correctness rests on every input being recorded, so the guard is the test-mode check that a rule
did not read something behind the engine's back. ``strict`` raises, ``warn`` records, ``off``
ignores. It is a guard for honest mistakes, not a sandbox (see docs/adr/0004): these tests pin the
behaviour it does have and name the limits it does not cover.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
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


# --- which audit events count -------------------------------------------------------------------
#
# The hook's decision is "could a rule body have read something through this event?". The watched
# set is a list of exact names plus three prefixes, and both are pinned below.
#
# Every case runs twice. Through ``sys.audit`` it proves the hook is installed and really sees
# interpreter events; through a direct call it exercises the same decision *with tracing enabled*,
# because CPython disables the tracer while an audit hook runs. Without the direct call, coverage
# reports the whole hook body as unexecuted however thoroughly it is tested.

type Emitter = Callable[..., None]


@pytest.fixture(params=["sys.audit", "direct"])
def emit(request: pytest.FixtureRequest) -> Emitter:
    hermetic.install_guard()
    if request.param == "sys.audit":
        return sys.audit

    def direct(event: str, *args: object) -> None:
        hermetic._hook(event, args)

    return direct


_NAMED_EVENTS = [
    "open",
    "glob.glob",
    "sqlite3.connect",
    "subprocess.Popen",
    "ctypes.dlopen",
    "webbrowser.open",
]


@pytest.mark.parametrize("event", _NAMED_EVENTS)
def test_each_named_event_is_a_violation(emit: Emitter, event: str) -> None:
    with hermetic.rule_scope("k:{}", "warn"):
        emit(event, "target")
    assert [v.event for v in hermetic.violations()] == [event]


@pytest.mark.parametrize("event", ["os.rename", "shutil.copytree", "socket.getaddrinfo"])
def test_the_watched_prefixes_cover_whole_families(emit: Emitter, event: str) -> None:
    with hermetic.rule_scope("k:{}", "warn"):
        emit(event, "target")
    recorded = hermetic.violations()
    assert [v.event for v in recorded] == [event]
    assert recorded[0].key_id == "k:{}"
    assert recorded[0].detail == "'target'", "the first argument is recorded, repr'd and truncated"


@pytest.mark.parametrize("event", ["compile", "exec", "object.__getattr__", "opening"])
def test_events_that_read_nothing_from_the_workspace_are_ignored(emit: Emitter, event: str) -> None:
    """``opening`` is deliberately here: the non-prefix names match exactly, not by prefix."""
    with hermetic.rule_scope("k:{}", "strict"):
        emit(event, "target")
    assert hermetic.violations() == []


def test_a_watched_event_outside_any_rule_body_is_ignored(emit: Emitter) -> None:
    emit("open", "target")
    assert hermetic.violations() == []


def test_a_watched_event_under_a_permit_is_ignored(emit: Emitter) -> None:
    with hermetic.rule_scope("k:{}", "strict"), hermetic.io_permit():
        emit("open", "target")
    assert hermetic.violations() == []


def test_off_mode_records_nothing_even_inside_a_rule_scope(emit: Emitter) -> None:
    with hermetic.rule_scope("k:{}", "off"):
        emit("open", "target")
    assert hermetic.violations() == []


def test_strict_mode_raises_and_still_records(emit: Emitter) -> None:
    with (
        pytest.raises(HermeticityError, match="outside BuildContext"),
        hermetic.rule_scope("k:{}", "strict"),
    ):
        emit("os.remove", "target")
    assert [v.event for v in hermetic.violations()] == ["os.remove"]


def test_an_event_with_no_arguments_records_an_empty_detail(emit: Emitter) -> None:
    with hermetic.rule_scope("k:{}", "warn"):
        emit("os.bare")
    assert hermetic.violations()[0].detail == ""


def test_a_long_argument_is_truncated_in_the_detail(emit: Emitter) -> None:
    with hermetic.rule_scope("k:{}", "warn"):
        emit("open", "x" * 500)
    assert len(hermetic.violations()[0].detail) == 200


def test_recorded_violations_are_capped(emit: Emitter) -> None:
    """The list grows from rule code: a rule in a loop must not be able to grow it without limit."""
    with hermetic.rule_scope("k:{}", "warn"):
        for index in range(hermetic._MAX_RECORDED + 25):
            emit("os.probe", index)
    assert len(hermetic.violations()) == hermetic._MAX_RECORDED


def test_an_import_inside_a_rule_body_is_not_a_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Importing reads files through the import machinery, which the hook exempts by frame walk.

    Without the exemption the first rule to import something lazily would fail in strict mode, and
    the engine does import lazily from inside rule bodies.
    """
    hermetic.install_guard()
    name = "iab_hermeticity_probe_module"
    (tmp_path / f"{name}.py").write_text("VALUE = 7\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, name, raising=False)

    with hermetic.rule_scope("k:{}", "strict"):
        module = importlib.import_module(name)

    assert module.VALUE == 7
    assert hermetic.violations() == [], "the import machinery's own reads are not rule reads"


def test_violations_returns_a_copy_that_clearing_does_not_empty() -> None:
    hermetic.install_guard()
    with hermetic.rule_scope("k:{}", "warn"):
        sys.audit("os.stat", "x")
    snapshot = hermetic.violations()
    hermetic.clear_violations()
    assert hermetic.violations() == []
    assert len(snapshot) == 1, "the caller's list is its own"
