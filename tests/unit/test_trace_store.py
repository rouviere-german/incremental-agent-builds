"""The SQLite trace store: round trips, row authentication, runs and readiness."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from incremental_agent_builds.engine.errors import IntegrityError
from incremental_agent_builds.engine.hashing import value_hash
from incremental_agent_builds.engine.store import (
    ConstructiveEntry,
    Dep,
    DepKind,
    StepRecord,
    Trace,
    TraceStore,
    TraceUsage,
    decode_deps,
    encode_deps,
)

pytestmark = pytest.mark.unit

_DEPS = (
    (Dep(DepKind.RULE, "echo:{}", value_hash("a")), Dep(DepKind.RULE, "echo:{}2", value_hash("b"))),
    (Dep(DepKind.FILE, "in/a.txt", value_hash("x")),),
    (Dep(DepKind.CONFIG, "iab.toml#demo.factor", value_hash(2)),),
)


def _trace(key_id: str = "doubled:{}", **overrides: object) -> Trace:
    base: dict[str, object] = {
        "key_id": key_id,
        "rule": "doubled",
        "version": 3,
        "deps": _DEPS,
        "result_hash": value_hash("11"),
        "built_run": 4,
        "changed_run": 2,
        "usage": TraceUsage(
            llm_steps=2, provider_calls=1, input_tokens=90, output_tokens=12, usd_micros=7
        ),
        "origin": "executed",
    }
    base.update(overrides)
    return Trace(**base)  # type: ignore[arg-type]


@pytest.fixture
def store() -> TraceStore:
    return TraceStore(":memory:", mac_key=os.urandom(32))


def test_a_trace_round_trips_with_its_dependency_groups_and_usage(store: TraceStore) -> None:
    trace = _trace()
    store.put_traces([trace])
    loaded = store.get_trace(trace.key_id)
    assert loaded == trace
    assert loaded is not None
    assert loaded.dep_count() == 4
    assert store.load_traces() == {trace.key_id: trace}


def test_putting_a_trace_twice_replaces_it(store: TraceStore) -> None:
    store.put_traces([_trace()])
    store.put_traces([_trace(result_hash=value_hash("22"))])
    loaded = store.get_trace("doubled:{}")
    assert loaded is not None
    assert loaded.result_hash == value_hash("22")
    assert len(store.load_traces()) == 1


def test_deleting_traces_removes_them(store: TraceStore) -> None:
    store.put_traces([_trace(), _trace("echo:{}", rule="echo")])
    store.delete_traces(["doubled:{}"])
    assert set(store.load_traces()) == {"echo:{}"}


def test_dependency_groups_survive_encoding(store: TraceStore) -> None:
    assert decode_deps(encode_deps(_DEPS)) == _DEPS


@pytest.mark.parametrize(
    "text",
    ["{}", '["not a group"]', "[[]]", '[[["file","a"]]]', '[[["nope","a","b"]]]', "[[[1,2,3]]]"],
)
def test_malformed_dependency_encodings_are_refused(text: str) -> None:
    with pytest.raises(ValueError, match=r"dep|group|list|is not a valid"):
        decode_deps(text)


def test_a_tampered_row_is_reported_and_treated_as_absent(store: TraceStore) -> None:
    store.put_traces([_trace()])
    store.raw_execute(
        "UPDATE traces SET result_hash = ? WHERE key_id = ?", ("0" * 64, "doubled:{}")
    )
    assert store.get_trace("doubled:{}") is None
    assert store.load_traces() == {}
    assert store.integrity_events[-1].table == "traces"
    assert store.integrity_events[-1].reason == "MAC mismatch"


def test_a_row_with_a_valid_mac_but_broken_json_is_reported(store: TraceStore) -> None:
    """A row can only get here by a store bug or a partial write; it must not crash a build."""
    from incremental_agent_builds.engine.store import _mac

    key = store._key
    parts = ("trace", "broken:{}", "echo", "1", "not json", "0" * 64, "1", "1", "{}", "executed")
    store.raw_execute(
        "INSERT INTO traces (key_id, rule, version, deps, result_hash, built_run, changed_run, "
        "usage, origin, mac) VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("broken:{}", "echo", 1, "not json", "0" * 64, 1, 1, "{}", "executed", _mac(key, *parts)),
    )
    assert store.get_trace("broken:{}") is None
    assert store.integrity_events[-1].reason == "malformed row"


def test_steps_round_trip_and_tampering_drops_them(store: TraceStore) -> None:
    step = StepRecord(
        "a" * 64, "llm", value_hash({"x": 1}), {"model": "m", "input_tokens": 5}, 12.5
    )
    store.put_steps([step])
    assert store.load_steps() == {step.fingerprint: step}
    store.raw_execute(
        "UPDATE steps SET meta = ? WHERE fingerprint = ?", ('{"model":"other"}', step.fingerprint)
    )
    assert store.load_steps() == {}
    assert store.integrity_events[-1].table == "steps"


def test_constructive_entries_are_deduplicated_by_entry_id(store: TraceStore) -> None:
    body = {
        "kind": "trace",
        "key_id": "echo:{}",
        "rule": "echo",
        "version": 1,
        "deps": [],
        "result_hash": "0" * 64,
        "usage": {},
    }
    entry = ConstructiveEntry("e1", "echo:{}", body, "signer1", "sig")
    assert store.put_constructive([entry]) == 1
    assert store.put_constructive([entry]) == 0, "the same entry id is ignored, not duplicated"
    assert store.constructive_count() == 1
    found = store.constructive_for("echo:{}")
    assert len(found) == 1
    assert found[0].signer == "signer1"
    assert found[0].result_hash == "0" * 64
    assert found[0].version == 1


def test_a_tampered_constructive_row_is_skipped(store: TraceStore) -> None:
    body = {
        "kind": "trace",
        "key_id": "echo:{}",
        "rule": "echo",
        "version": 1,
        "deps": [],
        "result_hash": "0" * 64,
        "usage": {},
    }
    store.put_constructive([ConstructiveEntry("e1", "echo:{}", body, "signer1", "sig")])
    store.raw_execute("UPDATE constructive SET signer = ? WHERE entry_id = ?", ("attacker", "e1"))
    assert store.constructive_for("echo:{}") == []
    assert store.integrity_events[-1].table == "constructive"


def test_runs_are_numbered_and_recorded(store: TraceStore) -> None:
    assert store.last_run_id() == 0
    first = store.begin_run()
    second = store.begin_run()
    assert (first, second) == (1, 2)
    store.finish_run(second, {"ok": True})
    assert store.last_run_id() == 2


def test_check_writable_is_the_readiness_probe(store: TraceStore) -> None:
    assert store.check_writable() is True


def test_a_short_mac_key_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 32 bytes"):
        TraceStore(":memory:", mac_key=b"short")


def test_an_on_disk_store_creates_its_parent_directory(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "nested" / "traces.sqlite", mac_key=os.urandom(32))
    store.put_traces([_trace()])
    store.close()
    reopened = TraceStore(tmp_path / "nested" / "traces.sqlite", mac_key=os.urandom(32))
    assert reopened.get_trace("doubled:{}") is None, (
        "a different MAC key must not authenticate rows"
    )
    reopened.close()


def test_rows_written_with_one_key_verify_with_the_same_key(tmp_path: Path) -> None:
    key = os.urandom(32)
    path = tmp_path / "traces.sqlite"
    first = TraceStore(path, mac_key=key)
    first.put_traces([_trace()])
    first.close()
    second = TraceStore(path, mac_key=key)
    assert second.get_trace("doubled:{}") == _trace()
    second.close()


def test_trace_usage_parses_partial_json() -> None:
    assert TraceUsage.from_json({"llm_steps": 2}).llm_steps == 2
    assert TraceUsage.from_json({}).tokens == 0
    with pytest.raises(ValueError, match="must be an object"):
        TraceUsage.from_json([1, 2])


def test_allocating_a_run_id_reports_store_failures(store: TraceStore) -> None:
    store.close()
    with pytest.raises((IntegrityError, Exception)):
        store.begin_run()
