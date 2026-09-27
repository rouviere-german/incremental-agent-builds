"""SQLite store for verifying traces, step records, imported constructive traces and runs.

A *verifying trace* records, for one key, the ordered groups of dependencies the rule read during
its last execution (each with the hash of the value it saw) and the hash of the result. Values
themselves live in the content-addressed store; this database only holds hashes.

Every row carries an HMAC-SHA256 over its exact stored text, keyed with a per-store secret. A
row whose MAC does not verify (an edited dependency hash, a swapped result hash, a row copied to
another key) is treated as absent and reported, so the key is recomputed instead of trusting it.
A store rolled back to an older but correctly MAC'd state is harmless: traces are always checked
against the current inputs, so an old trace only causes extra work, never a stale value.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from incremental_agent_builds.engine.errors import IntegrityError

__all__ = [
    "ConstructiveEntry",
    "Dep",
    "DepGroups",
    "DepKind",
    "IntegrityEvent",
    "StepRecord",
    "Trace",
    "TraceStore",
    "TraceUsage",
    "decode_deps",
    "encode_deps",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS traces (
    key_id TEXT PRIMARY KEY,
    rule TEXT NOT NULL,
    version INTEGER NOT NULL,
    deps TEXT NOT NULL,
    result_hash TEXT NOT NULL,
    built_run INTEGER NOT NULL,
    changed_run INTEGER NOT NULL,
    usage TEXT NOT NULL,
    origin TEXT NOT NULL,
    mac TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS steps (
    fingerprint TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    result_hash TEXT NOT NULL,
    meta TEXT NOT NULL,
    recorded_at REAL NOT NULL,
    mac TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS constructive (
    entry_id TEXT PRIMARY KEY,
    key_id TEXT NOT NULL,
    body TEXT NOT NULL,
    signer TEXT NOT NULL,
    signature TEXT NOT NULL,
    imported_at REAL NOT NULL,
    mac TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS constructive_key ON constructive (key_id);
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    finished_at REAL,
    stats TEXT
);
"""


class DepKind(StrEnum):
    RULE = "rule"
    FILE = "file"
    GLOB = "glob"
    CONFIG = "config"
    LLM = "llm"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class Dep:
    """One recorded dependency: what was read (``ref``) and the hash of what it returned."""

    kind: DepKind
    ref: str
    hash: str


type DepGroups = tuple[tuple[Dep, ...], ...]


def encode_deps(groups: DepGroups) -> str:
    return json.dumps(
        [[[dep.kind.value, dep.ref, dep.hash] for dep in group] for group in groups],
        separators=(",", ":"),
        ensure_ascii=False,
    )


def decode_deps(text: str) -> DepGroups:
    raw = json.loads(text)
    if not isinstance(raw, list):
        # Malformed stored data, not a caller type error: the row is reported and treated as absent.
        raise ValueError("deps must be a list of groups")  # noqa: TRY004
    groups: list[tuple[Dep, ...]] = []
    for group in raw:
        if not isinstance(group, list) or not group:
            raise ValueError("each dependency group must be a non-empty list")
        items: list[Dep] = []
        for item in group:
            if not (
                isinstance(item, list) and len(item) == 3 and all(isinstance(x, str) for x in item)
            ):
                raise ValueError("each dependency must be [kind, ref, hash]")
            items.append(Dep(DepKind(item[0]), item[1], item[2]))
        groups.append(tuple(items))
    return tuple(groups)


@dataclass(frozen=True, slots=True)
class TraceUsage:
    """LLM usage attributed to one rule execution.

    ``llm_steps`` counts the LLM steps the rule made and the token fields hold what those steps
    cost when they were first computed (also when this execution reused a recorded output). The
    planner uses them to project the cost of re-running the rule. ``provider_calls`` counts calls
    that actually reached the provider during this execution.
    """

    llm_steps: int = 0
    provider_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd_micros: int = 0

    def to_json(self) -> dict[str, int]:
        return {
            "llm_steps": self.llm_steps,
            "provider_calls": self.provider_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "usd_micros": self.usd_micros,
        }

    @classmethod
    def from_json(cls, data: Any) -> TraceUsage:
        if not isinstance(data, dict):
            raise ValueError("usage must be an object")  # noqa: TRY004 - malformed stored data
        return cls(**{name: int(data.get(name, 0)) for name in cls.__dataclass_fields__})

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class Trace:
    key_id: str
    rule: str
    version: int
    deps: DepGroups
    result_hash: str
    built_run: int
    changed_run: int
    usage: TraceUsage = field(default_factory=TraceUsage)
    origin: str = "executed"

    def dep_count(self) -> int:
        return sum(len(group) for group in self.deps)


@dataclass(frozen=True, slots=True)
class StepRecord:
    """A memoized LLM or tool step: fingerprint -> result hash (plus usage metadata)."""

    fingerprint: str
    kind: str
    result_hash: str
    meta: dict[str, JsonValue]
    recorded_at: float


@dataclass(frozen=True, slots=True)
class ConstructiveEntry:
    """An imported, signature-verified constructive trace (key + dep hashes -> result)."""

    entry_id: str
    key_id: str
    body: dict[str, Any]
    signer: str
    signature: str

    @property
    def version(self) -> int:
        return int(self.body["version"])

    @property
    def result_hash(self) -> str:
        return str(self.body["result_hash"])

    @property
    def deps(self) -> DepGroups:
        return decode_deps(json.dumps(self.body["deps"]))


@dataclass(frozen=True, slots=True)
class IntegrityEvent:
    table: str
    ref: str
    reason: str


def _mac(key: bytes, *parts: str) -> str:
    message = "\x1f".join(("iab-store-v1", *parts)).encode("utf-8")
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def _trace_parts(trace: Trace, deps_text: str, usage_text: str) -> tuple[str, ...]:
    return (
        "trace",
        trace.key_id,
        trace.rule,
        str(trace.version),
        deps_text,
        trace.result_hash,
        str(trace.built_run),
        str(trace.changed_run),
        usage_text,
        trace.origin,
    )


class TraceStore:
    """Thread-safe SQLite access. ``path`` may be ``":memory:"`` for tests and benchmarks."""

    def __init__(self, path: Path | str, *, mac_key: bytes) -> None:
        if len(mac_key) < 32:
            raise ValueError("the store MAC key must be at least 32 bytes")
        self._key = mac_key
        self._path = str(path)
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False, timeout=30.0)
        with self._lock:
            if self._path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        self.integrity_events: list[IntegrityEvent] = []

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _report(self, table: str, ref: str, reason: str) -> None:
        self.integrity_events.append(IntegrityEvent(table, ref, reason))

    # --- traces ---------------------------------------------------------------------------

    def _trace_from_row(self, row: Sequence[Any]) -> Trace | None:
        key_id, rule, version, deps_text, result_hash, built, changed, usage_text, origin, mac = row
        parts = (
            "trace",
            key_id,
            rule,
            str(version),
            deps_text,
            result_hash,
            str(built),
            str(changed),
            usage_text,
            origin,
        )
        if not hmac.compare_digest(_mac(self._key, *parts), mac):
            self._report("traces", key_id, "MAC mismatch")
            return None
        try:
            deps = decode_deps(deps_text)
            usage = TraceUsage.from_json(json.loads(usage_text))
        except (ValueError, TypeError, json.JSONDecodeError):
            self._report("traces", key_id, "malformed row")
            return None
        return Trace(key_id, rule, int(version), deps, result_hash, built, changed, usage, origin)

    def load_traces(self) -> dict[str, Trace]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT key_id, rule, version, deps, result_hash, built_run, changed_run, usage, "
                "origin, mac FROM traces"
            ).fetchall()
        traces: dict[str, Trace] = {}
        for row in rows:
            trace = self._trace_from_row(row)
            if trace is not None:
                traces[trace.key_id] = trace
        return traces

    def get_trace(self, key_id: str) -> Trace | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT key_id, rule, version, deps, result_hash, built_run, changed_run, usage, "
                "origin, mac FROM traces WHERE key_id = ?",
                (key_id,),
            ).fetchone()
        return None if row is None else self._trace_from_row(row)

    def put_traces(self, traces: Iterable[Trace]) -> None:
        rows = []
        for trace in traces:
            deps_text = encode_deps(trace.deps)
            usage_text = json.dumps(trace.usage.to_json(), sort_keys=True, separators=(",", ":"))
            mac = _mac(self._key, *_trace_parts(trace, deps_text, usage_text))
            rows.append(
                (
                    trace.key_id,
                    trace.rule,
                    trace.version,
                    deps_text,
                    trace.result_hash,
                    trace.built_run,
                    trace.changed_run,
                    usage_text,
                    trace.origin,
                    mac,
                )
            )
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO traces (key_id, rule, version, deps, result_hash, "
                "built_run, changed_run, usage, origin, mac) VALUES (?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            self._conn.commit()

    def delete_traces(self, key_ids: Iterable[str]) -> None:
        with self._lock:
            self._conn.executemany("DELETE FROM traces WHERE key_id = ?", [(k,) for k in key_ids])
            self._conn.commit()

    # --- steps ------------------------------------------------------------------------------

    def _step_from_row(self, row: Sequence[Any]) -> StepRecord | None:
        fingerprint, kind, result_hash, meta_text, recorded_at, mac = row
        parts = ("step", fingerprint, kind, result_hash, meta_text, repr(float(recorded_at)))
        if not hmac.compare_digest(_mac(self._key, *parts), mac):
            self._report("steps", fingerprint, "MAC mismatch")
            return None
        try:
            meta = json.loads(meta_text)
        except json.JSONDecodeError:
            self._report("steps", fingerprint, "malformed row")
            return None
        if not isinstance(meta, dict):
            self._report("steps", fingerprint, "malformed row")
            return None
        return StepRecord(fingerprint, kind, result_hash, meta, float(recorded_at))

    def load_steps(self) -> dict[str, StepRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT fingerprint, kind, result_hash, meta, recorded_at, mac FROM steps"
            ).fetchall()
        steps: dict[str, StepRecord] = {}
        for row in rows:
            step = self._step_from_row(row)
            if step is not None:
                steps[step.fingerprint] = step
        return steps

    def put_steps(self, steps: Iterable[StepRecord]) -> None:
        rows = []
        for step in steps:
            meta_text = json.dumps(step.meta, sort_keys=True, separators=(",", ":"))
            parts = (
                "step",
                step.fingerprint,
                step.kind,
                step.result_hash,
                meta_text,
                repr(float(step.recorded_at)),
            )
            rows.append(
                (
                    step.fingerprint,
                    step.kind,
                    step.result_hash,
                    meta_text,
                    float(step.recorded_at),
                    _mac(self._key, *parts),
                )
            )
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO steps (fingerprint, kind, result_hash, meta, recorded_at, "
                "mac) VALUES (?,?,?,?,?,?)",
                rows,
            )
            self._conn.commit()

    # --- constructive (imported) traces -----------------------------------------------------

    def put_constructive(self, entries: Iterable[ConstructiveEntry]) -> int:
        rows = []
        now = time.time()
        for entry in entries:
            body = json.dumps(entry.body, sort_keys=True, separators=(",", ":"))
            parts = ("constructive", entry.entry_id, entry.key_id, body, entry.signer)
            rows.append(
                (
                    entry.entry_id,
                    entry.key_id,
                    body,
                    entry.signer,
                    entry.signature,
                    now,
                    _mac(self._key, *parts, entry.signature),
                )
            )
        with self._lock:
            before = self._conn.total_changes
            self._conn.executemany(
                "INSERT OR IGNORE INTO constructive (entry_id, key_id, body, signer, signature, "
                "imported_at, mac) VALUES (?,?,?,?,?,?,?)",
                rows,
            )
            self._conn.commit()
            return self._conn.total_changes - before

    def constructive_for(self, key_id: str) -> list[ConstructiveEntry]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT entry_id, key_id, body, signer, signature, mac FROM constructive "
                "WHERE key_id = ? ORDER BY imported_at DESC, entry_id",
                (key_id,),
            ).fetchall()
        entries: list[ConstructiveEntry] = []
        for entry_id, row_key, body_text, signer, signature, mac in rows:
            parts = ("constructive", entry_id, row_key, body_text, signer)
            if not hmac.compare_digest(_mac(self._key, *parts, signature), mac):
                self._report("constructive", entry_id, "MAC mismatch")
                continue
            try:
                body = json.loads(body_text)
            except json.JSONDecodeError:
                self._report("constructive", entry_id, "malformed row")
                continue
            entries.append(ConstructiveEntry(entry_id, row_key, body, signer, signature))
        return entries

    def constructive_count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM constructive").fetchone()
        return int(row[0])

    # --- runs -------------------------------------------------------------------------------

    def begin_run(self) -> int:
        with self._lock:
            cursor = self._conn.execute("INSERT INTO runs (started_at) VALUES (?)", (time.time(),))
            self._conn.commit()
            run_id = cursor.lastrowid
        if run_id is None:  # pragma: no cover - sqlite always sets lastrowid on INSERT
            raise IntegrityError("could not allocate a run id")
        return int(run_id)

    def finish_run(self, run_id: int, stats: dict[str, JsonValue]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET finished_at = ?, stats = ? WHERE run_id = ?",
                (time.time(), json.dumps(stats, sort_keys=True), run_id),
            )
            self._conn.commit()

    def last_run_id(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT MAX(run_id) FROM runs").fetchone()
        return int(row[0] or 0)

    # --- health -----------------------------------------------------------------------------

    def check_writable(self) -> bool:
        """Readiness probe: a write transaction can be opened and committed."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO meta (k, v) VALUES ('ready_probe', ?)", (str(time.time()),)
            )
            self._conn.commit()
        return True

    # --- test and tooling helpers -------------------------------------------------------------

    def raw_execute(self, sql: str, params: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        """Run raw SQL (tamper tests and the ``store`` CLI). Not used by the engine."""
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
            self._conn.commit()
        return [tuple(row) for row in rows]
