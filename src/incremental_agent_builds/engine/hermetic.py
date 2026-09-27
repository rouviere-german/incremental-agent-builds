"""Hermeticity guard: detect rule bodies that do I/O outside BuildContext.

Incremental correctness rests on one contract: every input a rule reads goes through its
BuildContext, so it is recorded in the trace. This module checks the contract at runtime with a
``sys.addaudithook`` hook (PEP 578). While a rule body runs, the session marks the task's context
with the rule's key; BuildContext methods grant a short-lived I/O permit around their own reads.
Any audited file, directory, process, socket or database event outside such a permit is a
violation: recorded in ``warn`` mode, raised as :class:`HermeticityError` in ``strict`` mode
(the test suite runs strict).

Limits (see docs/adr/0004): audit hooks see only audited operations. Reading the clock, the RNG or
already-open file handles is invisible to them; a semgrep rule in CI flags those in rule bodies.
Native extensions can bypass the hook. This is a guard for honest mistakes, not a sandbox.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal

from incremental_agent_builds.engine.errors import HermeticityError

__all__ = [
    "GuardMode",
    "Violation",
    "active_rule",
    "clear_rule_marker",
    "clear_violations",
    "install_guard",
    "io_permit",
    "rule_scope",
    "violations",
]

GuardMode = Literal["off", "warn", "strict"]

_active_rule: ContextVar[str | None] = ContextVar("iab_active_rule", default=None)
_mode: ContextVar[GuardMode] = ContextVar("iab_guard_mode", default="off")
_permit: ContextVar[bool] = ContextVar("iab_io_permit", default=False)

_WATCHED_EXACT = frozenset(
    {
        "open",
        "subprocess.Popen",
        "sqlite3.connect",
        "urllib.Request",
        "http.client.connect",
        "ftplib.connect",
        "smtplib.connect",
        "webbrowser.open",
        "ctypes.dlopen",
        "glob.glob",
    }
)
_WATCHED_PREFIXES = ("os.", "shutil.", "socket.")
_MAX_RECORDED = 1_000


@dataclass(frozen=True, slots=True)
class Violation:
    key_id: str
    event: str
    detail: str


_violations: list[Violation] = []
_installed: list[bool] = []


def _watched(event: str) -> bool:
    return event in _WATCHED_EXACT or event.startswith(_WATCHED_PREFIXES)


def _inside_import_machinery() -> bool:
    frame = sys._getframe(2)
    depth = 0
    while frame is not None and depth < 64:
        if frame.f_code.co_filename.startswith("<frozen importlib"):
            return True
        frame = frame.f_back  # type: ignore[assignment]
        depth += 1
    return False


def _hook(event: str, args: tuple[Any, ...]) -> None:
    key_id = _active_rule.get()
    if key_id is None or _permit.get() or not _watched(event):
        return
    mode = _mode.get()
    if mode == "off" or _inside_import_machinery():
        return
    detail = repr(args[0])[:200] if args else ""
    if len(_violations) < _MAX_RECORDED:
        _violations.append(Violation(key_id, event, detail))
    if mode == "strict":
        raise HermeticityError(
            f"rule {key_id} performed {event} outside BuildContext ({detail}); route the "
            "input through ctx.read/ctx.glob/ctx.config/ctx.tool/ctx.llm"
        )


def install_guard() -> None:
    """Install the audit hook once per process (audit hooks cannot be removed)."""
    if not _installed:
        sys.addaudithook(_hook)
        _installed.append(True)


def violations() -> list[Violation]:
    return list(_violations)


def clear_violations() -> None:
    _violations.clear()


def active_rule() -> str | None:
    return _active_rule.get()


@contextmanager
def rule_scope(key_id: str, mode: GuardMode) -> Iterator[None]:
    """Mark the current context as running the body of ``key_id``'s rule."""
    rule_token = _active_rule.set(key_id)
    mode_token = _mode.set(mode)
    permit_token = _permit.set(False)
    try:
        yield
    finally:
        _permit.reset(permit_token)
        _mode.reset(mode_token)
        _active_rule.reset(rule_token)


@contextmanager
def io_permit() -> Iterator[None]:
    """Allow audited I/O for engine code called from inside a rule (BuildContext methods)."""
    token = _permit.set(True)
    try:
        yield
    finally:
        _permit.reset(token)


def clear_rule_marker() -> None:
    """Clear the rule marker in the current context (run inside a copied context for new tasks)."""
    _active_rule.set(None)
    _permit.set(False)
