"""The AST outline tool: what it extracts, and how the parent contains a bad child process.

``ast.parse`` on hostile input can exhaust the stack or memory, so parsing runs in a separate
``python -I`` interpreter that only ever receives source text. The parent enforces a timeout and an
output cap and replaces a worker that dies. The tests below drive the pool with substitute child
commands so each containment path is exercised without needing hostile input.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest
from pydantic import JsonValue

from incremental_agent_builds.engine._ast_worker import outline
from incremental_agent_builds.engine.ast_tool import (
    AstOutlineArgs,
    AstWorker,
    AstWorkerPool,
    make_ast_outline_tool,
)
from incremental_agent_builds.engine.errors import ToolStepError
from tests.jsonshape import at, text

pytestmark = pytest.mark.unit

_SOURCE = '''\
"""Module docstring. Second sentence."""

from __future__ import annotations

from pkg.core import engine
from pkg import io

import os


# a comment the outline must ignore
def public(path: str, limit: int = 3) -> bool:
    """Do the thing."""
    return True


async def fetch(url: str) -> bytes:
    """Fetch it."""
    return b""


def _private() -> None:
    """Hidden."""


class Store(dict):
    """A store."""

    def put(self, key: str) -> None:
        """Put it."""

    def _internal(self) -> None:
        """Hidden."""
'''


# --- the pure outline function ------------------------------------------------------------------


def test_the_outline_lists_public_symbols_with_signatures_and_first_doc_lines() -> None:
    result = outline(_SOURCE)
    assert result["doc"].startswith("Module docstring.")
    names = [symbol["name"] for symbol in result["symbols"]]
    assert names == ["public", "fetch", "Store"], "private names are excluded"
    public = result["symbols"][0]
    assert public["signature"] == "def public(path: str, limit: int=3) -> bool"
    assert public["doc"] == "Do the thing."
    assert result["symbols"][1]["signature"].startswith("async def fetch")
    store = result["symbols"][2]
    assert store["signature"] == "class Store(dict)"
    assert [method["name"] for method in store["methods"]] == ["put"]


def test_the_outline_records_imports_with_their_level_and_names() -> None:
    result = outline(_SOURCE)
    assert {"module": "pkg.core", "level": 0, "names": ["engine"]} in result["imports"]
    assert {"module": "pkg", "level": 0, "names": ["io"]} in result["imports"]
    assert {"module": "os", "level": 0, "names": []} in result["imports"]


def test_relative_imports_keep_their_level() -> None:
    result = outline("from . import sibling\nfrom ..up import other\n")
    assert result["imports"] == [
        {"module": "", "level": 1, "names": ["sibling"]},
        {"module": "up", "level": 2, "names": ["other"]},
    ]


def test_comments_and_layout_do_not_change_the_outline() -> None:
    """This is what makes a whitespace-only or comment-only commit cut off immediately."""
    reformatted = _SOURCE.replace("\n\n", "\n\n\n").replace("# a comment", "# another comment")
    assert outline(reformatted) == outline(_SOURCE)


def test_a_syntax_error_becomes_a_named_error_not_an_exception() -> None:
    assert outline("def broken(:\n") == {"error": "SyntaxError"}
    assert outline("x = '\x00'") in ({"error": "SyntaxError"}, {"error": "ValueError"})


def test_an_empty_module_has_an_empty_outline() -> None:
    assert outline("") == {"doc": "", "imports": [], "symbols": []}


# --- the worker pool ----------------------------------------------------------------------------


def test_the_pool_outlines_source_in_a_child_process() -> None:
    pool = AstWorkerPool(size=1)
    try:
        assert text(pool.outline(_SOURCE)["doc"]).startswith("Module docstring.")
        outlined: JsonValue = dict(pool.outline("def f() -> None:\n    pass\n"))
        assert at(outlined, "symbols", 0, "name") == "f"
    finally:
        pool.close()


def test_the_pool_reuses_workers_across_requests() -> None:
    pool = AstWorkerPool(size=2)
    try:
        for _ in range(6):
            assert pool.outline("x = 1\n")["symbols"] == []
    finally:
        pool.close()


def test_a_pool_needs_at_least_one_worker() -> None:
    with pytest.raises(ValueError, match="pool size"):
        AstWorkerPool(size=0)


def _worker(script: str, **kwargs: Any) -> AstWorker:
    return AstWorker(command=[sys.executable, "-I", "-c", script], **kwargs)


def test_a_worker_that_never_answers_is_killed_and_reported() -> None:
    worker = _worker("import time\ntime.sleep(30)\n", timeout_seconds=0.5, max_output_bytes=1024)
    try:
        with pytest.raises(ToolStepError, match="timed out"):
            worker.request("x = 1")
    finally:
        worker.close()


def test_a_timeout_is_not_retried() -> None:
    """Retrying a timeout would pay the wall-clock cost twice for the same verdict."""
    worker = _worker("import time\ntime.sleep(30)\n", timeout_seconds=0.4, max_output_bytes=1024)
    try:
        with pytest.raises(ToolStepError, match="timed out"):
            worker.request("x = 1")
        assert worker.restarts == 0
    finally:
        worker.close()


def test_a_worker_that_floods_stdout_is_cut_off_at_the_cap() -> None:
    script = (
        "import sys\nsys.stdin.readline()\nsys.stdout.write('x' * 100000)\nsys.stdout.flush()\n"
    )
    worker = _worker(script, timeout_seconds=5.0, max_output_bytes=256)
    try:
        with pytest.raises(ToolStepError, match="exceeded"):
            worker.request("x = 1")
    finally:
        worker.close()


def test_a_worker_that_exits_immediately_is_reported_after_one_retry() -> None:
    worker = _worker("raise SystemExit(1)\n", timeout_seconds=5.0, max_output_bytes=1024)
    try:
        with pytest.raises(ToolStepError, match="exited unexpectedly"):
            worker.request("x = 1")
        assert worker.restarts == 1, "the retry happens once, not in a loop"
    finally:
        worker.close()


def test_malformed_worker_output_is_refused() -> None:
    script = "import sys\nsys.stdin.readline()\nprint('not json')\n"
    worker = _worker(script, timeout_seconds=5.0, max_output_bytes=1024)
    try:
        with pytest.raises(ToolStepError, match="malformed"):
            worker.request("x = 1")
    finally:
        worker.close()


def test_a_reply_for_another_request_id_is_refused() -> None:
    """Matching ids is what keeps a reused worker from pairing answers with the wrong question."""
    script = 'import sys\nsys.stdin.readline()\nprint(\'{"id": 999, "result": {}}\')\n'
    worker = _worker(script, timeout_seconds=5.0, max_output_bytes=1024)
    try:
        with pytest.raises(ToolStepError, match="does not match"):
            worker.request("x = 1")
    finally:
        worker.close()


def test_a_non_object_result_is_refused() -> None:
    script = 'import sys\nsys.stdin.readline()\nprint(\'{"id": 1, "result": []}\')\n'
    worker = _worker(script, timeout_seconds=5.0, max_output_bytes=1024)
    try:
        with pytest.raises(ToolStepError, match="non-object"):
            worker.request("x = 1")
    finally:
        worker.close()


def test_a_worker_that_dies_after_answering_is_replaced_transparently() -> None:
    """A worker killed between requests must not fail the build: the tool is a pure function."""
    script = (
        "import json, sys\n"
        "line = sys.stdin.readline()\n"
        "print(json.dumps({'id': json.loads(line)['id'], 'result': {'ok': True}}))\n"
        "sys.stdout.flush()\n"
    )
    worker = _worker(script, timeout_seconds=5.0, max_output_bytes=1024)
    try:
        assert worker.request("x = 1") == {"ok": True}
        assert worker.request("x = 2") == {"ok": True}, "the second request restarts the worker"
        assert worker.restarts >= 1
    finally:
        worker.close()


def test_closing_a_worker_twice_is_harmless() -> None:
    worker = AstWorker(timeout_seconds=5.0, max_output_bytes=1024)
    worker.close()
    worker.close()


# --- the build tool wrapper ---------------------------------------------------------------------


def test_the_tool_declares_its_input_and_returns_the_outline() -> None:
    pool = AstWorkerPool(size=1)
    try:
        tool = make_ast_outline_tool(pool)
        assert tool.name == "ast_outline"
        args, canonical = tool.validate({"path": "pkg/a.py"})
        assert tool.inputs(args) == ("pkg/a.py",)
        assert canonical == {"path": "pkg/a.py"}
        result = tool.handler(args, {"pkg/a.py": b"def f() -> None:\n    pass\n"})
        assert result["symbols"][0]["name"] == "f"  # type: ignore[index,call-overload]
    finally:
        pool.close()


def test_the_tool_reports_a_missing_or_undecodable_file_as_data() -> None:
    """A missing file is a value, not a failure: the trace records that it was absent."""
    pool = AstWorkerPool(size=1)
    try:
        tool = make_ast_outline_tool(pool)
        args, _ = tool.validate({"path": "pkg/a.py"})
        assert tool.handler(args, {"pkg/a.py": None}) == {"error": "missing"}
        assert tool.handler(args, {"pkg/a.py": b"\xff\xfe"}) == {"error": "UnicodeDecodeError"}
    finally:
        pool.close()


@pytest.mark.parametrize("path", ["", "a.txt", "a.py.bak", "x" * 513])
def test_the_tool_only_accepts_python_paths(path: str) -> None:
    with pytest.raises(ValueError, match="validation error"):
        AstOutlineArgs(path=path)
