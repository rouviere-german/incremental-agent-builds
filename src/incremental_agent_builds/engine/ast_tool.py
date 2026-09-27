"""The AST outline tool: parses untrusted Python source in a separate process.

``ast.parse`` on hostile input can exhaust the stack or memory, so parsing runs in a child
interpreter started in isolated mode (``python -I``) that reads one JSON request per line on stdin
and writes one JSON response per line on stdout. The parent enforces a per-request timeout and an
output cap; a worker that times out, overflows the cap or dies is killed and replaced on the next
request. The child receives source text, never a path, so it has no reason to touch the
filesystem. Workers are pooled and reused across requests and builds.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import queue
import subprocess
import sys
import threading
import weakref
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from incremental_agent_builds.engine.errors import ToolStepError
from incremental_agent_builds.engine.tools import BuildTool

__all__ = [
    "AstOutlineArgs",
    "AstWorker",
    "AstWorkerPool",
    "make_ast_outline_tool",
]


def _worker_source() -> str:
    return Path(__file__).with_name("_ast_worker.py").read_text(encoding="utf-8")


_EOF = object()
_OVERSIZE = object()


class _WorkerGoneError(Exception):
    """Internal: the child process is not there any more, so the request can be retried."""


def _kill(process: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(OSError):
        process.kill()
    with contextlib.suppress(OSError, subprocess.TimeoutExpired):
        process.wait(timeout=5)
    for stream in (process.stdin, process.stdout):
        if stream is not None:
            with contextlib.suppress(OSError):
                stream.close()


class AstWorker:
    """One child process plus a reader thread that enforces the output cap."""

    def __init__(
        self,
        *,
        timeout_seconds: float,
        max_output_bytes: int,
        command: Sequence[str] | None = None,
    ) -> None:
        self._timeout = timeout_seconds
        self._cap = max_output_bytes
        self._command = list(command or [sys.executable, "-I", "-c", _worker_source()])
        self._process: subprocess.Popen[bytes] | None = None
        self._lines: queue.Queue[object] = queue.Queue()
        self._ids = itertools.count(1)
        self.restarts = 0

    def _start(self) -> subprocess.Popen[bytes]:
        process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._lines = queue.Queue()
        lines = self._lines
        cap = self._cap

        def pump(stream: Any) -> None:
            while True:
                line = stream.readline(cap + 1)
                if not line:
                    lines.put(_EOF)
                    return
                if len(line) > cap and not line.endswith(b"\n"):
                    lines.put(_OVERSIZE)
                    return
                lines.put(line)

        threading.Thread(target=pump, args=(process.stdout,), daemon=True).start()
        weakref.finalize(self, _kill, process)
        self._process = process
        return process

    def close(self) -> None:
        if self._process is not None:
            _kill(self._process)
            self._process = None

    def request(self, source: str) -> dict[str, JsonValue]:
        """Outline ``source``, restarting a dead worker once.

        A pooled worker can be gone by the time the next request arrives: the OS killed it, or it
        exited between two requests. That is not a property of the input, and the tool is a pure
        function of its input, so re-running it on a fresh process is sound and cheap. A timeout, an
        oversized reply or a malformed reply are *not* retried: those say something about this
        request, and retrying would pay the cost twice for the same answer.
        """
        try:
            return self._attempt(source)
        except _WorkerGoneError:
            self.restarts += 1
        try:
            return self._attempt(source)
        except _WorkerGoneError as exc:
            raise ToolStepError("AST worker exited unexpectedly") from exc

    def _attempt(self, source: str) -> dict[str, JsonValue]:
        process = self._process
        if process is None or process.poll() is not None:
            if process is not None:
                self.close()
            process = self._start()
        request_id = next(self._ids)
        payload = json.dumps({"id": request_id, "source": source}).encode("ascii") + b"\n"
        try:
            if process.stdin is None:  # pragma: no cover - Popen(stdin=PIPE) always sets it
                raise _WorkerGoneError
            process.stdin.write(payload)
            process.stdin.flush()
        except OSError as exc:
            self.close()
            raise _WorkerGoneError from exc
        try:
            line = self._lines.get(timeout=self._timeout)
        except queue.Empty as exc:
            self.close()
            raise ToolStepError(f"AST worker timed out after {self._timeout}s") from exc
        if line is _OVERSIZE:
            self.close()
            raise ToolStepError(f"AST worker output exceeded {self._cap} bytes")
        if line is _EOF or not isinstance(line, bytes):
            self.close()
            raise _WorkerGoneError
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            self.close()
            raise ToolStepError("AST worker returned malformed output") from exc
        if not isinstance(response, dict) or response.get("id") != request_id:
            self.close()
            raise ToolStepError("AST worker response does not match the request")
        result = response.get("result")
        if not isinstance(result, dict):
            raise ToolStepError("AST worker returned a non-object result")
        return result


class AstWorkerPool:
    """A small pool of workers; each request uses one idle worker exclusively."""

    def __init__(
        self,
        *,
        size: int = 2,
        timeout_seconds: float = 10.0,
        max_output_bytes: int = 1_048_576,
        command: Sequence[str] | None = None,
    ) -> None:
        if size < 1:
            raise ValueError("pool size must be >= 1")
        self._idle: queue.Queue[AstWorker] = queue.Queue()
        self._workers = [
            AstWorker(
                timeout_seconds=timeout_seconds, max_output_bytes=max_output_bytes, command=command
            )
            for _ in range(size)
        ]
        for worker in self._workers:
            self._idle.put(worker)

    def outline(self, source: str) -> dict[str, JsonValue]:
        worker = self._idle.get()
        try:
            return worker.request(source)
        finally:
            self._idle.put(worker)

    def close(self) -> None:
        for worker in self._workers:
            worker.close()


class AstOutlineArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str = Field(min_length=1, max_length=512, pattern=r"\.py$")


def make_ast_outline_tool(pool: AstWorkerPool, *, version: int = 1) -> BuildTool[AstOutlineArgs]:
    """The ``ast_outline`` build tool: public API outline of one module, via the worker pool."""

    def handler(args: AstOutlineArgs, inputs: dict[str, bytes | None] | Any) -> JsonValue:
        data = inputs.get(args.path)
        if data is None:
            return {"error": "missing"}
        try:
            source = data.decode("utf-8")
        except UnicodeDecodeError:
            return {"error": "UnicodeDecodeError"}
        return pool.outline(source)

    return BuildTool(
        name="ast_outline",
        description="Outline of a Python module's top-level imports and public symbols.",
        args_model=AstOutlineArgs,
        handler=handler,
        inputs=lambda args: (args.path,),
        version=version,
    )
