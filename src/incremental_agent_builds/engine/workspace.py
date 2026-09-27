"""Workspaces: the only place rules read inputs from.

Paths are workspace-relative POSIX paths (``pkg/io/loader.py``). They are validated before use:
no absolute paths, drive letters, backslashes, NUL bytes, ``.`` or ``..`` segments. On disk,
:class:`FileWorkspace` resolves every path with ``os.path.realpath`` and refuses anything that lands
outside the resolved root (for example through a symlink), re-checks the opened file's identity to
narrow the check-then-open race, and caps file size. This is a jail for well-behaved rule code and
untrusted *content*; it is not a sandbox against a concurrently malicious local user.

Both implementations share one glob matcher, so an in-memory workspace used by property tests
behaves exactly like the on-disk one.
"""

from __future__ import annotations

import functools
import os
import re
import stat
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Protocol

from incremental_agent_builds.engine.errors import WorkspaceError

__all__ = [
    "FileWorkspace",
    "MemoryWorkspace",
    "Workspace",
    "glob_regex",
    "validate_relative_path",
]

_MAX_PATH = 512
_SEGMENT = re.compile(r"^[^\x00-\x1f<>:\"|?*\\]+$")
_PATTERN_SEGMENT = re.compile(r"^[^\x00-\x1f<>:\"|\\]+$")


def _check_segments(text: str, *, what: str, segment_re: re.Pattern[str]) -> None:
    if not text or len(text) > _MAX_PATH:
        raise WorkspaceError(f"{what} must be 1..{_MAX_PATH} characters")
    if text.startswith("/") or "\\" in text or "\x00" in text:
        raise WorkspaceError(f"{what} must be a relative POSIX path: {text!r}")
    for segment in text.split("/"):
        if segment in {"", ".", ".."}:
            raise WorkspaceError(f"{what} has an empty, '.' or '..' segment: {text!r}")
        if not segment_re.fullmatch(segment):
            raise WorkspaceError(f"{what} has an invalid segment: {text!r}")


def validate_relative_path(path: str) -> str:
    """Return ``path`` if it is a safe workspace-relative POSIX path, else raise."""
    _check_segments(path, what="path", segment_re=_SEGMENT)
    return path


@functools.lru_cache(maxsize=256)
def glob_regex(pattern: str) -> re.Pattern[str]:
    """Compile a glob (``*``, ``?``, ``**`` for any number of directories) to a regex."""
    _check_segments(pattern, what="glob pattern", segment_re=_PATTERN_SEGMENT)
    parts: list[str] = []
    segments = pattern.split("/")
    for index, segment in enumerate(segments):
        last = index == len(segments) - 1
        if segment == "**":
            parts.append(".*" if last else "(?:[^/]+/)*")
            continue
        body = ""
        for char in segment:
            if char == "*":
                body += "[^/]*"
            elif char == "?":
                body += "[^/]"
            else:
                body += re.escape(char)
        parts.append(body if last else body + "/")
    return re.compile("^" + "".join(parts) + "$")


class Workspace(Protocol):
    def read_bytes(self, path: str) -> bytes | None:
        """File content, or None when the file does not exist."""
        ...

    def glob(self, pattern: str) -> list[str]:
        """Sorted relative paths of the files matching ``pattern``."""
        ...


class MemoryWorkspace:
    """A workspace held in a dict (tests, property checks and benchmarks)."""

    def __init__(self, files: Mapping[str, bytes | str] | None = None) -> None:
        self.files: dict[str, bytes] = {}
        for path, content in (files or {}).items():
            self.write(path, content)

    def write(self, path: str, content: bytes | str) -> None:
        validate_relative_path(path)
        self.files[path] = content.encode("utf-8") if isinstance(content, str) else content

    def delete(self, path: str) -> None:
        self.files.pop(validate_relative_path(path), None)

    def read_bytes(self, path: str) -> bytes | None:
        return self.files.get(validate_relative_path(path))

    def glob(self, pattern: str) -> list[str]:
        regex = glob_regex(pattern)
        return sorted(path for path in self.files if regex.fullmatch(path))


class FileWorkspace:
    """An on-disk workspace jailed to ``root`` (see the module docstring)."""

    def __init__(self, root: Path, *, max_file_bytes: int = 1_048_576) -> None:
        if not root.is_dir():
            raise WorkspaceError(f"workspace root {root} is not a directory")
        self._root = Path(os.path.realpath(root))
        self._max = max_file_bytes

    @property
    def root(self) -> Path:
        return self._root

    def _inside(self, real: str) -> bool:
        root = str(self._root)
        try:
            return os.path.commonpath([root, real]) == root
        except ValueError:  # different drives on Windows
            return False

    def resolve(self, path: str) -> Path:
        """The real path of a workspace file; raises if it escapes the root."""
        validate_relative_path(path)
        real = os.path.realpath(self._root.joinpath(*path.split("/")))
        if not self._inside(real):
            raise WorkspaceError(f"path escapes the workspace: {path!r}")
        return Path(real)

    def read_bytes(self, path: str) -> bytes | None:
        real = self.resolve(path)
        try:
            before = os.stat(real)
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(before.st_mode):
            raise WorkspaceError(f"not a regular file: {path!r}")
        if before.st_size > self._max:
            raise WorkspaceError(f"file exceeds {self._max} bytes: {path!r}")
        with open(real, "rb") as handle:
            opened = os.fstat(handle.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise WorkspaceError(f"file changed identity while opening: {path!r}")
            data = handle.read(self._max + 1)
        if len(data) > self._max:
            raise WorkspaceError(f"file exceeds {self._max} bytes: {path!r}")
        return data

    def _walk(self) -> Iterable[str]:
        root = str(self._root)
        for directory, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames.sort()
            relative_dir = os.path.relpath(directory, root)
            for name in filenames:
                full = os.path.join(directory, name)
                if not self._inside(os.path.realpath(full)):
                    continue  # a symlink pointing outside the jail is invisible
                rel = name if relative_dir == "." else f"{relative_dir}/{name}"
                yield rel.replace(os.sep, "/")

    def glob(self, pattern: str) -> list[str]:
        regex = glob_regex(pattern)
        return sorted(path for path in self._walk() if regex.fullmatch(path))
