"""The workspace jail: which paths are accepted, and what glob means.

The in-memory and on-disk workspaces share one path validator and one glob matcher, so the property
tests' ``MemoryWorkspace`` behaves exactly like the ``FileWorkspace`` a real build reads. The
parametrized tests below assert that by running the same cases against both.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from incremental_agent_builds.engine.errors import WorkspaceError
from incremental_agent_builds.engine.workspace import (
    FileWorkspace,
    MemoryWorkspace,
    Workspace,
    glob_regex,
    validate_relative_path,
)

pytestmark = pytest.mark.unit

_FILES = {
    "a.txt": "a",
    "pkg/__init__.py": "",
    "pkg/core/engine.py": "core",
    "pkg/core/deep/nested.py": "deep",
    "pkg/io/reader.py": "io",
    "templates/one.txt": "t",
}


@pytest.fixture(params=["memory", "file"])
def workspace(request: pytest.FixtureRequest, tmp_path: Path) -> Workspace:
    if request.param == "memory":
        return MemoryWorkspace(dict(_FILES))
    root = tmp_path / "ws"
    for relative, text in _FILES.items():
        path = root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return FileWorkspace(root)


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "..",
        "../secrets.txt",
        "pkg/../../out.txt",
        "pkg/./x.py",
        "pkg\\core\\engine.py",
        "C:/Windows/system.ini",
        "pkg/\x00.py",
        "",
        "x" * 513,
        "pkg//double.py",
    ],
)
def test_unsafe_paths_are_refused(workspace: Workspace, path: str) -> None:
    with pytest.raises(WorkspaceError):
        workspace.read_bytes(path)


@pytest.mark.parametrize("path", ["a.txt", "pkg/core/engine.py", "pkg/core/deep/nested.py"])
def test_safe_paths_are_read(workspace: Workspace, path: str) -> None:
    assert workspace.read_bytes(path) == _FILES[path].encode("utf-8")


def test_a_missing_file_reads_as_none(workspace: Workspace) -> None:
    assert workspace.read_bytes("pkg/absent.py") is None


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("*.txt", ["a.txt"]),
        ("pkg/*/*.py", ["pkg/core/engine.py", "pkg/io/reader.py"]),
        (
            "pkg/**/*.py",
            [
                "pkg/__init__.py",
                "pkg/core/deep/nested.py",
                "pkg/core/engine.py",
                "pkg/io/reader.py",
            ],
        ),
        ("pkg/core/**", ["pkg/core/deep/nested.py", "pkg/core/engine.py"]),
        # '?' is exactly one character and '*' never crosses a '/'.
        ("pkg/?o*/*.py", ["pkg/core/engine.py", "pkg/io/reader.py"]),
        ("pkg/?or*/*.py", ["pkg/core/engine.py"]),
        ("pkg/*/*/*.py", ["pkg/core/deep/nested.py"]),
        ("nothing/*.py", []),
    ],
)
def test_glob_matches_the_same_paths_in_both_workspaces(
    workspace: Workspace, pattern: str, expected: list[str]
) -> None:
    assert workspace.glob(pattern) == expected


def test_glob_results_are_sorted(workspace: Workspace) -> None:
    matches = workspace.glob("pkg/**/*.py")
    assert matches == sorted(matches)


@pytest.mark.parametrize("pattern", ["/abs/*", "../*", "pkg/../*", "a\\*"])
def test_unsafe_glob_patterns_are_refused(workspace: Workspace, pattern: str) -> None:
    with pytest.raises(WorkspaceError):
        workspace.glob(pattern)


def test_validate_relative_path_returns_the_path_it_accepted() -> None:
    assert validate_relative_path("pkg/core/engine.py") == "pkg/core/engine.py"


def test_glob_regex_is_cached_and_anchored() -> None:
    first = glob_regex("pkg/*.py")
    assert glob_regex("pkg/*.py") is first
    assert first.fullmatch("pkg/a.py")
    assert not first.fullmatch("x/pkg/a.py")
    assert not first.fullmatch("pkg/a.py.bak")


def test_a_file_larger_than_the_cap_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "big.txt").write_bytes(b"x" * 100)
    workspace = FileWorkspace(root, max_file_bytes=64)
    with pytest.raises(WorkspaceError, match="exceeds"):
        workspace.read_bytes("big.txt")


def test_a_directory_is_not_a_readable_file(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    (root / "sub").mkdir(parents=True)
    with pytest.raises(WorkspaceError, match="regular file"):
        FileWorkspace(root).read_bytes("sub")


def test_a_missing_root_is_refused(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError, match="not a directory"):
        FileWorkspace(tmp_path / "absent")


@pytest.mark.skipif(
    sys.platform == "win32" and not os.environ.get("CI"),
    reason="symlinks need privileges on Windows",
)
def test_a_symlink_out_of_the_jail_is_refused_and_invisible(tmp_path: Path) -> None:
    """``realpath`` resolution is what makes the jail hold for links."""
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    root = tmp_path / "ws"
    root.mkdir()
    (root / "ok.txt").write_text("ok", encoding="utf-8")
    try:
        (root / "escape.txt").symlink_to(outside)
    except (OSError, NotImplementedError):  # pragma: no cover - unprivileged Windows
        pytest.skip("cannot create symlinks here")
    workspace = FileWorkspace(root)
    with pytest.raises(WorkspaceError, match="escapes"):
        workspace.read_bytes("escape.txt")
    assert workspace.glob("*.txt") == ["ok.txt"], "a link out of the jail is not listed"


def test_memory_workspace_write_and_delete_validate_paths() -> None:
    workspace = MemoryWorkspace()
    workspace.write("a/b.txt", "x")
    assert workspace.read_bytes("a/b.txt") == b"x"
    workspace.write("a/b.txt", b"y")
    assert workspace.read_bytes("a/b.txt") == b"y"
    workspace.delete("a/b.txt")
    assert workspace.read_bytes("a/b.txt") is None
    workspace.delete("a/b.txt")  # deleting twice is not an error
    with pytest.raises(WorkspaceError):
        workspace.write("../escape.txt", "x")


def test_resolve_exposes_the_real_path_inside_the_jail(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.txt").write_text("a", encoding="utf-8")
    workspace = FileWorkspace(root)
    assert workspace.resolve("a.txt") == Path(os.path.realpath(root / "a.txt"))
