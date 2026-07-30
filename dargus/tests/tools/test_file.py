"""Tests for basic file Tools — read_file / write_file (#56).

Real tmp_path; no filesystem mocks (9_quality_and_experience.md).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dargus.runtime.workspace import WorkspaceGuard
from dargus.tools.file import (
    _MAX_READ_BYTES,
    _impl_read_file,
    _impl_write_file,
    make_read_file_tool,
    make_write_file_tool,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ws_root(tmp_path: Path) -> Path:
    d = tmp_path / "workspace"
    d.mkdir()
    return d


@pytest.fixture
def guard(ws_root: Path) -> WorkspaceGuard:
    return WorkspaceGuard(root=str(ws_root))


@pytest.fixture
def read_tool(guard: WorkspaceGuard):
    return make_read_file_tool(guard)


@pytest.fixture
def write_tool(guard: WorkspaceGuard):
    return make_write_file_tool(guard)


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_read_existing_file_in_workspace(self, read_tool, ws_root: Path):
        f = ws_root / "data.txt"
        f.write_text("hello world")
        result = read_tool.execute(path=str(f))
        assert result["content"] == "hello world"
        assert os.path.realpath(result["path"]) == os.path.realpath(str(f))

    def test_read_file_outside_workspace(self, read_tool, tmp_path: Path):
        f = tmp_path / "outside.txt"
        f.write_text("secret")
        result = read_tool.execute(path=str(f))
        assert "error" in result
        assert "path" in result
        assert "secret" not in str(result)

    def test_read_nonexistent_file(self, read_tool, ws_root: Path):
        result = read_tool.execute(path=str(ws_root / "nope.txt"))
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_read_binary_file_returns_error(self, read_tool, ws_root: Path):
        f = ws_root / "binary.bin"
        f.write_bytes(b"\x00\x01\x02\xff")
        result = read_tool.execute(path=str(f))
        assert "error" in result
        assert "utf-8" in result["error"].lower()

    def test_read_truncates_over_limit(self, guard: WorkspaceGuard, ws_root: Path):
        f = ws_root / "large.txt"
        big_content = "x" * (_MAX_READ_BYTES + 500)
        f.write_text(big_content, encoding="utf-8")
        result = _impl_read_file(guard, path=str(f))
        assert len(result["content"]) == _MAX_READ_BYTES

    def test_read_authorized_file_outside_root(self, guard: WorkspaceGuard, tmp_path: Path):
        f = tmp_path / "authorized.txt"
        f.write_text("approved")
        guard.authorize(str(f))
        result = _impl_read_file(guard, path=str(f))
        assert result["content"] == "approved"

    def test_read_without_guard_raises(self):
        with pytest.raises(RuntimeError, match="without WorkspaceGuard"):
            _impl_read_file(None, path="/tmp/x")


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


class TestWriteFile:
    def test_write_inside_workspace_succeeds(self, write_tool, ws_root: Path):
        result = write_tool.execute(path=str(ws_root / "output.txt"), content="test data")
        assert "error" not in result
        assert result["bytes_written"] == 9
        assert os.path.realpath(result["path"]) == os.path.realpath(str(ws_root / "output.txt"))
        # Verify on disk
        assert (ws_root / "output.txt").read_text() == "test data"

    def test_write_creates_parent_dirs(self, write_tool, ws_root: Path):
        nested = ws_root / "deep" / "nested" / "file.txt"
        result = write_tool.execute(path=str(nested), content="deep content")
        assert "error" not in result
        assert nested.exists()
        assert nested.read_text() == "deep content"

    def test_write_outside_workspace_rejected(self, write_tool, tmp_path: Path):
        outside = tmp_path / "unauthorized.txt"
        result = write_tool.execute(path=str(outside), content="bad")
        assert "error" in result
        assert "path" in result
        # Nothing written
        assert not outside.exists()

    def test_write_dotdot_traversal_rejected(self, write_tool, ws_root: Path, tmp_path: Path):
        # Try to escape via ..
        escaped = ws_root / ".." / "escaped.txt"
        result = write_tool.execute(path=str(escaped), content="bad")
        assert "error" in result
        assert not (tmp_path / "escaped.txt").exists()

    def test_write_symlink_escape_rejected(self, write_tool, ws_root: Path, tmp_path: Path):
        outside = tmp_path / "outside_via_link.txt"
        link = ws_root / "escape_link"
        os.symlink(str(outside), str(link))
        result = write_tool.execute(path=str(link), content="bad")
        assert "error" in result
        assert not outside.exists()

    def test_write_without_guard_raises(self):
        with pytest.raises(RuntimeError, match="without WorkspaceGuard"):
            _impl_write_file(None, path="/tmp/x", content="test")


# ---------------------------------------------------------------------------
# Tool metadata (integration check)
# ---------------------------------------------------------------------------


class TestToolMetadata:
    def test_read_file_has_path_param(self, read_tool):
        assert any(p.name == "path" and p.type == "path" for p in read_tool.parameters)

    def test_write_file_has_path_param(self, write_tool):
        assert any(p.name == "path" and p.type == "path" for p in write_tool.parameters)

    def test_read_file_side_effect_is_read(self, read_tool):
        assert read_tool.side_effect == "read"

    def test_write_file_side_effect_is_write(self, write_tool):
        assert write_tool.side_effect == "write"

    def test_both_tools_have_guard(self, read_tool, write_tool, guard):
        assert read_tool._guard is guard
        assert write_tool._guard is guard
