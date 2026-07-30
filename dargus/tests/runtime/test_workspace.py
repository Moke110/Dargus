"""Tests for WorkspaceGuard — ADR-0001 workspace safety.

Following 9_quality_and_experience.md: real tmp_path directories and symlinks,
no filesystem mocks.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dargus.runtime.workspace import WorkspaceGuard, _format_workspace_error, _WorkspaceError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ws_root(tmp_path: Path) -> Path:
    """A canonical workspace root."""
    return tmp_path / "workspace"


@pytest.fixture
def guard(ws_root: Path) -> WorkspaceGuard:
    ws_root.mkdir(parents=True, exist_ok=True)
    return WorkspaceGuard(root=str(ws_root))


@pytest.fixture
def authorized_dir(tmp_path: Path) -> Path:
    """An authorized directory outside the workspace root."""
    d = tmp_path / "authorized_input"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Construction + session binding
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_root_is_canonical_cwd(self):
        g = WorkspaceGuard()
        assert g.root == os.path.realpath(os.getcwd())

    def test_explicit_root_is_canonicalized(self, tmp_path: Path):
        # Use a relative path that symlinks don't affect
        root = tmp_path / "root"
        root.mkdir()
        g = WorkspaceGuard(root=str(root))
        assert g.root == os.path.realpath(str(root))

    def test_root_defaults_to_canonical_cwd(self):
        g = WorkspaceGuard()
        assert g.root == os.path.realpath(".")

    def test_authorized_paths_frozen_view(self, guard: WorkspaceGuard, tmp_path: Path):
        guard.authorize(str(tmp_path / "ext"))
        paths = guard.authorized_paths
        assert isinstance(paths, frozenset)

    def test_authorized_paths_canonicalized_on_entry(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        g = WorkspaceGuard(root=str(tmp_path / "ws"), authorized_paths=[str(sub)])
        assert os.path.realpath(str(sub)) in g.authorized_paths


# ---------------------------------------------------------------------------
# Write checks
# ---------------------------------------------------------------------------


class TestWriteChecks:
    def test_write_inside_root_passes(self, guard: WorkspaceGuard, ws_root: Path):
        target = ws_root / "output.txt"
        canonical = guard.check_write(str(target))
        assert canonical == os.path.realpath(str(target))

    def test_write_to_root_itself_passes(self, guard: WorkspaceGuard, ws_root: Path):
        canonical = guard.check_write(str(ws_root))
        assert canonical == guard.root

    def test_write_outside_root_rejected(self, guard: WorkspaceGuard, tmp_path: Path):
        target = tmp_path / "outside.txt"
        with pytest.raises(_WorkspaceError, match="outside workspace root"):
            guard.check_write(str(target))

    def test_write_dotdot_traversal_rejected(self, guard: WorkspaceGuard, ws_root: Path):
        target = ws_root / ".." / "escaped.txt"
        with pytest.raises(_WorkspaceError, match="outside workspace root"):
            guard.check_write(str(target))

    def test_write_symlink_escape_rejected(
        self, guard: WorkspaceGuard, ws_root: Path, tmp_path: Path
    ):
        """Symlink inside root pointing outside -> write is rejected."""
        outside = tmp_path / "outside_file.txt"
        outside.write_text("bad")
        link = ws_root / "link_to_outside"
        os.symlink(str(outside), str(link))
        # Writing through the symlink should resolve to outside and be rejected.
        with pytest.raises(_WorkspaceError, match="outside workspace root"):
            guard.check_write(str(link))


# ---------------------------------------------------------------------------
# Read checks
# ---------------------------------------------------------------------------


class TestReadChecks:
    def test_read_inside_root_passes(self, guard: WorkspaceGuard, ws_root: Path):
        f = ws_root / "data.txt"
        f.write_text("hello")
        canonical = guard.check_read(str(f))
        assert canonical == os.path.realpath(str(f))

    def test_read_outside_root_rejected(self, guard: WorkspaceGuard, tmp_path: Path):
        f = tmp_path / "secret.txt"
        f.write_text("secret")
        with pytest.raises(_WorkspaceError, match="not in any Authorized Path"):
            guard.check_read(str(f))

    def test_read_authorized_file_exact_match(
        self, guard: WorkspaceGuard, authorized_dir: Path
    ):
        f = authorized_dir / "input.pdf"
        f.write_text("data")
        guard.authorize(str(f))
        canonical = guard.check_read(str(f))
        assert canonical == os.path.realpath(str(f))

    def test_read_authorized_directory_prefix_match(
        self, guard: WorkspaceGuard, authorized_dir: Path
    ):
        f = authorized_dir / "sub" / "data.csv"
        f.parent.mkdir(exist_ok=True)
        f.write_text("csv")
        guard.authorize(str(authorized_dir))
        canonical = guard.check_read(str(f))
        assert canonical == os.path.realpath(str(f))

    def test_file_authorization_does_not_leak_to_siblings(
        self, guard: WorkspaceGuard, authorized_dir: Path
    ):
        good = authorized_dir / "good.txt"
        bad = authorized_dir / "bad.txt"
        good.write_text("ok")
        bad.write_text("no")
        guard.authorize(str(good))
        # good.txt is authorized
        guard.check_read(str(good))  # ok
        # bad.txt in same dir is NOT authorized
        with pytest.raises(_WorkspaceError, match="not in any Authorized Path"):
            guard.check_read(str(bad))

    def test_read_symlink_under_authorized_dir_allowed(
        self, guard: WorkspaceGuard, authorized_dir: Path, tmp_path: Path
    ):
        """Symlink inside an authorized dir pointing to an outside file is
        resolved: the resolved path is OUTSIDE the authorized dir, so it is
        rejected. Authorized Paths are NOT transitive through symlinks."""
        outside = tmp_path / "outside_data.txt"
        outside.write_text("data")
        link = authorized_dir / "link_to_outside"
        os.symlink(str(outside), str(link))
        guard.authorize(str(authorized_dir))
        # The resolved target is outside the authorized dir.
        with pytest.raises(_WorkspaceError, match="not in any Authorized Path"):
            guard.check_read(str(link))

    def test_read_symlink_into_authorized_file_allowed(
        self, guard: WorkspaceGuard, authorized_dir: Path, ws_root: Path
    ):
        """Symlink from outside pointing to an authorized file.
        The resolved path IS the authorized file, so read is allowed."""
        good = authorized_dir / "good.txt"
        good.write_text("ok")
        guard.authorize(str(good))
        # Create a symlink from workspace root pointing to the authorized file
        link = ws_root / "link_to_authorized"
        os.symlink(str(good), str(link))
        canonical = guard.check_read(str(link))
        assert canonical == os.path.realpath(str(good))


# ---------------------------------------------------------------------------
# Root canonicalization when cwd is a symlink
# ---------------------------------------------------------------------------


class TestRootSymlink:
    def test_root_canonicalizes_symlinked_cwd(self, tmp_path: Path):
        real_dir = tmp_path / "real_ws"
        real_dir.mkdir()
        link_dir = tmp_path / "link_ws"
        os.symlink(str(real_dir), str(link_dir))
        g = WorkspaceGuard(root=str(link_dir))
        assert g.root == os.path.realpath(str(real_dir))


# ---------------------------------------------------------------------------
# Structured error format
# ---------------------------------------------------------------------------


class TestStructuredError:
    def test_format_produces_error_dict(self, tmp_path: Path):
        exc = _WorkspaceError("bad path", "/tmp/bad")
        result = _format_workspace_error(exc)
        assert result["error"] == "bad path"
        assert result["path"] == "/tmp/bad"

    def test_format_keys(self, tmp_path: Path):
        """The structured error is the documented return shape for file Tools."""
        exc = _WorkspaceError("Access denied", "data/../outside.txt")
        result = _format_workspace_error(exc)
        assert set(result.keys()) == {"error", "path"}
        assert isinstance(result["error"], str)
        assert isinstance(result["path"], str)
