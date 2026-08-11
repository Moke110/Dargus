"""Standalone smoke: Tools — registry happy path + WorkspaceGuard boundary (offline).

Pins two tool invariants: (1) the runtime's ToolRegistry exposes the
general-purpose file Tools (read_file / write_file) wired to the workspace
guard, and (2) the WorkspaceGuard rejects a write that escapes the workspace
root while accepting one inside it — the file-access boundary is enforced.

Contract: prints a ``PASS`` / ``FAIL`` / ``SKIP`` verdict line and exits 0 on
pass/skip, non-zero on fail. Run directly:  python smoke_tools.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from _bootstrap import ensure_dargus_on_path

ensure_dargus_on_path()


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "workspace"
        root.mkdir()

        from dargus.runtime.context import DargusRuntime
        from dargus.runtime.workspace import WorkspaceGuard, _WorkspaceError

        # 1. A runtime wires a guard + registry; the file Tools are registered.
        rt = DargusRuntime(config={"workspace_root": str(root)})
        assert rt.workspace_guard is not None
        assert rt.workspace_root == str(root)
        names = {t.name for t in rt.tool_registry.list_all()}
        assert {"read_file", "write_file"} <= names, f"missing file tools in {names}"

        # 2. A standalone guard enforces the write boundary.
        guard = WorkspaceGuard(root=str(root))

        inside = root / "output.txt"
        assert guard.check_write(str(inside)) == str(inside.resolve())

        outside = Path(tmp) / "escaped.txt"
        try:
            guard.check_write(str(outside))
        except _WorkspaceError as exc:
            assert "outside workspace root" in str(exc)
        else:
            raise AssertionError("write outside workspace root was NOT rejected")

        # 3. The registered write_file tool rejects the escape through the real
        #    guard path (not just the standalone guard).
        write_tool = rt.tool_registry.get("write_file")
        result = write_tool.execute(path=str(outside), content="bad")
        assert isinstance(result, dict)
        assert "error" in result, f"write_file escape not rejected: {result}"
        assert not outside.exists(), "escaped file was written to disk"

        # 4. A write inside the root through the tool succeeds.
        ok = write_tool.execute(path=str(root / "ok.txt"), content="data")
        assert isinstance(ok, dict) and (root / "ok.txt").read_text() == "data"

    print("PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — smoke scripts report any failure as FAIL
        print(f"FAIL: {exc.__class__.__name__}: {exc}")
        sys.exit(1)
