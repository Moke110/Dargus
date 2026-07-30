"""Basic file Tools — read_file / write_file with workspace enforcement (#56).

ADR-0001: these are the first general-purpose file Tools every Agent may be
granted. Both receive the WorkspaceGuard at construction and call it first
inside execute(). A rejected write returns a structured error so the Agent
can retry with a corrected path — hallucinated paths are mistakes, not attacks.
"""

from __future__ import annotations

import logging
from pathlib import Path

from dargus.runtime.workspace import _format_workspace_error, _WorkspaceError
from dargus.tools.base import Tool, ToolParam

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MAX_READ_BYTES = 1_000_000  # 1 MB limit to prevent memory exhaustion


def _build_read_file_tool() -> Tool:
    """Build the ``read_file`` Tool definition."""
    return Tool(
        name="read_file",
        description=(
            "Read the contents of a file at the given path "
            "within the workspace or Authorized Paths."
        ),
        parameters=[
            ToolParam(
                name="path",
                type="path",
                required=True,
                description="Path to the file to read (absolute or relative).",
            ),
        ],
        output={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "File content as a string."},
            },
        },
        timeout_ms=30_000,
        fallback="error",
        side_effect="read",
    )


def _build_write_file_tool() -> Tool:
    """Build the ``write_file`` Tool definition."""
    return Tool(
        name="write_file",
        description=(
            "Write content to a file at the given path within the workspace. "
            "Parent directories are created automatically inside the workspace."
        ),
        parameters=[
            ToolParam(
                name="path",
                type="path",
                required=True,
                description=(
                    "Path to write to " "(absolute or relative, must be inside the workspace)."
                ),
            ),
            ToolParam(
                name="content",
                type="string",
                required=True,
                description="String content to write.",
            ),
        ],
        output={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Canonical path the file was written to.",
                },
                "bytes_written": {"type": "integer", "description": "Number of bytes written."},
            },
        },
        timeout_ms=30_000,
        fallback="error",
        side_effect="write",
    )


# ---------------------------------------------------------------------------
# Implementation callables
# ---------------------------------------------------------------------------


def _impl_read_file(guard, path: str) -> dict:
    """Implementation for ``read_file``."""
    if guard is None:
        raise RuntimeError("read_file invoked without WorkspaceGuard")

    try:
        canonical = guard.check_read(path)
    except _WorkspaceError as exc:
        return _format_workspace_error(exc)

    p = Path(canonical)
    if not p.exists():
        return {"error": f"File not found: {path!r}", "path": str(p)}
    if not p.is_file():
        return {"error": f"Path is not a file: {path!r}", "path": str(p)}

    try:
        content = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Read as bytes and return a note; don't silently corrupt binary data.
        return {"error": f"File {path!r} is not valid UTF-8 text", "path": str(p)}

    if len(content) > _MAX_READ_BYTES:
        content = content[:_MAX_READ_BYTES]
        logger.warning("read_file truncated %r at %d bytes", path, _MAX_READ_BYTES)

    return {"content": content, "path": str(p)}


def _impl_write_file(guard, path: str, content: str) -> dict:
    """Implementation for ``write_file``."""
    if guard is None:
        raise RuntimeError("write_file invoked without WorkspaceGuard")

    try:
        canonical = guard.check_write(path)
    except _WorkspaceError as exc:
        return _format_workspace_error(exc)

    p = Path(canonical)
    p.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode("utf-8")
    p.write_bytes(encoded)

    return {"path": str(p), "bytes_written": len(encoded)}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_read_file_tool(guard) -> Tool:
    """Create a fully wired ``read_file`` Tool.

    Args:
        guard: The runtime's ``WorkspaceGuard`` instance.
    """
    tool = _build_read_file_tool()
    tool.inject_guard(guard)
    # Bind closure so guard is captured
    tool.bind(lambda **kwargs: _impl_read_file(guard, **kwargs))
    return tool


def make_write_file_tool(guard) -> Tool:
    """Create a fully wired ``write_file`` Tool.

    Args:
        guard: The runtime's ``WorkspaceGuard`` instance.
    """
    tool = _build_write_file_tool()
    tool.inject_guard(guard)
    tool.bind(lambda **kwargs: _impl_write_file(guard, **kwargs))
    return tool
