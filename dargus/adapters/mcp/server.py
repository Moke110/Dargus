"""MCP server — exposes Dargus as a Model Context Protocol server (stdio).

All 14 tools are registered from ``dargus.adapters.mcp.tools``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from dargus._env import load_dotenv
from dargus.adapters.mcp.tools import TOOL_DISPATCH, TOOLS

app = Server("dargus")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [Tool(**t) for t in TOOLS]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name not in TOOL_DISPATCH:
        return [
            TextContent(
                type="text",
                text=json.dumps({"success": False, "error": f"Unknown tool {name}"}),
            )
        ]
    result = TOOL_DISPATCH[name](**arguments)
    return [TextContent(type="text", text=json.dumps(result))]


def list_tools_sync() -> list[Tool]:
    """Synchronous helper for tests."""
    return [Tool(**t) for t in TOOLS]


async def run_server() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main() -> None:
    load_dotenv()
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
