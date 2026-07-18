from __future__ import annotations

import asyncio
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from dargus.mcp_tools import (
    tool_ingest_data,
    tool_predict,
    tool_query_dbase,
    tool_search_literature,
    tool_start_project,
    tool_status,
)

TOOLS: list[Tool] = [
    Tool(
        name="dargus_start_project",
        description="Create a new Dargus project and initialize its D-Base.",
        inputSchema={
            "type": "object",
            "required": ["disease"],
            "properties": {
                "disease": {"type": "string"},
                "target": {"type": "string"},
                "endpoints": {"type": "array", "items": {"type": "string"}},
                "data_paths": {"type": "array", "items": {"type": "string"}},
                "projects_root": {"type": "string", "default": "projects"},
            },
        },
    ),
    Tool(
        name="dargus_ingest_data",
        description="Scan a local data directory and write records into D-Base.",
        inputSchema={
            "type": "object",
            "required": ["project_id", "datadir"],
            "properties": {
                "project_id": {"type": "string"},
                "datadir": {"type": "string"},
                "projects_root": {"type": "string", "default": "projects"},
            },
        },
    ),
    Tool(
        name="dargus_search_literature",
        description="Search literature for drug-disease evidence.",
        inputSchema={
            "type": "object",
            "required": ["project_id", "drug_ids", "disease_id"],
            "properties": {
                "project_id": {"type": "string"},
                "drug_ids": {"type": "array", "items": {"type": "string"}},
                "disease_id": {"type": "string"},
                "projects_root": {"type": "string", "default": "projects"},
            },
        },
    ),
    Tool(
        name="dargus_predict",
        description="Run Iris prediction for drugs against a disease.",
        inputSchema={
            "type": "object",
            "required": ["project_id", "drug_ids", "disease_id"],
            "properties": {
                "project_id": {"type": "string"},
                "drug_ids": {"type": "array", "items": {"type": "string"}},
                "disease_id": {"type": "string"},
                "endpoints": {"type": "array", "items": {"type": "string"}},
                "projects_root": {"type": "string", "default": "projects"},
            },
        },
    ),
    Tool(
        name="dargus_query_dbase",
        description="Query records from the D-Base.",
        inputSchema={
            "type": "object",
            "required": ["project_id"],
            "properties": {
                "project_id": {"type": "string"},
                "drug_id": {"type": "string"},
                "disease_id": {"type": "string"},
                "template_id": {"type": "string"},
                "projects_root": {"type": "string", "default": "projects"},
            },
        },
    ),
    Tool(
        name="dargus_status",
        description="Get status of a Dargus project.",
        inputSchema={
            "type": "object",
            "required": ["project_id"],
            "properties": {
                "project_id": {"type": "string"},
                "projects_root": {"type": "string", "default": "projects"},
            },
        },
    ),
]

TOOL_DISPATCH: dict[str, Any] = {
    "dargus_start_project": tool_start_project,
    "dargus_ingest_data": tool_ingest_data,
    "dargus_search_literature": tool_search_literature,
    "dargus_predict": tool_predict,
    "dargus_query_dbase": tool_query_dbase,
    "dargus_status": tool_status,
}

app = Server("dargus")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    import json

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
    return TOOLS


async def run_server() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main() -> None:
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
