"""
MCP Server — the same tools the web app uses, exposed as MCP tools so Claude Desktop
(or any MCP client) can ask questions about your listening history natively.

Install:
  uv add mcp

Wire it into Claude Desktop by adding to claude_desktop_config.json:

  {
    "mcpServers": {
      "listening-archive": {
        "command": "uv",
        "args": ["run", "python", "-m", "backend.mcp_server"]
      }
    }
  }

Then ask Claude things like "search my listening memory for late-night sessions in March."
"""
from __future__ import annotations

import asyncio
import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from backend.config import settings
from backend.database import Database
from backend.rag.embeddings import Embedder
from backend.rag.lyrics_rag import LyricsRAG
from backend.rag.memory_rag import MemoryRAG
from backend.rag.artist_rag import ArtistRAG
from backend.rag.notes_rag import NotesRAG


server = Server("listening-archive")

# Initialise stateless — each tool call reuses the same persistent stores
db = Database(settings.DB_PATH)
embedder = Embedder()
lyrics_rag = LyricsRAG(embedder, settings.CHROMA_PATH)
memory_rag = MemoryRAG(embedder, settings.CHROMA_PATH, db)
artist_rag = ArtistRAG(embedder, settings.CHROMA_PATH)
notes_rag = NotesRAG(embedder, settings.CHROMA_PATH, db)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_lyrics",
            description="Semantic search over indexed song lyrics in the user's library.",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "k": {"type": "integer", "default": 4}},
                "required": ["query"],
            },
        ),
        Tool(
            name="search_memory",
            description="Search the user's listening history by semantic context (e.g. 'late nights in March').",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "k": {"type": "integer", "default": 5}},
                "required": ["query"],
            },
        ),
        Tool(
            name="get_artist_context",
            description="Retrieve indexed Wikipedia context for a named artist.",
            inputSchema={
                "type": "object",
                "properties": {"artist": {"type": "string"}},
                "required": ["artist"],
            },
        ),
        Tool(
            name="get_recent_plays",
            description="Return the user's most recent N plays from local SQLite cache.",
            inputSchema={
                "type": "object",
                "properties": {"limit": {"type": "integer", "default": 20}},
            },
        ),
        Tool(
            name="search_notes",
            description="Search the user's own annotations about tracks.",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "search_lyrics":
            result = lyrics_rag.search(arguments["query"], k=arguments.get("k", 4))
        elif name == "search_memory":
            result = memory_rag.search(arguments["query"], k=arguments.get("k", 5))
        elif name == "get_artist_context":
            result = artist_rag.get(arguments["artist"])
        elif name == "get_recent_plays":
            result = db.recent_plays(limit=arguments.get("limit", 20))
        elif name == "search_notes":
            result = notes_rag.search(arguments["query"])
        else:
            result = {"error": f"unknown tool: {name}"}
    except Exception as e:
        result = {"error": str(e)}
    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
