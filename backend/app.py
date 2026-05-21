"""
The Listening Archive — FastAPI backend.
Serves the magazine frontend and orchestrates Spotify, Gemma, and the 5 RAG layers.
"""
from __future__ import annotations

import argparse
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import settings
from backend.database import Database
from backend.llm_client import LLMClient
from backend.spotify_client import SpotifyClient
from backend.rag.embeddings import Embedder
from backend.rag.lyrics_rag import LyricsRAG
from backend.rag.artist_rag import ArtistRAG
from backend.rag.memory_rag import MemoryRAG
from backend.rag.notes_rag import NotesRAG
from backend.rag.genre_rag import GenreRAG

ROOT = Path(__file__).parent.parent
FRONTEND_DIR = ROOT / "frontend"
DATA_DIR = ROOT / "data"


# ---------- Application state ----------

class State:
    db: Database
    llm: LLMClient
    spotify: SpotifyClient
    embedder: Embedder
    lyrics_rag: LyricsRAG
    artist_rag: ArtistRAG
    memory_rag: MemoryRAG
    notes_rag: NotesRAG
    genre_rag: GenreRAG
    demo_mode: bool = False


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("→ Initialising archive…")
    state.db = Database(settings.DB_PATH)
    state.db.init_schema()
    state.embedder = Embedder()
    state.llm = LLMClient()
    state.spotify = SpotifyClient(state.db)
    state.lyrics_rag = LyricsRAG(state.embedder, settings.CHROMA_PATH)
    state.artist_rag = ArtistRAG(state.embedder, settings.CHROMA_PATH)
    state.memory_rag = MemoryRAG(state.embedder, settings.CHROMA_PATH, state.db)
    state.notes_rag = NotesRAG(state.embedder, settings.CHROMA_PATH, state.db)
    state.genre_rag = GenreRAG(state.embedder, settings.CHROMA_PATH)

    if state.demo_mode:
        await load_demo_data()
    print("✓ Archive ready.")
    yield
    print("→ Closing archive.")


app = FastAPI(title="The Listening Archive", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------- Frontend ----------

@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")


# ---------- Spotify OAuth ----------

@app.get("/api/auth/login")
async def auth_login():
    return RedirectResponse(state.spotify.build_authorize_url())


@app.get("/api/callback")
async def auth_callback(code: str, state_param: str = Query(None, alias="state")):
    try:
        token = await state.spotify.exchange_code(code)
        return RedirectResponse("/?auth=ok")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# ---------- Wrapped data ----------

@app.get("/api/wrapped")
async def get_wrapped():
    """Return the full magazine-page data."""
    if state.demo_mode:
        with open(DATA_DIR / "demo_data.json") as f:
            return json.load(f)
    top_artists = await state.spotify.get_top_artists(time_range="long_term", limit=10)
    top_tracks = await state.spotify.get_top_tracks(time_range="long_term", limit=10)
    recent = await state.spotify.get_recently_played(limit=50)
    return {
        "top_artists": top_artists,
        "top_tracks": top_tracks,
        "recent_plays": recent,
    }


# ---------- Lyrics RAG ----------

class LyricsQuery(BaseModel):
    query: str
    k: int = 4


@app.post("/api/lyrics/search")
async def search_lyrics(body: LyricsQuery):
    results = state.lyrics_rag.search(body.query, k=body.k)
    return {"results": results}


# ---------- Memory RAG ----------

@app.post("/api/memory/search")
async def search_memory(body: LyricsQuery):
    results = state.memory_rag.search(body.query, k=body.k)
    return {"results": results}


# ---------- Notes RAG ----------

class Note(BaseModel):
    track_id: str
    text: str


@app.post("/api/notes")
async def add_note(note: Note):
    state.notes_rag.add(note.track_id, note.text)
    return {"ok": True}


@app.post("/api/notes/search")
async def search_notes(body: LyricsQuery):
    results = state.notes_rag.search(body.query, k=body.k)
    return {"results": results}


# ---------- Chat (the agentic endpoint) ----------

class ChatMessage(BaseModel):
    message: str
    history: list[dict] = []


@app.post("/api/chat")
async def chat(body: ChatMessage):
    """
    The agentic endpoint. Gemma calls our tools (Spotify + RAG) and writes a grounded reply.
    """
    tools = build_tool_definitions()
    response = await state.llm.chat_with_tools(
        message=body.message,
        history=body.history,
        tools=tools,
        tool_handler=handle_tool_call,
        system=ARCHIVIST_SYSTEM_PROMPT,
    )
    return response


ARCHIVIST_SYSTEM_PROMPT = """You are The Archivist — a writer and analyst with access to one
listener's full music catalogue. You have five retrieval tools (lyrics, artist context,
listening memory, personal notes, genre knowledge) and direct Spotify queries via MCP-style
tools. Your style is editorial and considered — closer to a music critic than a chatbot.

Rules:
- Always retrieve before claiming a fact. Cite the source in superscript like <sup>1</sup>.
- Be specific (track names, play counts, dates) — never generic.
- Keep replies under 5 sentences unless asked for more.
- If retrieval returns nothing, say so plainly.
"""


def build_tool_definitions() -> list[dict]:
    """Tool schemas for Gemma function calling (Google AI Studio format)."""
    return [
        {
            "name": "search_lyrics",
            "description": "Semantic search over the user's indexed song lyrics. Use for themes, moods, or phrases.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
                "required": ["query"],
            },
        },
        {
            "name": "search_memory",
            "description": "Search the listening-history index by semantic context (e.g. 'exams in March').",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
                "required": ["query"],
            },
        },
        {
            "name": "get_artist_context",
            "description": "Retrieve indexed Wikipedia context for a named artist.",
            "parameters": {
                "type": "object",
                "properties": {"artist": {"type": "string"}},
                "required": ["artist"],
            },
        },
        {
            "name": "get_top_artists",
            "description": "Get the user's top N artists for a time range (short_term, medium_term, long_term).",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_range": {"type": "string", "enum": ["short_term", "medium_term", "long_term"]},
                    "limit": {"type": "integer"},
                },
            },
        },
        {
            "name": "search_notes",
            "description": "Search the user's own annotations about tracks/artists.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    ]


async def handle_tool_call(name: str, args: dict) -> dict:
    """Dispatcher called by the LLM client when Gemma requests a tool."""
    # Gemini returns numeric args as floats (e.g. 5.0); coerce to int where we use them as slice indices.
    def _int(v, default):
        try:
            return int(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    if name == "search_lyrics":
        return {"results": state.lyrics_rag.search(args["query"], k=_int(args.get("k"), 4))}
    if name == "search_memory":
        return {"results": state.memory_rag.search(args["query"], k=_int(args.get("k"), 5))}
    if name == "get_artist_context":
        return {"context": state.artist_rag.get(args["artist"])}
    if name == "get_top_artists":
        limit = _int(args.get("limit"), 10)
        if state.demo_mode:
            with open(DATA_DIR / "demo_data.json") as f:
                return {"artists": json.load(f).get("top_artists", [])[:limit]}
        return {"artists": await state.spotify.get_top_artists(args.get("time_range", "long_term"), limit)}
    if name == "search_notes":
        return {"results": state.notes_rag.search(args["query"])}
    return {"error": f"unknown tool {name}"}


# ---------- Demo data loader ----------

async def load_demo_data():
    """Populate all RAGs with pre-baked sample data so the app works without Spotify."""
    print("→ Loading demo data…")
    with open(DATA_DIR / "demo_data.json") as f:
        data = json.load(f)
    for track in data.get("lyrics_corpus", []):
        state.lyrics_rag.index_document(
            doc_id=track["id"], text=track["lyrics"],
            metadata={"title": track["title"], "artist": track["artist"]},
        )
    for artist in data.get("artist_bios", []):
        state.artist_rag.index_document(
            doc_id=artist["name"], text=artist["bio"],
            metadata={"name": artist["name"]},
        )
    for note in data.get("notes", []):
        state.notes_rag.add(note["track_id"], note["text"])
    print(f"✓ Indexed {len(data.get('lyrics_corpus', []))} lyrics, {len(data.get('artist_bios', []))} bios.")


# ---------- Health ----------

@app.get("/api/health")
async def health():
    return {"ok": True, "demo": state.demo_mode}


# ---------- Entry ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The Listening Archive")
    parser.add_argument("--demo", action="store_true", help="Run with pre-baked sample data, no Spotify needed.")
    parser.add_argument("--port", type=int, default=3000)
    args = parser.parse_args()
    state.demo_mode = args.demo
    if args.demo:
        print("✦ Running in DEMO mode — no Spotify credentials required.")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
