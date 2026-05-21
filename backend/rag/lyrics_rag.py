"""
RAG Layer 1 — Lyrics.

Pulls song lyrics from Genius, chunks them, embeds locally, stores in ChromaDB.
Enables queries like "songs about loneliness late at night" → returns the songs
whose lyric meaning is closest in vector space.

Why this matters: Spotify removed audio-features in 2024, so we can't ask
"is this song sad?" via the API. But we CAN ask via the lyrics — and that's
arguably a more honest signal of what a song is actually about.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import chromadb
import httpx
from chromadb.config import Settings as ChromaSettings

from backend.config import settings
from backend.rag.embeddings import Embedder


COLLECTION = "lyrics"
GENIUS_BASE = "https://api.genius.com"


class LyricsRAG:
    def __init__(self, embedder: Embedder, chroma_path: Path):
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=str(chroma_path), settings=ChromaSettings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    # --- Indexing ---

    def index_document(self, doc_id: str, text: str, metadata: dict) -> None:
        """Embed and store a single lyric document."""
        # Chunk long lyrics into ~200-token windows for finer retrieval
        chunks = _chunk(text, max_words=80)
        if not chunks:
            return
        embeddings = self.embedder.encode(chunks)
        ids = [f"{doc_id}::chunk_{i}" for i in range(len(chunks))]
        metas = [{**metadata, "chunk_index": i, "parent_id": doc_id} for i in range(len(chunks))]
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metas)

    async def ingest_track_from_genius(self, title: str, artist: str, track_id: str) -> bool:
        """Fetch lyrics from Genius for one track and index them. Returns True on success."""
        if not settings.GENIUS_TOKEN:
            print("(no GENIUS_TOKEN set, skipping)")
            return False
        try:
            import asyncio
            import lyricsgenius
            genius = lyricsgenius.Genius(
                settings.GENIUS_TOKEN,
                verbose=False,
                remove_section_headers=True,
                timeout=15,
                retries=2,
            )
            # lyricsgenius is sync — run in a thread so we don't block the event loop
            song = await asyncio.to_thread(genius.search_song, title, artist)
            if not song or not getattr(song, "lyrics", None):
                return False
            self.index_document(
                track_id,
                song.lyrics,
                {
                    "title": title,
                    "artist": artist,
                    "genius_url": song.url,
                },
            )
            return True
        except Exception as e:
            print(f"  ✗ Lyrics ingest failed for '{title}' by {artist}: {e}")
            return False

    # --- Search ---

    def search(self, query: str, k: int = 4) -> list[dict]:
        """Semantic search. Returns top-k chunks with cosine similarity."""
        if self.collection.count() == 0:
            return []
        q_emb = self.embedder.encode_one(query)
        res = self.collection.query(query_embeddings=[q_emb], n_results=k)
        results = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            results.append({
                "title": meta.get("title", "—"),
                "artist": meta.get("artist", "—"),
                "snippet": doc[:280],
                "full_text": doc,  # the entire indexed chunk for the expand view
                "score": round(1.0 - dist, 3),
                "parent_id": meta.get("parent_id"),
                "genius_url": meta.get("genius_url"),
            })
        return results

    def count(self) -> int:
        return self.collection.count()


def _chunk(text: str, max_words: int = 80) -> list[str]:
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _extract_lyrics_from_html(html: str) -> str:
    """Deprecated — lyricsgenius handles parsing now. Kept for backward compat."""
    return ""
