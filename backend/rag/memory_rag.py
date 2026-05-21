"""
RAG Layer 3 — Listening Memory.

Embeds each play event with rich text context:
"Played 'Holocene' by Bon Iver at 02:47am on 2026-02-18 — late night session, second time this week."

This makes the listening history semantically searchable.
Ask: "what was I listening to during exams?" → time-window context + semantic
similarity to "studying / exams / late nights" → relevant plays surface.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.database import Database
from backend.rag.embeddings import Embedder


COLLECTION = "listening_memory"


class MemoryRAG:
    def __init__(self, embedder: Embedder, chroma_path: Path, db: Database):
        self.embedder = embedder
        self.db = db
        self.client = chromadb.PersistentClient(path=str(chroma_path), settings=ChromaSettings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def index_play(self, *, track_id: str, track_name: str, artist_name: str, played_at: str) -> None:
        """Convert a play event into a rich text representation and embed it."""
        text = self._format_event(track_name, artist_name, played_at)
        doc_id = f"{track_id}::{played_at}"
        emb = self.embedder.encode_one(text)
        self.collection.upsert(
            ids=[doc_id], embeddings=[emb], documents=[text],
            metadatas=[{
                "track_id": track_id, "track": track_name,
                "artist": artist_name, "played_at": played_at,
            }],
        )

    def reindex_all_plays(self) -> int:
        """Walk all plays in SQLite and embed them. Useful one-shot after big ingest."""
        plays = self.db.recent_plays(limit=10_000)
        n = 0
        for p in plays:
            self.index_play(
                track_id=p["track_id"], track_name=p["track_name"],
                artist_name=p["artist_name"], played_at=p["played_at"],
            )
            n += 1
        return n

    def search(self, query: str, k: int = 5) -> list[dict]:
        if self.collection.count() == 0:
            return []
        q_emb = self.embedder.encode_one(query)
        res = self.collection.query(query_embeddings=[q_emb], n_results=k)
        return [
            {
                "track": meta.get("track"),
                "artist": meta.get("artist"),
                "played_at": meta.get("played_at"),
                "context": doc,
                "score": round(1.0 - dist, 3),
            }
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    @staticmethod
    def _format_event(track: str, artist: str, played_at: str) -> str:
        """Build the text representation that gets embedded."""
        try:
            ts = datetime.fromisoformat(played_at.replace("Z", "+00:00"))
            time_of_day = _time_of_day(ts.hour)
            weekday = ts.strftime("%A")
            month = ts.strftime("%B %Y")
            return (
                f"Played '{track}' by {artist} at {ts.strftime('%H:%M')} on {weekday}, {month}. "
                f"This was a {time_of_day} listening session."
            )
        except Exception:
            return f"Played '{track}' by {artist} at {played_at}."


def _time_of_day(hour: int) -> str:
    if hour < 5: return "late night"
    if hour < 9: return "early morning"
    if hour < 12: return "morning"
    if hour < 14: return "midday"
    if hour < 18: return "afternoon"
    if hour < 22: return "evening"
    return "night"
