"""
RAG Layer 4 — User Notes.

Stores and semantically searches the user's own annotations.
"This song reminds me of summer 2024" → embedded, indexed, retrievable.
The most personal layer. Lives only on this machine.
"""
from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.database import Database
from backend.rag.embeddings import Embedder


COLLECTION = "user_notes"


class NotesRAG:
    def __init__(self, embedder: Embedder, chroma_path: Path, db: Database):
        self.embedder = embedder
        self.db = db
        self.client = chromadb.PersistentClient(path=str(chroma_path), settings=ChromaSettings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def add(self, track_id: str, text: str) -> int:
        note_id = self.db.add_note(track_id, text)
        emb = self.embedder.encode_one(text)
        self.collection.upsert(
            ids=[f"note_{note_id}"], embeddings=[emb], documents=[text],
            metadatas=[{"note_id": note_id, "track_id": track_id}],
        )
        return note_id

    def search(self, query: str, k: int = 4) -> list[dict]:
        if self.collection.count() == 0:
            return []
        q_emb = self.embedder.encode_one(query)
        res = self.collection.query(query_embeddings=[q_emb], n_results=k)
        return [
            {
                "note": doc, "track_id": meta.get("track_id"),
                "note_id": meta.get("note_id"), "score": round(1.0 - dist, 3),
            }
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]
