"""
RAG Layer 5 — Genre Knowledge.

Indexes Wikipedia articles for the user's top genres (e.g. 'slowcore', 'shoegaze')
so the LLM can ground genre explanations in real text.

"What is slowcore?" → retrieves the genre article → answers with citation.
"""
from __future__ import annotations

from pathlib import Path

import chromadb
import httpx
from chromadb.config import Settings as ChromaSettings

from backend.rag.embeddings import Embedder


COLLECTION = "genres"
WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary"


class GenreRAG:
    def __init__(self, embedder: Embedder, chroma_path: Path):
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=str(chroma_path), settings=ChromaSettings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def index_document(self, doc_id: str, text: str, metadata: dict) -> None:
        chunks = _chunk(text, 120)
        if not chunks:
            return
        embs = self.embedder.encode(chunks)
        self.collection.upsert(
            ids=[f"{doc_id}::p{i}" for i in range(len(chunks))],
            embeddings=embs, documents=chunks,
            metadatas=[{**metadata, "chunk": i} for i in range(len(chunks))],
        )

    async def ingest_genre(self, genre: str) -> bool:
        title = genre.replace(" ", "_") + "_(music)"
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{WIKI_API}/{title}")
            if r.status_code != 200:
                r = await c.get(f"{WIKI_API}/{genre.replace(' ', '_')}")
                if r.status_code != 200:
                    return False
        data = r.json()
        extract = data.get("extract", "")
        if not extract:
            return False
        self.index_document(genre, extract, {"genre": genre})
        return True

    def search(self, query: str, k: int = 3) -> list[dict]:
        if self.collection.count() == 0:
            return []
        emb = self.embedder.encode_one(query)
        res = self.collection.query(query_embeddings=[emb], n_results=k)
        return [
            {"genre": meta.get("genre"), "text": doc, "score": round(1.0 - dist, 3)}
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]


def _chunk(text: str, max_words: int) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)] if words else []
