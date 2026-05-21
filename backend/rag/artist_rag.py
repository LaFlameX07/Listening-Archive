"""
RAG Layer 2 — Artist Context.

Indexes Wikipedia bios of the user's top artists.
Lets the LLM ground claims about artists in real text rather than hallucinating.
Free: Wikipedia REST API needs no auth.
"""
from __future__ import annotations

from pathlib import Path

import chromadb
import httpx
from chromadb.config import Settings as ChromaSettings

from backend.rag.embeddings import Embedder


COLLECTION = "artist_bios"
WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary"


class ArtistRAG:
    def __init__(self, embedder: Embedder, chroma_path: Path):
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=str(chroma_path), settings=ChromaSettings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def index_document(self, doc_id: str, text: str, metadata: dict) -> None:
        chunks = _chunk(text, max_words=120)
        if not chunks:
            return
        embeddings = self.embedder.encode(chunks)
        ids = [f"{doc_id}::p{i}" for i in range(len(chunks))]
        metas = [{**metadata, "chunk": i} for i in range(len(chunks))]
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metas)

    async def ingest_artist(self, name: str) -> bool:
        """Fetch the Wikipedia summary for an artist and index it."""
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{WIKI_API}/{name.replace(' ', '_')}")
        if r.status_code != 200:
            return False
        data = r.json()
        extract = data.get("extract", "")
        if not extract:
            return False
        self.index_document(name, extract, {"name": name, "wiki_title": data.get("title")})
        return True

    def get(self, artist: str, k: int = 3) -> list[dict]:
        """Retrieve top-k bio chunks for an artist by name + semantic match."""
        if self.collection.count() == 0:
            return []
        q_emb = self.embedder.encode_one(artist)
        res = self.collection.query(query_embeddings=[q_emb], n_results=k)
        return [
            {"text": doc, "name": meta.get("name"), "score": round(1.0 - dist, 3)}
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]


def _chunk(text: str, max_words: int = 120) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)] if words else []
