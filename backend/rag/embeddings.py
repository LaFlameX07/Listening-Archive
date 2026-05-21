"""
Embeddings — local, CPU, free.
Uses sentence-transformers/all-MiniLM-L6-v2 (~80MB, 384 dims).
Downloads once on first run; cached forever after in ~/.cache/torch.
"""
from __future__ import annotations

from typing import Sequence

from sentence_transformers import SentenceTransformer

from backend.config import settings


class Embedder:
    _model: SentenceTransformer | None = None

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL

    @property
    def model(self) -> SentenceTransformer:
        if Embedder._model is None:
            print(f"→ Loading embedding model {self.model_name}…")
            Embedder._model = SentenceTransformer(self.model_name)
            print(f"✓ Embedder ready (dim={Embedder._model.get_sentence_embedding_dimension()}).")
        return Embedder._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        vectors = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
