"""Lightweight embedding utilities for in-memory retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def _load_encoder(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


@dataclass(slots=True)
class TransformerEmbedder:
    """Wrapper around a huggingface encoder for sentence embeddings."""

    model_name: str = DEFAULT_EMBED_MODEL
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer: AutoTokenizer = field(init=False, repr=False)
    model: AutoModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.tokenizer, self.model = _load_encoder(self.model_name)
        self.model.to(self.device)

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)

        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        with torch.no_grad():
            model_output = self.model(**encodings)
            token_embeddings = model_output.last_hidden_state  # (batch, seq, dim)
            attention_mask = encodings["attention_mask"].unsqueeze(-1)
            masked = token_embeddings * attention_mask
            summed = masked.sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1)
            embeddings = summed / counts

        embeddings = embeddings.cpu().numpy()
        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings.astype(np.float32)


def embed_text(text: str, embedder: TransformerEmbedder) -> np.ndarray:
    """Convenience helper to embed a single string."""
    return embedder.embed([text])[0]


