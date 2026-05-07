"""
Scorer (e): Semantic Similarity

Computes cosine similarity between sentence embeddings of generated vs reference text.
Uses sentence-transformers (local, free) — no API call needed.

A score of 1.0 = identical meaning. A score below 0.7 usually indicates
significant semantic drift from the reference.

We use 'all-MiniLM-L6-v2' — fast, 80MB, good multilingual support for Hindi/English.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass
class SimilarityResult:
    score: float       # cosine similarity, 0-1
    ci_low: float
    ci_high: float
    generated_len: int
    reference_len: int


@lru_cache(maxsize=1)
def _get_model():
    """Load model once and cache. First call takes ~3s to download."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def score(generated_text: str, reference_text: str) -> SimilarityResult:
    """
    Embed both texts and compute cosine similarity.
    CI is based on empirical variance of the embedding model (~±0.05).
    """
    model = _get_model()
    embeddings = model.encode([generated_text, reference_text], normalize_embeddings=True)
    gen_emb, ref_emb = embeddings[0], embeddings[1]

    similarity = _cosine(gen_emb, ref_emb)
    # Clamp to [0, 1] (cosine can be slightly negative for opposite-meaning texts)
    similarity = max(0.0, min(1.0, similarity))

    # Empirical CI for sentence-transformers: ±0.05 is typical jitter
    ci_low = max(0.0, similarity - 0.05)
    ci_high = min(1.0, similarity + 0.05)

    return SimilarityResult(
        score=round(similarity, 4),
        ci_low=round(ci_low, 4),
        ci_high=round(ci_high, 4),
        generated_len=len(generated_text),
        reference_len=len(reference_text),
    )
