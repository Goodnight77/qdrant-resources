"""
common.py : shared helpers for the Qdrant 1.19 demo scripts

we generate synthetic vectors with numpy instead of a real embedding model,
so the demos run anywhere in seconds with zero API keys and zero downloads.
the behavior of turbo4 / memory tiers / slicing / prefix matching is exactly
the same whether the vectors come from OpenAI or from numpy.
"""

import numpy as np
from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6333"

# 1024 dims to mimic a modern embedding model (e.g. Cohere embed-v3 / BGE-large)
DIM = 1024
N_POINTS = 5_000
SEED = 77


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, timeout=60)


def make_vectors(n: int = N_POINTS, dim: int = DIM, seed: int = SEED) -> np.ndarray:
    """generate reproducible unit-norm vectors (cosine-friendly)."""
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def brute_force_topk(vectors: np.ndarray, query: np.ndarray, k: int = 10) -> list[int]:
    """exact cosine search with numpy = the ground truth we measure recall against."""
    scores = vectors @ query  # vectors are normalized, so dot == cosine
    return list(np.argsort(-scores)[:k])


def recall_at_k(ground_truth: list[int], retrieved: list[int]) -> float:
    return len(set(ground_truth) & set(retrieved)) / len(ground_truth)
