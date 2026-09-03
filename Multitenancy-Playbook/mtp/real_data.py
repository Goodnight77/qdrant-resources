"""Real embeddings for experiments 1 and 2: OpenAI ada-002 (1536-d) vectors
from dbpedia-entities-openai-1M (https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M),
real Wikipedia articles, same dataset used by ../Quantization-FaceOff/ in this repo.

Downloads two parquet shards (~700MB, ~77k rows) via huggingface_hub on first
run and caches them under data/.hf_cache/ (gitignored). Tenant assignment is
still independent of content (see mtp/data.py's docstring for why): using
real embeddings instead of synthetic clusters mainly changes the *shape* of
that content structure to something a real HNSW graph actually has to deal
with, rather than a Gaussian mixture's artificially clean separation.
"""

import numpy as np
from huggingface_hub import hf_hub_download

REPO_ID = "KShivendu/dbpedia-entities-openai-1M"
SHARD_FILES = [
    "data/train-00000-of-00026-3c7b99d1c7eda36e.parquet",
    "data/train-00001-of-00026-2b24035a6390fdcb.parquet",
]
CACHE_DIR = "data/.hf_cache"


def load_dbpedia_vectors(n_query, seed):
    """Returns (base_vectors, query_vectors), both float32, cosine-ready.
    query_vectors are held out from the base set entirely (real, unseen
    documents, not synthetic perturbations)."""
    import pandas as pd

    frames = []
    for fname in SHARD_FILES:
        path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=fname, cache_dir=CACHE_DIR)
        frames.append(pd.read_parquet(path, columns=["openai"]))
    df = pd.concat(frames, ignore_index=True)
    vectors = np.stack(df["openai"].to_numpy()).astype(np.float32)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(vectors))
    query_idx, base_idx = idx[:n_query], idx[n_query:]
    return vectors[base_idx], vectors[query_idx]
