"""Shared config for the quantization faceoff.
Dataset: dbpedia-entities-openai-1M (OpenAI ada-002, 1536-d, cosine) high-dim text
embeddings with title/text payload, so the quantization tradeoff is dramatic and the
Qdrant UI shows real content.
"""

import os

from qdrant_client import models

QDRANT_URL = "http://localhost:6333"

RESULTS_DIR = "results"  # benchmark output JSONs land here


def results_path(name):
    """Path under the results dir, ensuring the dir exists."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, name)


# dbpedia-openai facts
DIM = 1536
DISTANCE = models.Distance.COSINE

# 100k = ~600MB embeddings, fast + tells the full story. Fixed at 100k: 1M needs
# ~25GB disk (3 collections keep f32 originals) which doesn't fit my 15GB budget
DATASET_N = 100_000
N_QUERY = 1_000  # held out from the tail; NOT inserted into base
K = 10  # Recall@K
GT_DEPTH = 100  # exact-NN depth to store as ground truth
BATCH_SIZE = 500  # upsert batch (kept modest; 1536-d float vectors are large)
DATA_DIR = "data/dbpedia"

# 3 configs under test. quantization=None => baseline f32.
COLLECTIONS = {
    "baseline": None,
    "sq": models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            always_ram=True,
        )
    ),
    # PQ; TurboQuant query-side optimizations apply automatically on 1.18+
    "pq": models.ProductQuantization(
        product=models.ProductQuantizationConfig(
            compression=models.CompressionRatio.X16,
            always_ram=True,
        )
    ),
    # Binary: 1 bit/dim = 32x smaller. Very lossy raw; needs rescore+oversampling
    "bq": models.BinaryQuantization(
        binary=models.BinaryQuantizationConfig(always_ram=True)
    ),
}
