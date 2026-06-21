"""Tangible search demo: pick an article, find the most similar ones.

No embedding model needed (we'd need OpenAI ada-002 to embed fresh text) — instead we
search BY EXAMPLE: take a stored article's vector and find its neighbors. Uses BQ x32 +
8x rescore (the recommended config) so you see 32x-compressed search returning real,
relevant titles.

Usage:
  uv run python search_demo.py                 # random seed article
  uv run python search_demo.py einstein        # first article whose title contains 'einstein'
"""

import random
import sys

from qdrant_client import QdrantClient, models

from qfo.config import QDRANT_URL, K
from qfo.data.dbpedia import load_dbpedia

RESCORE = models.SearchParams(
    quantization=models.QuantizationSearchParams(rescore=True, oversampling=8.0)
)


def main():
    base, payload, _q = load_dbpedia()
    client = QdrantClient(url=QDRANT_URL, timeout=60)

    arg = " ".join(sys.argv[1:]).strip()
    if arg:
        hits = [i for i, p in enumerate(payload) if arg.lower() in p["title"].lower()]
        if not hits:
            print(f"no article title contains {arg!r}")
            return
        seed = hits[0]
    else:
        seed = random.randrange(len(base))

    print(f"\nQuery article [{seed}]: {payload[seed]['title']}")
    print(f"  {payload[seed]['text'][:160]}...\n")

    res = client.query_points(
        "bq", query=base[seed].tolist(), limit=K + 1, search_params=RESCORE
    )
    neighbors = [p for p in res.points if p.id != seed][:K]
    assert neighbors, "no neighbors returned"

    print(f"Most similar (BQ x32 + rescore 8x), top {len(neighbors)}:")
    for p in neighbors:
        print(f"  {p.score:.3f}  {payload[p.id]['title']}")


if __name__ == "__main__":
    main()
