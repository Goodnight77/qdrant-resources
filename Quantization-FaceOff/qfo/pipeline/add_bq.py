"""Build a single collection by name without touching the others (no drops to the
existing baseline/sq/pq). Usage: uv run python add_bq.py [name]   default: bq

Logic stays under __main__ + main(): upload_collection(parallel>1) uses process spawn
on Windows, which re-imports this module — unguarded top-level code would re-run it.
"""
import sys

from qdrant_client import QdrantClient

from qfo.config import QDRANT_URL, BATCH_SIZE, COLLECTIONS
from qfo.data.dbpedia import load_dbpedia
from qfo.pipeline.ingest import make_collection, wait_green


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "bq"
    base, payload, _q = load_dbpedia()
    client = QdrantClient(url=QDRANT_URL, timeout=120)
    print(f"[{name}] create + upload {len(base)} vectors")
    make_collection(client, name, COLLECTIONS[name])
    client.upload_collection(
        collection_name=name, vectors=base, payload=payload,
        ids=list(range(len(base))), batch_size=BATCH_SIZE, parallel=4,
    )
    wait_green(client, name)
    print(f"[{name}] green, count={client.count(name).count}")


if __name__ == "__main__":
    main()
