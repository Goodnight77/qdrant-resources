"""Create the 3 collections (baseline / sq / pq) and load the dbpedia embeddings into each.
Waits for optimizer to reach green before exiting (isolation guardrail)."""
import time

from qdrant_client import QdrantClient, models

from qfo.config import (
    QDRANT_URL, DIM, DISTANCE, BATCH_SIZE, COLLECTIONS,
)
from qfo.data.dbpedia import load_dbpedia


def make_collection(client, name, quant):
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=DIM, distance=DISTANCE),
        quantization_config=quant,  # None => baseline f32
    )


def wait_green(client, name, timeout=3600):
    """Block until indexing/optimization finishes. Raises on timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = client.get_collection(name)
        if info.status == models.CollectionStatus.GREEN:
            return
        time.sleep(5)
    raise TimeoutError(f"{name} not green after {timeout}s (status={info.status})")


def main():
    base, payload, _queries = load_dbpedia()
    ids = list(range(len(base)))
    client = QdrantClient(url=QDRANT_URL, timeout=120)

    for name, quant in COLLECTIONS.items():
        print(f"[{name}] create + upload {len(base)} vectors")
        make_collection(client, name, quant)
        client.upload_collection(
            collection_name=name,
            vectors=base,
            payload=payload,      # title/text so the UI shows real content
            ids=ids,
            batch_size=BATCH_SIZE,
            parallel=4,
        )
        wait_green(client, name)
        cnt = client.count(name).count
        assert cnt == len(base), f"{name}: {cnt} != {len(base)}"
        print(f"[{name}] green, count={cnt}")

    print("ingest done: 3 collections green.")


if __name__ == "__main__":
    main()
