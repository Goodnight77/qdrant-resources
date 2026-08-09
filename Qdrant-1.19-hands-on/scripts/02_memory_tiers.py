"""
02_memory_tiers.py

Qdrant 1.19 replaces the confusing trio of `on_disk`, `always_ram`,
and `on_disk_payload` with ONE parameter that works the same everywhere:

    memory = "pinned" | "cached" | "cold"

  pinned -> loaded on the heap, never evicted (fastest, must fit in RAM)
  cached -> lives on disk, pre-warmed into the OS page cache at startup,
            evictable under memory pressure
  cold   -> lives on disk, loaded lazily on first access (cheapest RAM)

This script builds a collection where every component gets an explicit tier:
  - original vectors      -> cached  (mmap + warm page cache)
  - HNSW index            -> cold    (lazy, saves startup RAM)
  - quantized vectors     -> pinned  (tiny + hot path = keep in heap)
  - payloads              -> cached

Then it prints the applied config back from the server so you can verify.
"""

from qdrant_client import models

from common import get_client, make_vectors, DIM

NAME = "demo_memory_tiers"


def main():
    client = get_client()

    if client.collection_exists(NAME):
        client.delete_collection(NAME)

    client.create_collection(
        collection_name=NAME,
        vectors_config=models.VectorParams(
            size=DIM,
            distance=models.Distance.COSINE,
            memory=models.Memory.CACHED,          # original vectors: warm disk cache
        ),
        hnsw_config=models.HnswConfigDiff(
            memory=models.Memory.COLD,            # graph links: lazy-load from disk
        ),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                memory=models.Memory.PINNED,      # compressed copy: locked in RAM
            ),
        ),
        payload=models.PayloadStorageParams(
            memory=models.Memory.CACHED,          # payload JSON: warm disk cache
        ),
    )

    # upload a bit of data so the collection is not empty
    vectors = make_vectors(n=1_000)
    client.upload_collection(
        collection_name=NAME,
        vectors=vectors,
        ids=list(range(len(vectors))),
        payload=[{"doc": f"document {i}"} for i in range(len(vectors))],
        batch_size=512,
    )

    info = client.get_collection(NAME)
    cfg = info.config

    print(f"collection '{NAME}' created. tiers as applied by the server:\n")
    print(f"  dense vectors  memory : {cfg.params.vectors.memory}")
    print(f"  hnsw index     memory : {cfg.hnsw_config.memory}")
    print(f"  quantized vecs memory : {cfg.quantization_config.scalar.memory}")
    print(f"  payload        memory : {cfg.params.payload.memory if cfg.params.payload else 'default (cold)'}")

    print("""
why this particular combo works for a disk-first setup:
  - the int8 quantized copy is small, so pinning it costs little RAM
    and every search scores candidates from the heap (no disk hit)
  - the float32 originals are only read to rescore the top candidates,
    'cached' keeps those reads warm without locking gigabytes in RAM
  - the HNSW graph on 'cold' trades slower first queries for a leaner
    startup; switch it to 'cached' if p99 on cold starts matters to you

note: 'pinned' is rejected for dense vectors and payloads. those two only
support memory-mapped representations, so 'cached' is their fastest tier.
""")


if __name__ == "__main__":
    main()
