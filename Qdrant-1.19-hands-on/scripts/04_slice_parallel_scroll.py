"""
04_slice_parallel_scroll.py

The new `slice` filter condition (1.19) partitions a collection into
deterministic, disjoint subsets based on a stable hash of the point ID:

    a point belongs to slice `index` of `total` if  hash(id) % total == index

Two things this unlocks:
  1. PARALLEL SCROLLING: give each worker its own slice and scan the whole
     collection concurrently with ZERO coordination between workers.
  2. REPRODUCIBLE SAMPLING: the same slice always returns the same points,
     across queries and across Qdrant versions (the hash is SipHash-2-4 with
     a zero key and is guaranteed stable). perfect for benchmarks and
     train/test splits.

This script proves both properties:
  - scrolls 4 slices in parallel threads and verifies they are disjoint
    and together cover 100% of the collection
  - re-reads one slice twice and verifies it returns identical point sets
"""

import time
from concurrent.futures import ThreadPoolExecutor

from qdrant_client import models

from common import get_client, make_vectors

NAME = "demo_slices"
DIM_SMALL = 128
N = 2_000
TOTAL_SLICES = 4


def scroll_slice(slice_index: int) -> set:
    """each worker gets its own client + its own slice = zero coordination."""
    client = get_client()
    ids, offset = set(), None
    while True:
        points, offset = client.scroll(
            collection_name=NAME,
            scroll_filter=models.Filter(
                must=[
                    models.SliceCondition(
                        slice=models.Slice(index=slice_index, total=TOTAL_SLICES)
                    )
                ]
            ),
            limit=500,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        ids.update(p.id for p in points)
        if offset is None:
            break
    return ids


def main():
    client = get_client()

    if client.collection_exists(NAME):
        client.delete_collection(NAME)
    client.create_collection(
        collection_name=NAME,
        vectors_config=models.VectorParams(size=DIM_SMALL, distance=models.Distance.COSINE),
    )
    client.upload_collection(
        collection_name=NAME,
        vectors=make_vectors(n=N, dim=DIM_SMALL),
        ids=list(range(N)),
    )

    # --- property 1: parallel + disjoint + complete coverage -------------
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=TOTAL_SLICES) as pool:
        slice_sets = list(pool.map(scroll_slice, range(TOTAL_SLICES)))
    elapsed = time.perf_counter() - t0

    sizes = [len(s) for s in slice_sets]
    union = set().union(*slice_sets)
    overlap = sum(sizes) - len(union)

    print(f"scrolled {TOTAL_SLICES} slices in parallel in {elapsed:.2f}s")
    print(f"slice sizes            : {sizes}")
    print(f"total covered          : {len(union)} / {N} points")
    print(f"overlap between slices : {overlap}  (must be 0 -> disjoint)")
    assert len(union) == N and overlap == 0, "slices should partition the collection"

    # --- property 2: deterministic / reproducible sampling ---------------
    first = scroll_slice(0)
    second = scroll_slice(0)
    print(f"\nslice 0 read twice     : identical = {first == second}")
    print("-> the same slice ALWAYS returns the same subset. use it for recall")
    print("   benchmarks, canary rollouts, or train/test splits.")

    # bonus fact from the docs: slices with different totals are correlated.
    # slice 0 of total=4 is always a subset of slice 0 of total=2.
    half = scroll_slice_with_total(0, 2)
    quarter = scroll_slice_with_total(0, 4)
    print(f"\nslice 0/4 subset of slice 0/2 : {quarter.issubset(half)}")


def scroll_slice_with_total(slice_index: int, total: int) -> set:
    client = get_client()
    ids, offset = set(), None
    while True:
        points, offset = client.scroll(
            collection_name=NAME,
            scroll_filter=models.Filter(
                must=[models.SliceCondition(slice=models.Slice(index=slice_index, total=total))]
            ),
            limit=500,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        ids.update(p.id for p in points)
        if offset is None:
            break
    return ids


if __name__ == "__main__":
    main()
