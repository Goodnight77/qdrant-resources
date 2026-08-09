"""
03_prefix_matching.py

Before 1.19: keyword indexes only did EXACT matching. Filtering "give me every
point whose url starts with https://qdrant." meant either a full payload scan
or switching to a text index (which tokenizes and breaks exact matching).

In 1.19: keyword indexes accept `prefix=True`, and filters get a new
`MatchPrefix` condition served straight from a dedicated index structure,
so it's as fast as any other indexed filter.

This script:
  1. creates a collection with url payloads
  2. builds a keyword index with prefix support enabled
  3. runs prefix filters and prints the matches
"""

from common import get_client, make_vectors
from qdrant_client import models

NAME = "demo_prefix"
DIM_SMALL = 128

URLS = [
    "https://qdrant.tech/documentation/quickstart",
    "https://qdrant.tech/blog/qdrant-1.19.x",
    "https://qdrant.tech/articles/turboquant-quantization",
    "https://github.com/qdrant/qdrant",
    "https://github.com/Goodnight77/qdrant-resources",
    "https://medium.com/@mohammedarbinsibi",
    "s3://prod-bucket/tenant-a/2026/01/report.pdf",
    "s3://prod-bucket/tenant-a/2026/02/report.pdf",
    "s3://prod-bucket/tenant-b/2026/01/report.pdf",
    "s3://staging-bucket/tenant-a/2026/01/report.pdf",
]


def main():
    client = get_client()

    if client.collection_exists(NAME):
        client.delete_collection(NAME)

    client.create_collection(
        collection_name=NAME,
        vectors_config=models.VectorParams(
            size=DIM_SMALL, distance=models.Distance.COSINE
        ),
    )

    # keyword index WITH prefix support (new in 1.19)
    client.create_payload_index(
        collection_name=NAME,
        field_name="url",
        field_schema=models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD,
            prefix=True,  # <-- this single flag builds the prefix-capable index
        ),
    )

    vectors = make_vectors(n=len(URLS), dim=DIM_SMALL)
    client.upload_collection(
        collection_name=NAME,
        vectors=vectors,
        ids=list(range(len(URLS))),
        payload=[{"url": u} for u in URLS],
    )

    for prefix in [
        "https://qdrant.",
        "s3://prod-bucket/tenant-a/",
        "https://github.com/",
    ]:
        points, _ = client.scroll(
            collection_name=NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="url",
                        match=models.MatchPrefix(prefix=prefix),  # <-- new condition
                    )
                ]
            ),
            limit=100,
            with_payload=True,
        )
        print(f"prefix '{prefix}' -> {len(points)} matches")
        for p in points:
            print(f"    {p.payload['url']}")
        print()

    print("note: matching is byte-wise and case-sensitive, same as exact keyword")
    print("matching. this is what makes it right for urls, paths, and SKUs where")
    print("tokenization (a text index) would destroy the identifier structure.")


if __name__ == "__main__":
    main()
