"""BM25 on a full-text index (Qdrant 1.20): no sparse vectors, no embeddings."""

from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")
COL = "books"

books = [
    "The Time Machine",
    "A Brief History of Time",
    "Time Travel for Beginners: A Practical Guide",
    "The Time Traveler's Wife",
    "Travel Guide to Italy",
    "The Hitchhiker's Guide to the Galaxy",
    "Dune",
    "11/22/63: A Novel About Time Travel and Kennedy",
]

if client.collection_exists(COL):
    client.delete_collection(COL)

# no vectors at all, payload only
client.create_collection(collection_name=COL, vectors_config={})
client.upsert(
    collection_name=COL,
    points=[
        models.PointStruct(id=i, vector={}, payload={"title": t})
        for i, t in enumerate(books)
    ],
)

# 1.20: full-text index with BM25 scoring, built from the payload you already have
client.create_payload_index(
    collection_name=COL,
    field_name="title",
    field_schema=models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        tokenizer=models.TokenizerType.WORD,
        lowercase=True,
        scoring=models.TextScoringParams(type=models.TextScoringType.BM25),
    ),
    wait=True,
)


def show(label, res):
    print(f"\n{label}")
    for p in res.points:
        print(f"  {p.score:6.3f}  {p.payload['title']}")


# plain text query, default k=1.2, b=0.75
show(
    "query='time travel' (default k=1.2, b=0.75)",
    client.query_points(
        COL, query=models.TextQuery(text="time travel"), using="title", limit=5
    ),
)

# per-query BM25 params, no index rebuild
show(
    "query='time travel' (k=1.5, b=0.0 -> no length normalization)",
    client.query_points(
        COL,
        query=models.TextQuery(
            text=models.TextQueryInput(
                query="time travel",
                scoring=models.Bm25Scoring(bm25=models.Bm25Params(k=1.5, b=0.0)),
            )
        ),
        using="title",
        limit=5,
    ),
)
