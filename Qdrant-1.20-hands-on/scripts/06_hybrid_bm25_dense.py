"""Hybrid search: BM25 on a full-text index + dense vectors, fused with RRF (Qdrant 1.20).
Dense vectors here are random stand-ins so the script needs no model download;
swap in any real embedding model (e.g. FastEmbed) for actual semantic search."""

import numpy as np
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")
COL, DIM = "hybrid_demo", 64
rng = np.random.default_rng(1)
docs = [
    "Time Travel for Beginners",
    "The Time Machine",
    "Travel Guide to Italy",
    "A Brief History of Time",
    "Dune",
    "The Time Traveler's Wife",
]

if client.collection_exists(COL):
    client.delete_collection(COL)
client.create_collection(
    COL,
    vectors_config={
        "dense": models.VectorParams(
            size=DIM, distance=models.Distance.COSINE, datatype=models.Datatype.TURBO16
        )
    },
)
client.create_payload_index(
    COL,
    "title",
    field_schema=models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        tokenizer=models.TokenizerType.WORD,
        stemmer=models.SnowballParams(
            type=models.Snowball.SNOWBALL, language=models.SnowballLanguage.ENGLISH
        ),
        scoring=models.TextScoringParams(type=models.TextScoringType.BM25),
    ),
    wait=True,
)
client.upsert(
    COL,
    wait=True,
    points=[
        models.PointStruct(
            id=i,
            vector={"dense": rng.standard_normal(DIM).tolist()},
            payload={"title": t},
        )
        for i, t in enumerate(docs)
    ],
)

res = client.query_points(
    COL,
    prefetch=[
        models.Prefetch(
            query=models.TextQuery(text="time travel"), using="title", limit=20
        ),
        models.Prefetch(
            query=rng.standard_normal(DIM).tolist(), using="dense", limit=20
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=5,
)
for p in res.points:
    print(f"{p.score:.3f}  {p.payload['title']}")
