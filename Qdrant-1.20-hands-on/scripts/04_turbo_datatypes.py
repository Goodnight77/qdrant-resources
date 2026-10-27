"""recall@10 of float32 / float16 / turbo16 / turbo8 / turbo4 storage (Qdrant 1.20)."""

import json

import numpy as np
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333", timeout=120)
N, DIM, Q, K = 5_000, 768, 100, 10
rng = np.random.default_rng(0)


def dataset(kind):
    x = rng.standard_normal((N + Q, DIM)).astype(np.float32)
    if kind == "outliers":
        # mimic ada-002 / CLIP style embeddings: a handful of huge coordinates
        x[:, :8] *= 40.0
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x[:N], x[N:]


DTYPES = {
    "float32": models.Datatype.FLOAT32,
    "float16": models.Datatype.FLOAT16,
    "turbo16": models.Datatype.TURBO16,
    "turbo8": models.Datatype.TURBO8,
    "turbo4": models.Datatype.TURBO4,
}
BYTES = {"float32": 4, "float16": 2, "turbo16": 2, "turbo8": 1, "turbo4": 0.5}

out = {}
for kind in ["gaussian", "outliers"]:
    data, queries = dataset(kind)
    gt = np.argsort(-(queries @ data.T), axis=1)[:, :K]  # exact brute force
    for name, dt in DTYPES.items():
        col = f"dt_{kind}_{name}"
        if client.collection_exists(col):
            client.delete_collection(col)
        client.create_collection(
            col,
            vectors_config=models.VectorParams(
                size=DIM, distance=models.Distance.COSINE, datatype=dt
            ),
        )
        for s in range(0, N, 1000):
            client.upsert(
                col,
                wait=True,
                points=models.Batch(
                    ids=list(range(s, s + 1000)), vectors=data[s : s + 1000].tolist()
                ),
            )
        rec = []
        for qi, q in enumerate(queries):
            res = client.query_points(
                col,
                query=q.tolist(),
                limit=K,
                search_params=models.SearchParams(exact=True),
            )
            rec.append(len({p.id for p in res.points} & set(gt[qi])) / K)
        out[(kind, name)] = float(np.mean(rec))
        print(
            f"{kind:9s} {name:8s} {BYTES[name]:>4} B/dim  recall@10 = {out[(kind, name)]:.4f}"
        )
        client.delete_collection(col)

json.dump(
    {f"{k}|{n}": v for (k, n), v in out.items()},
    open("turbo_results.json", "w"),
    indent=1,
)
