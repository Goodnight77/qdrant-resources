"""Telemetry primitives + W&B wiring. No custom metric store, log straight to W&B.

- container RSS via `docker stats` (the RAM we profile; quantized vectors sit here with always_ram).
- per-query latency -> p50/p95/p99.
- cold-start vs warm-cache loops.
One W&B run per config so the project dashboard compares them (table / parallel-coords).
"""
import subprocess
import time

import numpy as np
import wandb
from dotenv import load_dotenv

load_dotenv()  # WANDB_API_KEY from .env

CONTAINER = "qfo-qdrant"
PROJECT = "quantization-faceoff"


def _to_mb(s):
    """'1.23GiB' / '512MiB' / '900B' -> megabytes (binary units)."""
    s = s.strip()
    for unit, mult in (("GiB", 1024), ("MiB", 1), ("KiB", 1 / 1024),
                       ("GB", 1000), ("MB", 1), ("kB", 1 / 1000), ("B", 1 / 1e6)):
        if s.endswith(unit):
            return float(s[: -len(unit)]) * mult
    return float(s)  # bare number = bytes? treat as MB-less fallback


def container_rss_mb(container=CONTAINER):
    """Resident memory of the Qdrant container, in MB."""
    out = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
        capture_output=True, text=True, check=True,
    ).stdout
    return _to_mb(out.split("/")[0])  # "used / limit"


def search_latencies_ms(client, name, queries, k):
    """One query at a time (latency, not throughput). Returns ms array."""
    lats = np.empty(len(queries), dtype=np.float64)
    for i, q in enumerate(queries):
        t = time.perf_counter()
        client.query_points(collection_name=name, query=q.tolist(), limit=k)
        lats[i] = (time.perf_counter() - t) * 1000.0
    return lats


def pctiles(lats):
    return {f"latency_p{p}_ms": float(np.percentile(lats, p)) for p in (50, 95, 99)}


def profile_config(client, name, queries, k):
    """Cold (first pass) + warm (second pass) latency + RAM for one collection."""
    cold = search_latencies_ms(client, name, queries, k)
    warm = search_latencies_ms(client, name, queries, k)
    rss = container_rss_mb()
    return {
        "rss_mb": rss,
        **{f"cold_{m}": v for m, v in pctiles(cold).items()},
        **{f"warm_{m}": v for m, v in pctiles(warm).items()},
        "n_queries": len(queries),
    }


def log_run(config_name, metrics, group="dbpedia", extra_config=None):
    """One W&B run per quant config. reinit so we can loop configs in one process.
    `group` keeps the workspace tidy: group by it in the W&B UI to separate the main
    faceoff from the rescore sweeps."""
    run = wandb.init(
        project=PROJECT,
        name=config_name,
        group=group,
        config={"quant": config_name, "family": group, **(extra_config or {})},
        reinit=True,
    )
    wandb.log(metrics)
    wandb.summary.update(metrics)
    run.finish()


if __name__ == "__main__":
    # cheap unit check for the parser (the only non-trivial logic here)
    assert abs(_to_mb("1.0GiB") - 1024) < 1e-6
    assert abs(_to_mb("512MiB") - 512) < 1e-6
    assert _to_mb("900B") < 1
    print("_to_mb OK")

    # live smoke: needs collections green + valid WANDB_API_KEY
    from qdrant_client import QdrantClient
    from qfo.config import QDRANT_URL, COLLECTIONS, K
    from qfo.data.dbpedia import load_dbpedia

    _b, _p, queries = load_dbpedia()
    queries = queries[:200]  # smoke slice
    client = QdrantClient(url=QDRANT_URL, timeout=120)
    for name in COLLECTIONS:
        m = profile_config(client, name, queries, K)
        print(name, m)
        log_run(name, m)
    print("smoke done -> check W&B project:", PROJECT)
