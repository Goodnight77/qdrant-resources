"""Experiment 1 on a Qdrant server: indexed shared vs. tenant-optimized search.

Both layouts keep native payload indexes and planner-selected scans enabled.
Timings include warm-up, three randomized paired passes, and recall checks
against offline exact cosine search. Only uniquely named test collections
are created or deleted. QDRANT_URL defaults to the local Docker server.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from uuid import uuid4

import numpy as np
from qdrant_client import QdrantClient, models

from mtp import config
from mtp.real_data import load_dbpedia_vectors


LAYOUTS = {
    "shared": {"m": 16, "payload_m": 16, "is_tenant": False},
    "tenant": {"m": 0, "payload_m": 16, "is_tenant": True},
}
REPEATS = 3
RECALL_TARGET = 0.99
EF_CANDIDATES = [100, 200, 400, 800, 1600, 3200, 6400]
INDEXING_THRESHOLD_KB = 1000
FULL_SCAN_THRESHOLD_KB = 10000


def summarize(client_ms, server_ms, recalls):
    return {
        "n_queries": len(recalls),
        "client_median_ms": float(np.median(client_ms)),
        "client_p95_ms": float(np.percentile(client_ms, 95)),
        "server_median_ms": float(np.median(server_ms)),
        "server_p95_ms": float(np.percentile(server_ms, 95)),
        "recall_at_10": float(np.mean(recalls)),
    }


def exact_topk(vectors, queries, ids, k):
    """Inputs are cosine-normalized; return global IDs for each query."""
    scores = queries @ vectors.T
    count = min(k, len(ids))
    selected = np.argpartition(-scores, count - 1, axis=1)[:, :count]
    return [set(int(i) for i in ids[row]) for row in selected]


def make_workload(vectors, query_pool, tenant_ids, sample_tenants):
    """Disjoint held-out queries for ef validation and measured evaluation."""
    evaluation, validation = [], []
    cursor = 0
    for tenant_number, tenant in enumerate(sample_tenants):
        ids = np.flatnonzero(tenant_ids == tenant)
        if len(ids) < config.E1_K:
            continue
        eval_ids = np.arange(cursor, cursor + config.E1_QUERIES_PER_TENANT)
        validation_ids = np.arange(len(query_pool) - 5 * (tenant_number + 1),
                                   len(query_pool) - 5 * tenant_number)
        cursor += config.E1_QUERIES_PER_TENANT
        if eval_ids[-1] >= len(query_pool) - 5 * len(sample_tenants):
            raise ValueError("Query pool must fit disjoint evaluation and validation sets")
        for destination, query_ids in [(evaluation, eval_ids), (validation, validation_ids)]:
            ground_truth = exact_topk(vectors[ids], query_pool[query_ids], ids, config.E1_K)
            for query_id, expected in zip(query_ids, ground_truth):
                destination.append((int(tenant), int(query_id), query_pool[query_id].tolist(), expected))
    if not evaluation:
        raise ValueError("No sampled tenant has enough documents for recall@10")
    return evaluation, validation


def wait_indexed(client, collection, n_points, timeout=1800):
    deadline = time.monotonic() + timeout
    last_progress = 0
    stable = 0
    while time.monotonic() < deadline:
        info = client.get_collection(collection)
        if info.optimizer_status != "ok":
            raise RuntimeError(f"Optimizer error for {collection}: {info.optimizer_status}")
        # Small appendable segments below the index threshold may stay unindexed.
        allowance = max(1, int(INDEXING_THRESHOLD_KB * 1024 / (config.E1_DIM * 4))) * info.segments_count
        ready = (info.status == models.CollectionStatus.GREEN and info.points_count == n_points
                 and (info.indexed_vectors_count or 0) >= n_points - allowance)
        stable = stable + 1 if ready else 0
        if stable >= 3:
            return {
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "segments_count": info.segments_count,
                "status": info.status.value,
                "config": info.config.model_dump(mode="json"),
                "payload_schema": {k: v.model_dump(mode="json") for k, v in info.payload_schema.items()},
            }
        if time.monotonic() - last_progress >= 30:
            print(f"  indexing: {info.indexed_vectors_count}/{n_points}, status={info.status}", flush=True)
            last_progress = time.monotonic()
        time.sleep(2)
    raise TimeoutError(f"Indexing did not settle for {collection}")


@contextmanager
def collections(client, vectors, tenant_ids, build_order):
    names = {}
    readiness = {}
    try:
        for layout in build_order:
            settings = LAYOUTS[layout]
            name = f"mtp_e1_{layout}_{uuid4().hex}"
            print(f"  building {layout}: {len(vectors)} vectors", flush=True)
            started = time.perf_counter()
            client.create_collection(
                name, vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE),
                shard_number=1, on_disk_payload=False,
                hnsw_config=models.HnswConfigDiff(
                    m=settings["m"], payload_m=settings["payload_m"], ef_construct=config.E1_EF_CONSTRUCTION,
                    full_scan_threshold=FULL_SCAN_THRESHOLD_KB, max_indexing_threads=4,
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0, default_segment_number=2, max_optimization_threads=1,
                ),
            )
            names[layout] = name
            client.create_payload_index(name, "tenant", field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD, is_tenant=settings["is_tenant"]), wait=True)
            for start in range(0, len(vectors), 256):
                end = min(start + 256, len(vectors))
                client.upsert(name, models.Batch(
                    ids=list(range(start, end)), vectors=vectors[start:end].tolist(),
                    payloads=[{"tenant": str(int(t))} for t in tenant_ids[start:end]],
                ), wait=True)
            client.update_collection(name, optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=INDEXING_THRESHOLD_KB))
            readiness[layout] = wait_indexed(client, name, len(vectors))
            readiness[layout]["build_seconds"] = time.perf_counter() - started
        yield names, readiness
    finally:
        for name in names.values():
            client.delete_collection(name)


def query(client, collection, case, ef, exact=False, shard_key=None):
    tenant, _, vector, expected = case
    request = models.QueryRequest(
        query=vector, filter=models.Filter(must=[models.FieldCondition(
            key="tenant", match=models.MatchValue(value=str(tenant)))]),
        params=models.SearchParams(hnsw_ef=ef, exact=exact),
        limit=len(expected), with_payload=False, with_vector=False, shard_key=shard_key,
    )
    start = time.perf_counter_ns()
    response = client.http.search_api.query_points(collection_name=collection, query_request=request)
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    hits = response.result.points
    recall = len({hit.id for hit in hits} & expected) / len(expected)
    return elapsed_ms, response.time * 1000, recall


def measure(client, names, evaluation, validation, seed, query_fn=None):
    run_query = query_fn or query
    exact_validation_recall = float(np.mean([
        run_query(client, names["shared"], case, EF_CANDIDATES[0], exact=True)[2] for case in validation
    ]))
    if exact_validation_recall < RECALL_TARGET:
        raise RuntimeError(f"Server exact search disagrees with offline ground truth: {exact_validation_recall}")
    # Same ef for both layouts; validation queries are excluded from timing results.
    validation_history = []
    for ef in EF_CANDIDATES:
        validation_recall = {
            layout: float(np.mean([run_query(client, name, case, ef)[2] for case in validation]))
            for layout, name in names.items()
        }
        validation_history.append({"hnsw_ef": ef, "recall": validation_recall})
        print(f"  validation ef={ef}: {validation_recall}", flush=True)
        if min(validation_recall.values()) >= RECALL_TARGET:
            break
    validation_target_met = min(validation_recall.values()) >= RECALL_TARGET
    if not validation_target_met:
        print("  Recall target not met; retain the failure and time the highest tested ef.", flush=True)
    print(f"  ef={ef}, validation recall={validation_recall}", flush=True)
    rng = np.random.default_rng(seed)
    # One complete unmeasured pass warms both configurations on the timed workload.
    for index in rng.permutation(len(evaluation)):
        for layout in rng.permutation(list(names)):
            run_query(client, names[layout], evaluation[index], ef)
    records = {layout: [] for layout in names}
    rounds = []
    for repeat in range(REPEATS):
        current = {layout: [] for layout in names}
        for index in rng.permutation(len(evaluation)):
            for layout in rng.permutation(list(names)):
                values = run_query(client, names[layout], evaluation[index], ef)
                current[layout].append(values)
                records[layout].append({"repeat": repeat + 1, "tenant": evaluation[index][0],
                                        "query_id": evaluation[index][1], "client_ms": values[0],
                                        "server_ms": values[1], "recall_at_10": values[2]})
        rounds.append({layout: summarize(*zip(*values)) for layout, values in current.items()})
        print(f"  repeat {repeat + 1}/{REPEATS}: " + str(rounds[-1]), flush=True)
    summary = {layout: summarize([r["client_ms"] for r in values], [r["server_ms"] for r in values],
                                [r["recall_at_10"] for r in values]) for layout, values in records.items()}
    return {"hnsw_ef": ef, "validation_recall": validation_recall,
            "exact_validation_recall": exact_validation_recall,
            "validation_history": validation_history,
            "validation_target_met": validation_target_met,
            "quality_comparable": validation_target_met and min(r["recall_at_10"] for r in summary.values()) >= RECALL_TARGET,
            "validation_queries": len(validation), "unique_evaluation_queries": len(evaluation),
            "summary": summary, "repeats": rounds, "measurements": records}


def write_results(path, result):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
    temporary.replace(path)


def run(output):
    client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"),
                          api_key=os.environ.get("QDRANT_API_KEY") or None, timeout=120)
    try:
        server = client.info().model_dump()
        print("Loading real embeddings ...", flush=True)
        vectors, query_pool = load_dbpedia_vectors(config.E1_N_QUERY_POOL, config.SEED)
        digest = hashlib.sha256(vectors.tobytes())
        digest.update(query_pool.tobytes())
        fingerprint = digest.hexdigest()
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        query_pool /= np.linalg.norm(query_pool, axis=1, keepdims=True)
        environment = {"client_os": platform.platform(), "python": platform.python_version(),
                       "client_cpu": platform.processor(), "client_logical_cpus": os.cpu_count(),
                       "qdrant_client": version("qdrant-client"), "server": server}
        # Optional machine metadata; no dependency on Docker for remote-server runs.
        try:
            docker = json.loads(subprocess.check_output(["docker", "info", "--format", "{{json .}}"], text=True))
            environment["local_docker"] = {k: docker[k] for k in ["NCPU", "MemTotal", "OperatingSystem", "Architecture"]}
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        result = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "complete": False,
                  "environment": environment, "corpus_sha256": fingerprint, "n_total": len(vectors),
                  "dim": vectors.shape[1], "query_pool_size": len(query_pool), "seed": config.SEED,
                  "layouts": LAYOUTS, "repeats": REPEATS, "recall_target": RECALL_TARGET,
                  "latency_scope": "serial warm REST queries; client round trip and server-reported processing time",
                  "rows": []}
        rng = np.random.default_rng(config.SEED)
        for n_tenants in config.E1_TENANT_COUNTS:
            tenant_ids = rng.integers(0, n_tenants, len(vectors))
            sample = rng.choice(n_tenants, size=min(config.E1_SAMPLE_TENANTS, n_tenants), replace=False)
            evaluation, validation = make_workload(vectors, query_pool, tenant_ids, sample)
            print(f"Tenants={n_tenants}, measured query pairs={len(evaluation)}", flush=True)
            build_order = ["shared", "tenant"] if len(result["rows"]) % 2 == 0 else ["tenant", "shared"]
            with collections(client, vectors, tenant_ids, build_order) as (names, readiness):
                measured = measure(client, names, evaluation, validation, config.SEED + n_tenants)
                # Confirm the optimizer stayed settled during the measurement window.
                for name in names.values():
                    if client.get_collection(name).status != models.CollectionStatus.GREEN:
                        raise RuntimeError("Optimizer became active during timing; discard this measurement")
            result["rows"].append({"n_tenants": n_tenants, "selectivity_pct": 100 / n_tenants,
                                   "tenant_assignment_sha256": hashlib.sha256(tenant_ids.tobytes()).hexdigest(),
                                   "sampled_tenants": [int(t) for t in sample], "indexes": readiness, **measured})
            write_results(output, result)
        result["complete"] = True
        write_results(output, result)
        return result
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(config.results_path("exp1_qdrant_graph_isolation.json")))
    run(parser.parse_args().output)
