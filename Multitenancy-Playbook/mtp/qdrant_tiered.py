"""Experiment 2: actual Qdrant shared vs. dedicated-whale/fallback shards.

Both layouts use native tenant indexes and identical graph settings. This
measures steady-state serial search, not live promotion or multi-node isolation.
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
from mtp.qdrant_graph_isolation import (
    FULL_SCAN_THRESHOLD_KB, INDEXING_THRESHOLD_KB, RECALL_TARGET, REPEATS,
    exact_topk, measure, query, wait_indexed, write_results,
)
from mtp.real_data import load_dbpedia_vectors


def tenant_filter(tenant):
    return models.Filter(must=[models.FieldCondition(
        key="tenant", match=models.MatchValue(value=str(tenant)))])


def routed_query(client, collection, case, ef, exact=False):
    return query(client, collection, case, ef, exact=exact,
                 shard_key=models.ShardKeyWithFallback(target=str(case[0]), fallback="fallback"))


def make_workloads(vectors, query_pool, tenant_ids, sample_minnows):
    """Whale: 60 evaluation/15 validation queries; minnows: 20/5 per tenant."""
    workloads = {group: {"evaluation": [], "validation": []} for group in ["minnows", "whale"]}
    eval_cursor, validation_cursor = 0, len(query_pool)
    for tenant in [0, *sample_minnows]:
        group = "whale" if tenant == 0 else "minnows"
        n_eval = config.E2_QUERIES_PER_TENANT * (3 if tenant == 0 else 1)
        n_validation = 15 if tenant == 0 else 5
        ids = np.flatnonzero(tenant_ids == tenant)
        if len(ids) < config.E2_K:
            raise ValueError("Every sampled tenant must have at least k documents")
        split_ids = {
            "evaluation": np.arange(eval_cursor, eval_cursor + n_eval),
            "validation": np.arange(validation_cursor - n_validation, validation_cursor),
        }
        eval_cursor += n_eval
        validation_cursor -= n_validation
        if eval_cursor > validation_cursor:
            raise ValueError("Query pool is too small for disjoint validation and evaluation")
        for split, query_ids in split_ids.items():
            truth = exact_topk(vectors[ids], query_pool[query_ids], ids, config.E2_K)
            workloads[group][split].extend(
                (int(tenant), int(qid), query_pool[qid].tolist(), expected)
                for qid, expected in zip(query_ids, truth))
    return workloads


def check_routing(client, name, layout, tenant_ids):
    """Verify exact shard contents and tenant routing before timing."""
    whale_n = int(np.sum(tenant_ids == 0))
    expected = {"fallback": len(tenant_ids)} if layout == "shared" else {
        "fallback": len(tenant_ids) - whale_n, "0": whale_n}
    counts = {key: client.count(name, exact=True, shard_key_selector=key).count for key in expected}
    if counts != expected:
        raise RuntimeError(f"Wrong shard populations: {counts}, expected {expected}")
    for tenant in np.unique(tenant_ids):
        actual = client.count(name, count_filter=tenant_filter(tenant), exact=True,
                              shard_key_selector=models.ShardKeyWithFallback(
                                  target=str(tenant), fallback="fallback")).count
        if actual != int(np.sum(tenant_ids == tenant)):
            raise RuntimeError(f"Wrong routed population for tenant {tenant}: {actual}")
    if layout == "tiered":
        leaked = client.count(name, count_filter=tenant_filter(0), exact=True,
                              shard_key_selector="fallback").count
        if leaked:
            raise RuntimeError("Whale points leaked into fallback shard")
    return counts


@contextmanager
def collections(client, vectors, tenant_ids):
    names, readiness = {}, {}
    try:
        for layout in ["shared", "tiered"]:
            name = f"mtp_e2_{layout}_{uuid4().hex}"
            started = time.perf_counter()
            print(f"Building {layout}: {len(vectors)} vectors", flush=True)
            client.create_collection(
                name, vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE),
                sharding_method=models.ShardingMethod.CUSTOM, shard_number=1, replication_factor=1,
                on_disk_payload=False,
                hnsw_config=models.HnswConfigDiff(m=0, payload_m=16,
                    ef_construct=config.E2_EF_CONSTRUCTION, full_scan_threshold=FULL_SCAN_THRESHOLD_KB,
                    max_indexing_threads=4),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0,
                    default_segment_number=2, max_optimization_threads=1),
            )
            names[layout] = name
            client.create_shard_key(name, "fallback", shards_number=1, replication_factor=1)
            if layout == "tiered":
                client.create_shard_key(name, "0", shards_number=1, replication_factor=1)
            client.create_payload_index(name, "tenant", field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD, is_tenant=True), wait=True)
            # Batches contain only one tenant, so fallback routing is unambiguous.
            for tenant in np.unique(tenant_ids):
                ids = np.flatnonzero(tenant_ids == tenant)
                for start in range(0, len(ids), 256):
                    batch_ids = ids[start:start + 256]
                    client.upsert(name, models.Batch(ids=batch_ids.tolist(),
                        vectors=vectors[batch_ids].tolist(),
                        payloads=[{"tenant": str(tenant)} for _ in batch_ids]), wait=True,
                        shard_key_selector=models.ShardKeyWithFallback(target=str(tenant), fallback="fallback"))
            client.update_collection(name, optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=INDEXING_THRESHOLD_KB))
            readiness[layout] = wait_indexed(client, name, len(vectors))
            readiness[layout]["shard_counts"] = check_routing(client, name, layout, tenant_ids)
            readiness[layout]["cluster"] = client.collection_cluster_info(name).model_dump(mode="json")
            readiness[layout]["build_and_verify_seconds"] = time.perf_counter() - started
        yield names, readiness
    finally:
        for name in names.values():
            client.delete_collection(name)


def run(output):
    client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"),
                          api_key=os.environ.get("QDRANT_API_KEY") or None, timeout=120)
    try:
        server = client.info().model_dump()
        print("Loading real embeddings ...", flush=True)
        vectors, query_pool = load_dbpedia_vectors(config.E2_N_QUERY_POOL, config.SEED)
        digest = hashlib.sha256(vectors.tobytes())
        digest.update(query_pool.tobytes())
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        query_pool /= np.linalg.norm(query_pool, axis=1, keepdims=True)
        minnow_n = (len(vectors) - int(len(vectors) * config.E2_WHALE_FRACTION)) // config.E2_MINNOW_COUNT
        whale_n = len(vectors) - minnow_n * config.E2_MINNOW_COUNT
        tenant_ids = np.concatenate([np.zeros(whale_n, dtype=np.int64),
            np.repeat(np.arange(1, config.E2_MINNOW_COUNT + 1, dtype=np.int64), minnow_n)])
        rng = np.random.default_rng(config.SEED)
        sample = rng.choice(np.arange(1, config.E2_MINNOW_COUNT + 1), config.E2_SAMPLE_MINNOWS, replace=False)
        workloads = make_workloads(vectors, query_pool, tenant_ids, sample)
        environment = {"server": server, "qdrant_client": version("qdrant-client"),
            "client_os": platform.platform(), "python": platform.python_version(),
            "client_cpu": platform.processor(), "client_logical_cpus": os.cpu_count()}
        try:
            docker = json.loads(subprocess.check_output(["docker", "info", "--format", "{{json .}}"], text=True))
            environment["local_docker"] = {key: docker[key] for key in ["NCPU", "MemTotal", "OperatingSystem", "Architecture"]}
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        result = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "complete": False,
            "environment": environment, "corpus_sha256": digest.hexdigest(),
            "tenant_assignment_sha256": hashlib.sha256(tenant_ids.tobytes()).hexdigest(),
            "n_total": len(vectors), "dim": vectors.shape[1], "seed": config.SEED,
            "query_pool_size": len(query_pool), "whale_n": whale_n,
            "minnow_count": config.E2_MINNOW_COUNT, "minnow_n_each": minnow_n,
            "sampled_minnows": sample.tolist(), "repeats": REPEATS, "recall_target": RECALL_TARGET,
            "latency_scope": "serial warm REST queries; steady-state shard layouts on one node",
            "groups": {}}
        with collections(client, vectors, tenant_ids) as (names, readiness):
            result["indexes"] = readiness
            for group, workload in workloads.items():
                print(f"Measuring {group}", flush=True)
                result["groups"][group] = measure(client, names, workload["evaluation"],
                    workload["validation"], config.SEED, query_fn=routed_query)
                write_results(output, result)
            for name in names.values():
                if client.get_collection(name).status != models.CollectionStatus.GREEN:
                    raise RuntimeError("Optimizer became active during timing; discard measurement")
        result["complete"] = True
        write_results(output, result)
        return result
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(config.results_path("exp2_qdrant_tiered.json")))
    run(parser.parse_args().output)
