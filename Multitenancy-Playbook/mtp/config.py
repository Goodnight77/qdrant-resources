"""Shared knobs for the multitenancy experiments.

Everything here is sized to run on a laptop in a few minutes total. Bump
DIM/N/T up if you want production-scale numbers (see the README).
"""

import os

RESULTS_DIR = "results"
ASSETS_DIR = "assets"


def results_path(name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, name)


def asset_path(name):
    os.makedirs(ASSETS_DIR, exist_ok=True)
    return os.path.join(ASSETS_DIR, name)


SEED = 42

# ---- Experiment 1: payload partitioning vs. dedicated (is_tenant) graphs ----
# Real OpenAI ada-002 embeddings (1536-d), dbpedia-entities-openai-1M; see mtp/real_data.py.
E1_DIM = 1536
E1_N_QUERY_POOL = 2000   # held-out real documents, reused (cycled) across the sweep below
E1_TENANT_COUNTS = [5, 20, 100, 500, 3000]  # selectivity = 1 / tenant_count
E1_SAMPLE_TENANTS = 25  # tenants actually queried per tenant_count (full sweep is redundant)
E1_QUERIES_PER_TENANT = 20
E1_K = 10
E1_HNSW_M = 16
E1_EF_CONSTRUCTION = 200
E1_EF_SEARCH = 100

# ---- Experiment 2: tiered multitenancy (whale + long tail) ----
# Same real embedding pool as experiment 1, re-split whale/minnow instead of even tenants.
E2_DIM = 1536
E2_N_QUERY_POOL = 2000
E2_WHALE_FRACTION = 0.6   # whale gets this share of the loaded pool
E2_MINNOW_COUNT = 400     # the rest is split evenly across this many minnow tenants
E2_SAMPLE_MINNOWS = 25
E2_QUERIES_PER_TENANT = 20
E2_K = 10
E2_HNSW_M = 16
E2_EF_CONSTRUCTION = 200
E2_EF_SEARCH = 100

# ---- Experiment 3: per-tenant IDF statistics (Qdrant 1.19) ----
# Real data: AG News (fancyzhx/ag_news), 4 real categories as tenants.
E3_AG_NEWS_DOCS_PER_CATEGORY = 3000
E3_AG_NEWS_TOP_K = 6  # discovered (common, rare) candidate words kept per category

# Synthetic worst-case mechanism illustration (see per_tenant_idf.py docstring):
E3_TENANTS = ["fintech", "healthcare", "gaming"]
E3_DOCS_PER_TENANT = 500
E3_DOC_LEN = (12, 24)          # min/max tokens per synthetic ticket
E3_N_COMMON_WORDS = 40         # shared "stopword-ish" vocabulary
E3_N_DOMAIN_COMMON = 6         # tenant-exclusive but frequent within the tenant
E3_N_DOMAIN_RARE = 6           # tenant-exclusive and genuinely rare
E3_DOMAIN_COMMON_PROB = 0.5    # P(doc contains a given domain-common word, if not the rare one)
E3_DOMAIN_RARE_PROB = 0.05     # P(doc contains a given domain-rare word)
E3_TRIALS = 200                # query trials to average over (flagship, 3 tenants)
E3_SWEEP_TRIALS = 80           # query trials per point in the tenant-count sweep
E3_SWEEP_TENANT_COUNTS = [2, 3, 6, 10, 20]
E3_K = 10
