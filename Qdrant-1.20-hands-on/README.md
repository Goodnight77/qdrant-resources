# Qdrant 1.20 hands-on

Companion code for the blog post [**"Qdrant 1.20: BM25 Without Sparse Vectors, Turbo16 & Turbo8"**](https://mohamedarbi.xyz/posts/qdrant-120).
Every script was run against Qdrant 1.20.0 with qdrant-client 1.20.0.

```bash
docker compose up -d
pip install -r requirements.txt
python scripts/01_bm25_fulltext.py
```

| script | what it shows | result on my run |
|---|---|---|
| `01_bm25_fulltext.py` | BM25 on a full-text index, per-query `k`/`b` | ranked titles, no vectors at all |
| `02_code_tokenizer.py` | `word` vs `code` tokenizer | `user name` → 2 vs 3 hits, `http request` → 0 vs 1 |
| `03_nested_filter.py` | regular vs per-element indexes, 100k points | 57.8 ms → 8.2 ms (7.0x), same 6,424 matches |
| `04_turbo_datatypes.py` | recall@10 per datatype, 5k × 768d | turbo16 1.000 · turbo8 0.987/0.999 · turbo4 0.863/0.976 |
| `05_tiers_and_bits8.py` | ID tracker tier + bits8 quantization | config read back from server |
| `06_hybrid_bm25_dense.py` | TextQuery + dense prefetch, RRF fusion | fused top-5 |

Notes: script 04 uses synthetic vectors (random gaussian is the worst case for any
quantizer). Script 06 uses random dense vectors so it runs without a model download.
Always benchmark on your own embeddings.
