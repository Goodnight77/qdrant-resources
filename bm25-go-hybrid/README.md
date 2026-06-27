# BM25 hybrid search in Go

A runnable demo of **dense + sparse hybrid search** against Qdrant from Go.

The sparse (BM25) vectors come from [`github.com/harsh04/bm25`](https://github.com/harsh04/bm25),
a byte-for-byte Go port of FastEmbed's `Qdrant/bm25` encoder — so the same sparse
vectors you'd get from Python FastEmbed, but produced client-side in Go. The
encoder emits term frequencies only; Qdrant applies IDF server-side via the
sparse vector's modifier. The two arms (dense + sparse) are fused with Reciprocal
Rank Fusion.

The dense embedder here is a tiny stand-in — swap `denseEmbed` for a real model
(Gemini, OpenAI, etc.) and bump `denseDim`.

## Run

```sh
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant   # any Qdrant works
go run .
```

Expected: each query prints its top hits, ranked by the fused score.

> Prefer less wiring? The [`qhybrid`](https://pkg.go.dev/github.com/harsh04/bm25/qhybrid)
> module wraps collection setup, upsert, and this RRF query behind a small API.
