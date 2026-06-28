# BM25 hybrid search in Go

A runnable demo of **dense + sparse hybrid search** against Qdrant from Go.

- **Sparse (BM25)** vectors come from
  [`github.com/harsh04/bm25`](https://github.com/harsh04/bm25), a byte-for-byte Go
  port of FastEmbed's `Qdrant/bm25` encoder — the same sparse vectors you'd get
  from Python FastEmbed, produced client-side in Go. It emits term frequencies
  only; Qdrant applies IDF server-side via the sparse vector's modifier.
- **Dense** vectors come from Google's Gemini embedding API (`gemini-embedding-001`).
- The two arms are fused with **Reciprocal Rank Fusion (RRF)** by Qdrant.

## Run

```sh
export GEMINI_API_KEY=...                              # https://aistudio.google.com/apikey
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant    # any Qdrant works
go run .
```

## Expected output

The first query overlaps the docs word-for-word, so BM25 alone would find it. The
other two share **no words** with the doc they should match — the dense arm is
what surfaces them. That's the whole point of hybrid:

```
query: "when can I check in"
  1.0000  Check-in is at 3 PM and check-out is at 11 AM.
  0.3333  Breakfast is served daily from 7 to 10.
  0.2500  The swimming pool is open from 6 AM to 9 PM.

query: "what time should I arrive"
  0.5000  Check-in is at 3 PM and check-out is at 11 AM.
  0.3333  The swimming pool is open from 6 AM to 9 PM.
  0.2500  Breakfast is served daily from 7 to 10.

query: "somewhere to eat early"
  0.5000  Breakfast is served daily from 7 to 10.
  0.3333  Check-in is at 3 PM and check-out is at 11 AM.
  0.2500  The swimming pool is open from 6 AM to 9 PM.
```

(Scores are RRF ranks, so they're stable; the top hit for each query is the
semantically correct doc.)

> Prefer less wiring? The [`qhybrid`](https://pkg.go.dev/github.com/harsh04/bm25/qhybrid)
> module wraps collection setup, upsert, and this RRF query behind a small API.
