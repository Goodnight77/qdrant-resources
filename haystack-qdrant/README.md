<h1 align="center">Haystack + Qdrant Integration</h1>

<p align="center">
  <img src="../assets/haystack.png" alt="Haystack logo" height="40">
  &nbsp;&nbsp;&#10133;&nbsp;&nbsp;
  <img src="../assets/qdrant_logo.svg" alt="Qdrant logo" height="40">
</p>

<p align="center">
  <a href="https://colab.research.google.com/drive/1A73k7Su85a1dSYJ4w0NuiEiKqYIS2plt">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
</p>

[`haystack_Qdrant_colab.ipynb`](haystack_Qdrant_colab.ipynb) shows how to use Qdrant Cloud as the vector database inside a Haystack pipeline. Built for Google Colab, it reads `QDRANT_URL` and `QDRANT_API_KEY` from Colab secrets.

## What it covers

- Create a cloud-backed `QdrantDocumentStore`
- Embed and index documents (`SetFit/ag_news` slice) with Haystack pipelines
- Semantic retrieval via `QdrantEmbeddingRetriever`
- Create a Qdrant payload index on `meta.label_name` for faster filtered search
- Combine vector search with metadata filters

## Run it

1. Open the notebook in Google Colab.
2. Add `QDRANT_URL` and `QDRANT_API_KEY` to Colab secrets.
3. Run cells top to bottom. The install cell pulls `haystack-ai`, `qdrant-haystack`, `sentence-transformers`, `datasets`, and `fastembed-haystack`.
