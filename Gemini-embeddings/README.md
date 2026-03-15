# One model to find them all : Multimodal Semantic Search with Gemini Embedding 2 + Qdrant

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1srnOdLTjJaoHNYPlwX3us8l86Xna2MJo?usp=sharing)

This folder contains resources and a tutorial for implementing multimodal semantic search using **Google's Gemini Embedding 2** model and **Qdrant** as the vector database.

> **[Read the article](https://mohamedarbi.xyz/posts/gemini2qdrant)** for a deep dive into the code and Gemini embeddings.

## Overview

`gemini-embedding-2-preview` is Google's first fully multimodal embedding model, capable of mapping text, images, audio, and PDFs into a **single unified vector space**. This enables seamless cross-modal search without the need for OCR pipelines or intermediate transcription steps.

## Setup

Go to [Google AI Studio](https://aistudio.google.com/api-keys), generate a Google API key, and save it as `GOOGLE_API_KEY` environment variable.

## What This Tutorial Covers

This project demonstrates the following capabilities end-to-end:

1.  **Image Retrieval**: Indexing research paper figures (image + caption) and querying by concept.
3.  **Image Embedding Proof**: Experiment proving the model reads image bytes, not just captions.
4.  **PDF RAG**: Indexing PDF pages as images and retrieving relevant pages via text query.
5.  **Audio RAG**: Chunking audio, embedding segments, and retrieving by semantic query.

## Key Concepts

-   **Unified Vector Space**: Text queries can retrieve images, audio segments, and PDF pages from the same index.
-   **MRL (Matryoshka Representation Learning)**: Vectors can be truncated with minimal quality loss for memory efficiency. For a deeper dive, **[read the full MRL post](https://mohamedarbi.xyz/posts/matryoshka)**.
-   **Task Types**: Utilizing `RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY` to optimize the embedding manifold.

### Author 
- [Mohamed Arbi Nsibi](https://www.linkedin.com/in/mohammed-arbi-nsibi-584a43241/)
