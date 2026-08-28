"""Pattern A: FastEmbed via qdrant-client's built-in local inference.

Everything here runs locally: passing a `models.Document` (raw text + a
model name) makes qdrant-client embed it in-process with FastEmbed before
sending vectors to Qdrant. The only network call this script makes is to
Qdrant itself - no embedding-provider API key is used or required.

Verified against qdrant-client==1.19.0 / fastembed==0.8.0.

Run against a local Qdrant instance:
    docker run -p 6333:6333 qdrant/qdrant
    pip install -r requirements.txt
    python ingest_and_query.py

Or run entirely in-process with no server at all, by pointing
QdrantClient at ":memory:" instead of a URL.
"""

from qdrant_client import QdrantClient, models

COLLECTION_NAME = "docs_local_embeddings"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # small CPU-friendly dense model

DOCUMENTS = [
    "Qdrant is a vector database written in Rust.",
    "FastEmbed generates embeddings locally using ONNX Runtime.",
    "Running embedding models next to your database avoids external API calls.",
    "A sidecar container can serve embeddings without any public network access.",
]


def main() -> None:
    client = QdrantClient(url="http://localhost:6333")

    vector_size = client.get_embedding_size(EMBEDDING_MODEL)
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
        )

    # Each point's vector is a Document (raw text + model name). qdrant-client
    # embeds it locally with FastEmbed before the point is sent to Qdrant -
    # no separate embedding call, no API key.
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=i,
                vector=models.Document(text=doc, model=EMBEDDING_MODEL),
                payload={"source": "readme", "text": doc},
            )
            for i, doc in enumerate(DOCUMENTS)
        ],
    )

    query = "How do I avoid calling an external embedding API?"
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=models.Document(text=query, model=EMBEDDING_MODEL),
        limit=2,
    )

    print(f"Query: {query!r}\n")
    for point in response.points:
        print(f"  score={point.score:.4f}  text={point.payload['text']!r}")


if __name__ == "__main__":
    main()
