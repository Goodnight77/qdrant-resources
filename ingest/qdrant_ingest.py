"""Simple reusable ingestion pipeline for chunking, embedding, and upsert."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
from qdrant_client.models import Distance, PointStruct, VectorParams


Payload = Dict[str, Any]
PayloadBuilder = Callable[[Dict[str, Any], int], Payload]


@dataclass
class IngestStats:
    chunks_created: int
    points_upserted: int


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks with predictable boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if not text:
        return []

    chunks: List[str] = []
    step = chunk_size - chunk_overlap
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


class QdrantIngestor:
    """Reusable helper that creates a collection and ingests text records."""

    def __init__(
        self,
        qdrant_client: Any,
        embedding_model: Any,
        collection_name: str,
        vector_size: int,
        *,
        distance: Distance = Distance.COSINE,
        batch_size: int = 100,
        recreate_collection: bool = False,
    ) -> None:
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        self.batch_size = batch_size
        self.recreate_collection = recreate_collection

    def prepare_collection(self) -> None:
        """Create (or recreate) the destination collection."""
        if self.recreate_collection:
            try:
                self.qdrant_client.delete_collection(collection_name=self.collection_name)
            except Exception:
                # Ignore errors when collection does not exist.
                pass

        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=self.distance),
        )

    def ingest_records(
        self,
        records: Sequence[Dict[str, Any]],
        *,
        text_key: str,
        payload_builder: Optional[PayloadBuilder] = None,
        start_id: int = 1,
        normalize_embeddings: bool = True,
    ) -> IngestStats:
        """
        Embed and upsert record batches.

        Each record must contain a `text_key` field with the text to embed.
        """
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.prepare_collection()
        total_points = 0

        for batch_start in range(0, len(records), self.batch_size):
            batch_records = records[batch_start : batch_start + self.batch_size]
            batch_texts = [record[text_key] for record in batch_records]
            embeddings = self._encode_texts(batch_texts, normalize_embeddings=normalize_embeddings)

            points: List[PointStruct] = []
            for offset, (record, vector) in enumerate(zip(batch_records, embeddings)):
                point_id = start_id + batch_start + offset
                if payload_builder:
                    payload = payload_builder(record, point_id)
                else:
                    payload = {k: v for k, v in record.items() if k != text_key}
                    payload[text_key] = record[text_key]

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                )

            if points:
                self.qdrant_client.upsert(collection_name=self.collection_name, points=points)
                total_points += len(points)

        return IngestStats(chunks_created=len(records), points_upserted=total_points)

    def _encode_texts(self, texts: Iterable[str], *, normalize_embeddings: bool) -> List[List[float]]:
        text_list = list(texts)
        if hasattr(self.embedding_model, "encode"):
            vectors = self.embedding_model.encode(
                text_list,
                normalize_embeddings=normalize_embeddings,
            )
        elif callable(self.embedding_model):
            vectors = self.embedding_model(text_list)
        else:
            raise TypeError(
                "embedding_model must provide .encode(...) or be a callable that accepts a list of strings."
            )

        array = np.asarray(vectors)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array.tolist()
