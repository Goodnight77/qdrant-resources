"""ID tracker memory tier + bits8 TurboQuant quantization (Qdrant 1.20)."""

from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")
COL = "tiers_demo"
if client.collection_exists(COL):
    client.delete_collection(COL)

client.create_collection(
    collection_name=COL,
    vectors_config=models.VectorParams(
        size=1024, distance=models.Distance.COSINE, memory=models.Memory.CACHED
    ),
    # 1.20: the ID tracker gets a memory tier too (default: pinned)
    id_tracker=models.IdTrackerParams(memory=models.Memory.COLD),
    # 1.20: 8-bit TurboQuant quantization, ~4x compression (bits4 stays the default)
    quantization_config=models.TurboQuantization(
        turbo=models.TurboQuantQuantizationConfig(
            bits=models.TurboQuantBitSize.BITS8,
            memory=models.Memory.PINNED,
        )
    ),
)

cfg = client.get_collection(COL).config
print(
    "id_tracker  :",
    cfg.params.model_dump().get("id_tracker") or getattr(cfg, "id_tracker", None),
)
print("quantization:", cfg.quantization_config)
