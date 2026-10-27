"""word tokenizer vs the new code tokenizer (Qdrant 1.20) on source-code identifiers."""
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

snippets = [
    "def getUserName(id): ...",
    "def get_user_name(id): ...",
    "GET_USER_NAME = 'user'",
    "xhr = new XMLHttpRequest()",
    "let map = HashMap::new();",
    "MAX_BUFFER_SIZE = 1_000_000",
    "digest = sha256(utf8Bytes)",
]
queries = ["user name", "get_user_name", "hashmap", "http request", "1000000"]

def build(name, tokenizer):
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(collection_name=name, vectors_config={})
    client.upsert(name, points=[models.PointStruct(id=i, vector={}, payload={"code": s})
                                for i, s in enumerate(snippets)])
    client.create_payload_index(
        name, field_name="code",
        field_schema=models.TextIndexParams(
            type=models.TextIndexType.TEXT, tokenizer=tokenizer, lowercase=True,
            scoring=models.TextScoringParams(type=models.TextScoringType.BM25)),
        wait=True)

build("code_word", models.TokenizerType.WORD)
build("code_code", models.TokenizerType.CODE)   # <-- the 1.20 tokenizer

for q in queries:
    print(f"\nquery: {q!r}")
    for col in ["code_word", "code_code"]:
        res = client.query_points(col, query=models.TextQuery(text=q), using="code", limit=7)
        hits = [p.payload["code"] for p in res.points]
        print(f"  {col:10s} -> {len(hits)} hit(s): {hits}")
