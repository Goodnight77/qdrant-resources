"""Nested filters: regular indexes vs per-element indexes (Qdrant 1.20)."""
import random, time, statistics
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333", timeout=120)
N, SHOPS = 100_000, [f"shop_{c}" for c in "ABCDEFGHIJ"]
random.seed(42)

def make_payload():
    return {"offers": [{"shop": random.choice(SHOPS), "price": round(random.uniform(5, 500), 2)}
                       for _ in range(random.randint(1, 6))]}

payloads = [make_payload() for _ in range(N)]

def build(name, nested):
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(collection_name=name, vectors_config={})
    kw = {"nested": "offers"} if nested else {}
    client.create_payload_index(name, "offers[].shop",
        field_schema=models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, **kw), wait=True)
    client.create_payload_index(name, "offers[].price",
        field_schema=models.FloatIndexParams(type=models.FloatIndexType.FLOAT, **kw), wait=True)
    for s in range(0, N, 5000):
        client.upsert(name, wait=True, points=[models.PointStruct(id=i, vector={}, payload=payloads[i])
                                               for i in range(s, min(s + 5000, N))])

# "one single offer from shop_A that costs < $100"
nested_filter = models.Filter(must=[models.NestedCondition(nested=models.Nested(
    key="offers",
    filter=models.Filter(must=[
        models.FieldCondition(key="shop", match=models.MatchValue(value="shop_A")),
        models.FieldCondition(key="price", range=models.Range(lt=100)),
    ])))])

# ground truth computed in python, per element
truth = sum(any(o["shop"] == "shop_A" and o["price"] < 100 for o in p["offers"]) for p in payloads)
naive = sum(any(o["shop"] == "shop_A" for o in p["offers"]) and any(o["price"] < 100 for o in p["offers"])
            for p in payloads)
print(f"points: {N:,}  | true nested matches: {truth:,}  | 'any shop_A AND any <100' (wrong): {naive:,}")

results = {}
for name, nested in [("offers_regular", False), ("offers_per_element", True)]:
    build(name, nested)
    cnt = client.count(name, count_filter=nested_filter, exact=True).count
    times = []
    for _ in range(20):
        t = time.perf_counter()
        client.count(name, count_filter=nested_filter, exact=True)
        times.append((time.perf_counter() - t) * 1000)
    results[name] = statistics.median(times)
    print(f"{name:20s} count={cnt:,}  median latency={results[name]:.1f} ms")

print(f"\nspeedup: {results['offers_regular'] / results['offers_per_element']:.1f}x")
