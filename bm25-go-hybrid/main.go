// Demo: dense + sparse (BM25) hybrid search against Qdrant, in Go.
// Sparse vectors come from github.com/harsh04/bm25 (FastEmbed Qdrant/bm25 parity).
package main

import (
	"context"
	"fmt"
	"hash/fnv"
	"log"

	"github.com/harsh04/bm25"
	"github.com/qdrant/go-client/qdrant"
)

const (
	collection = "bm25_hybrid_demo"
	denseDim   = 8 // tiny for the demo; use your model's real dimension
)

func ptr[T any](v T) *T { return &v }

// denseEmbed is a stand-in. Replace with a real embedding model and keep the
// dimension/distance consistent with the collection below.
func denseEmbed(text string) []float32 {
	v := make([]float32, denseDim)
	h := fnv.New32a()
	for _, b := range []byte(text) {
		h.Write([]byte{b})
		v[h.Sum32()%denseDim] += float32(h.Sum32()%97) / 97
	}
	return v
}

func main() {
	ctx := context.Background()
	client, err := qdrant.NewClient(&qdrant.Config{Host: "127.0.0.1", Port: 6334})
	if err != nil {
		log.Fatal(err)
	}
	enc := bm25.New()

	// Collection: a dense vector + a sparse slot with the IDF modifier.
	_ = client.DeleteCollection(ctx, collection)
	if err := client.CreateCollection(ctx, &qdrant.CreateCollection{
		CollectionName: collection,
		VectorsConfig: qdrant.NewVectorsConfigMap(map[string]*qdrant.VectorParams{
			"dense": {Size: denseDim, Distance: qdrant.Distance_Cosine},
		}),
		SparseVectorsConfig: qdrant.NewSparseVectorsConfig(map[string]*qdrant.SparseVectorParams{
			"sparse": {Modifier: qdrant.Modifier_Idf.Enum()},
		}),
	}); err != nil {
		log.Fatal(err)
	}

	docs := []string{
		"Check-in is at 3 PM and check-out is at 11 AM.",
		"Free WiFi is available throughout the hotel.",
		"The swimming pool is open from 6 AM to 9 PM.",
		"Pets are welcome with a cleaning fee.",
		"Breakfast is served daily from 7 to 10.",
		"Airport shuttle runs every 30 minutes.",
	}
	var points []*qdrant.PointStruct
	for i, d := range docs {
		idx, val := enc.Encode(d) // sparse BM25 vector
		points = append(points, &qdrant.PointStruct{
			Id: qdrant.NewIDNum(uint64(i)),
			Vectors: qdrant.NewVectorsMap(map[string]*qdrant.Vector{
				"dense":  qdrant.NewVector(denseEmbed(d)...),
				"sparse": qdrant.NewVectorSparse(idx, val),
			}),
			Payload: qdrant.NewValueMap(map[string]any{"text": d}),
		})
	}
	if _, err := client.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: collection, Points: points, Wait: ptr(true),
	}); err != nil {
		log.Fatal(err)
	}

	for _, query := range []string{"when can I check in", "is there wifi"} {
		dense := denseEmbed(query)
		sIdx, sVal := enc.Encode(query)
		hits, err := client.Query(ctx, &qdrant.QueryPoints{
			CollectionName: collection,
			Prefetch: []*qdrant.PrefetchQuery{
				{Query: qdrant.NewQueryDense(dense), Using: ptr("dense"), Limit: ptr(uint64(10))},
				{Query: qdrant.NewQuerySparse(sIdx, sVal), Using: ptr("sparse"), Limit: ptr(uint64(10))},
			},
			Query:       qdrant.NewQueryFusion(qdrant.Fusion_RRF), // fuse dense + sparse
			Limit:       ptr(uint64(3)),
			WithPayload: qdrant.NewWithPayload(true),
		})
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("\nquery: %q\n", query)
		for _, h := range hits {
			fmt.Printf("  %.4f  %s\n", h.Score, h.Payload["text"].GetStringValue())
		}
	}
}
