// Demo: dense + sparse (BM25) hybrid search against Qdrant, in Go.
//
// Sparse vectors come from github.com/harsh04/bm25 (FastEmbed Qdrant/bm25
// parity). Dense vectors come from Google's Gemini embedding API. The two arms
// are fused with Reciprocal Rank Fusion (RRF) server-side by Qdrant.
//
// Run:
//
//	export GEMINI_API_KEY=...          # https://aistudio.google.com/apikey
//	docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
//	go run .
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"

	"github.com/harsh04/bm25"
	"github.com/qdrant/go-client/qdrant"
)

const (
	collection = "bm25_go_hybrid_demo"
	embedModel = "gemini-embedding-001"
	denseDim   = 1536 // Gemini embedding output dimensionality
)

func ptr[T any](v T) *T { return &v }

// denseEmbed returns a real semantic embedding from the Gemini API. Truncated
// (sub-3072) outputs are L2-normalized, as Google recommends, so cosine behaves.
func denseEmbed(ctx context.Context, text string) ([]float32, error) {
	key := os.Getenv("GEMINI_API_KEY")
	if key == "" {
		return nil, fmt.Errorf("set GEMINI_API_KEY (https://aistudio.google.com/apikey)")
	}
	reqBody, _ := json.Marshal(map[string]any{
		"content":              map[string]any{"parts": []map[string]string{{"text": text}}},
		"outputDimensionality": denseDim,
	})
	url := "https://generativelanguage.googleapis.com/v1beta/models/" + embedModel + ":embedContent"
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", key) // header, not URL, so it can't leak into errors
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("gemini %d: %s", resp.StatusCode, b)
	}
	var out struct {
		Embedding struct {
			Values []float32 `json:"values"`
		} `json:"embedding"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return l2norm(out.Embedding.Values), nil
}

func l2norm(v []float32) []float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return v
	}
	n := float32(math.Sqrt(sum))
	for i := range v {
		v[i] /= n
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
		"The airport shuttle runs every 30 minutes.",
	}
	var points []*qdrant.PointStruct
	for i, d := range docs {
		dense, err := denseEmbed(ctx, d)
		if err != nil {
			log.Fatal(err)
		}
		idx, val := enc.Encode(d) // sparse BM25 vector
		points = append(points, &qdrant.PointStruct{
			Id: qdrant.NewIDNum(uint64(i)),
			Vectors: qdrant.NewVectorsMap(map[string]*qdrant.Vector{
				"dense":  qdrant.NewVector(dense...),
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

	// The first query shares words with a doc, so BM25 alone finds it. The other
	// two use different words from the docs they should match, so the dense arm
	// is what surfaces them — that's the point of hybrid.
	queries := []string{
		"when can I check in",       // lexical overlap -> sparse (BM25) finds it
		"what time should I arrive", // same meaning, no shared words -> dense arm
		"somewhere to eat early",    // -> breakfast, purely semantic -> dense arm
	}
	for _, query := range queries {
		dense, err := denseEmbed(ctx, query)
		if err != nil {
			log.Fatal(err)
		}
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
