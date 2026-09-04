import math
import unittest
import warnings
from unittest.mock import patch

import pandas as pd
from qdrant_client import QdrantClient, models

from mtp.real_text_data import tokens_to_bm25
from mtp.sim.per_tenant_idf import _experiment_collection, _index_real_sample


class Bm25Tests(unittest.TestCase):
    def test_term_frequency_saturates(self):
        vocab = {"term": 0}
        self.assertAlmostEqual(tokens_to_bm25(["term"], vocab, 1)[0], 1)
        self.assertAlmostEqual(tokens_to_bm25(["term"] * 3, vocab, 3)[0], 11 / 7)
        self.assertLess(tokens_to_bm25(["term"] * 100, vocab, 100)[0], 2.2)

    def test_length_normalization_and_disabled_normalization(self):
        vocab = {"term": 0, "filler": 1}
        short = tokens_to_bm25(["term"], vocab, 2)
        long = tokens_to_bm25(["term", "filler", "filler"], vocab, 2)
        self.assertGreater(short[0], long[0])
        self.assertEqual(tokens_to_bm25(["term"], vocab, 2, b=0)[0], 1)
        self.assertEqual(tokens_to_bm25(["term", "filler", "filler"], vocab, 2, b=0)[0], 1)
        self.assertEqual(tokens_to_bm25([], vocab, 2), {})

    def test_invalid_parameters(self):
        for params in [{"avg_len": 0}, {"avg_len": 1, "k1": 0}, {"avg_len": 1, "b": 2}]:
            with self.assertRaises(ValueError):
                tokens_to_bm25(["term"], {"term": 0}, **params)

    def test_qdrant_applies_idf_once_to_bm25_weights(self):
        sample = pd.DataFrame({
            "category": ["a", "a", "b"],
            "tokens": [["term"] * 3, ["term"], ["else"]],
        })
        with patch("mtp.sim.per_tenant_idf._make_client", side_effect=lambda: QdrantClient(":memory:")), \
                warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with _experiment_collection() as (client, collection):
                count = _index_real_sample(client, collection, sample, {"term": 0, "else": 1},
                                           {"avg_len": 5 / 3, "k1": 1.2, "b": 0.75})
                self.assertEqual(count, 3)
                tenant_filter = models.Filter(must=[models.FieldCondition(
                    key="tenant", match=models.MatchValue(value="a")
                )])
                for params, idf in [(None, math.log(1.6)),
                                    (models.SearchParams(idf=models.IdfCorpusParams(corpus=tenant_filter)), math.log(1.2))]:
                    hits = client.query_points(
                        collection, query=models.SparseVector(indices=[0], values=[1.0]),
                        using="bm25", query_filter=tenant_filter, search_params=params, limit=10,
                    ).points
                    self.assertEqual([hit.id for hit in hits], [0, 1])
                    self.assertAlmostEqual(hits[0].score, (6.6 / 4.92) * idf, places=6)
                    self.assertAlmostEqual(hits[1].score, (2.2 / 1.84) * idf, places=6)


if __name__ == "__main__":
    unittest.main()
