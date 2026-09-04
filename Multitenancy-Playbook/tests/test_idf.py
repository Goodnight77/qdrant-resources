import math
import unittest
import warnings
from unittest.mock import Mock, patch

from qdrant_client import QdrantClient, models

from mtp.real_text_data import discover_word_pairs
from mtp.sim.per_tenant_idf import _experiment_collection, _ndcg_at_k, _relevant_count


class NdcgTests(unittest.TestCase):
    def test_missing_relevant_documents_reduce_score(self):
        expected = 1 / sum(1 / math.log2(i + 2) for i in range(10))
        self.assertAlmostEqual(_ndcg_at_k([1] + [0] * 9, 10, 20), expected)
        self.assertAlmostEqual(_ndcg_at_k([1], 10, 20), expected)

    def test_fewer_relevant_documents_than_k(self):
        self.assertEqual(_ndcg_at_k([1, 1, 0], 10, 2), 1)
        self.assertEqual(_ndcg_at_k([], 10, 0), 0)
        self.assertEqual(_ndcg_at_k([0, 0], 10, 2), 0)
        self.assertEqual(_ndcg_at_k([0, 1], 1, 1), 0)


class CollectionTests(unittest.TestCase):
    def test_tenant_relevance_and_cleanup_preserve_existing_collection(self):
        client = QdrantClient(":memory:")
        client.create_collection("mtp_per_tenant_idf", vectors_config={})
        try:
            with patch("mtp.sim.per_tenant_idf._make_client", return_value=client), \
                    patch.object(client, "close") as close, warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with self.assertRaisesRegex(RuntimeError, "query failed"):
                    with _experiment_collection() as (active, name):
                        active.upsert(name, points=[
                            models.PointStruct(
                                id=i,
                                payload={"tenant": tenant, "rare_hits": ["rare"]},
                                vector={"bm25": models.SparseVector(indices=[0], values=[1.0])},
                            )
                            for i, tenant in enumerate(["a", "a", "b"])
                        ])
                        self.assertEqual(_relevant_count(active, name, "a", "rare", "rare_hits"), 2)
                        self.assertEqual(_relevant_count(active, name, "a", "missing", "rare_hits"), 0)
                        raise RuntimeError("query failed")
                self.assertFalse(client.collection_exists(name))
                self.assertTrue(client.collection_exists("mtp_per_tenant_idf"))
                close.assert_called_once()
        finally:
            client.close()

    def test_failed_creation_does_not_delete_anything(self):
        client = Mock()
        client.create_collection.side_effect = RuntimeError("already exists")
        with patch("mtp.sim.per_tenant_idf._make_client", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                with _experiment_collection():
                    self.fail("creation should fail")
        client.delete_collection.assert_not_called()
        client.close.assert_called_once()

    def test_successive_cases_have_unique_names_and_cleanup(self):
        client = Mock()
        names = []
        with patch("mtp.sim.per_tenant_idf._make_client", return_value=client):
            for _ in range(2):
                with _experiment_collection() as (_, name):
                    names.append(name)
        self.assertNotEqual(*names)
        self.assertEqual([call.args[0] for call in client.delete_collection.call_args_list], names)


class DiscoveryTests(unittest.TestCase):
    def test_frequency_ties_choose_words_alphabetically(self):
        words = ["hotel", "golf", "foxtrot", "echo", "delta", "charlie", "bravo", "alpha"]
        sample = {
            "category": ["a"] * 100 + ["b"] * 100,
            "tokens": [words] * 10 + [[]] * 190,
        }
        pairs = discover_word_pairs(sample, ["a", "b"])
        self.assertEqual(pairs["a"]["domain_common"], sorted(words)[:6])


if __name__ == "__main__":
    unittest.main()
