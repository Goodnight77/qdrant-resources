import unittest
from unittest.mock import Mock, patch

import numpy as np

from mtp.qdrant_graph_isolation import collections, exact_topk, make_workload, measure, summarize


class BenchmarkTests(unittest.TestCase):
    def test_ground_truth_returns_global_ids(self):
        vectors = np.array([[1., 0.], [0., 1.], [-1., 0.]])
        queries = np.array([[1., 0.], [-1., 0.]])
        self.assertEqual(exact_topk(vectors, queries, np.array([12, 25, 99]), 1), [{12}, {99}])
        self.assertEqual(exact_topk(vectors, queries[:1], np.array([12, 25, 99]), 10), [{12, 25, 99}])

    def test_median_p95_and_recall(self):
        result = summarize([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4], [1, 1, 0.9, 0.9])
        self.assertEqual(result["n_queries"], 4)
        self.assertEqual(result["client_median_ms"], 2.5)
        self.assertAlmostEqual(result["client_p95_ms"], 3.85)
        self.assertAlmostEqual(result["server_median_ms"], 0.25)
        self.assertAlmostEqual(result["recall_at_10"], 0.95)

    def test_validation_and_evaluation_are_disjoint_and_tenant_scoped(self):
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(100, 4))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        queries = rng.normal(size=(100, 4))
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        tenant_ids = np.repeat([0, 1], 50)
        evaluation, validation = make_workload(vectors, queries, tenant_ids, [0, 1])
        self.assertTrue({c[1] for c in evaluation}.isdisjoint(c[1] for c in validation))
        for tenant, _, _, expected in evaluation + validation:
            self.assertEqual(len(expected), 10)
            self.assertTrue(all(tenant_ids[i] == tenant for i in expected))

    def test_failed_second_creation_cleans_only_the_first_owned_collection(self):
        client = Mock()
        client.create_collection.side_effect = [True, RuntimeError("creation failed")]
        with patch("mtp.qdrant_graph_isolation.wait_indexed", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "creation failed"):
                with collections(client, np.ones((2, 4)), np.array([0, 1]), ["shared", "tenant"]):
                    self.fail("second creation should fail")
        owned = client.create_collection.call_args_list[0].args[0]
        client.delete_collection.assert_called_once_with(owned)

    def test_failed_recall_target_is_reported_without_being_hidden(self):
        def response(client, collection, case, ef, exact=False):
            return 2.0, 0.5, 1.0 if exact or collection == "tenant" else 0.95

        case = (0, 0, [1.0], {0})
        with patch("mtp.qdrant_graph_isolation.query", side_effect=response), \
                patch("mtp.qdrant_graph_isolation.EF_CANDIDATES", [100]), \
                patch("mtp.qdrant_graph_isolation.REPEATS", 1):
            result = measure(Mock(), {"shared": "shared", "tenant": "tenant"}, [case], [case], 42)
        self.assertFalse(result["validation_target_met"])
        self.assertFalse(result["quality_comparable"])
        self.assertEqual(result["summary"]["shared"]["recall_at_10"], 0.95)
        self.assertEqual(result["validation_history"], [{"hnsw_ef": 100, "recall": {"shared": 0.95, "tenant": 1.0}}])


if __name__ == "__main__":
    unittest.main()
