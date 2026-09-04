import unittest
from unittest.mock import Mock, patch

import numpy as np

from mtp.qdrant_tiered import check_routing, collections, make_workloads, routed_query


class TieredTests(unittest.TestCase):
    def test_query_routes_to_tenant_with_fallback_and_keeps_exact_flag(self):
        case = (12, 0, [1.0], {3})
        with patch("mtp.qdrant_tiered.query", return_value=(1, 1, 1)) as query:
            routed_query(Mock(), "test", case, 100, exact=True)
        kwargs = query.call_args.kwargs
        self.assertTrue(kwargs["exact"])
        self.assertEqual(kwargs["shard_key"].target, "12")
        self.assertEqual(kwargs["shard_key"].fallback, "fallback")

    def test_workloads_are_disjoint_and_ground_truth_is_tenant_scoped(self):
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(60, 4))
        queries = rng.normal(size=(200, 4))
        tenants = np.repeat([0, 1, 2], 20)
        groups = make_workloads(vectors, queries, tenants, [1, 2])
        evaluation = [q for w in groups.values() for q in w["evaluation"]]
        validation = [q for w in groups.values() for q in w["validation"]]
        self.assertTrue({q[1] for q in evaluation}.isdisjoint(q[1] for q in validation))
        self.assertEqual(len(groups["whale"]["evaluation"]), 60)
        for tenant, _, _, expected in evaluation + validation:
            self.assertEqual(len(expected), 10)
            self.assertTrue(all(tenants[i] == tenant for i in expected))

    def test_rejects_wrong_shard_population(self):
        client = Mock()
        client.count.return_value.count = 100
        with self.assertRaisesRegex(RuntimeError, "Wrong shard populations"):
            check_routing(client, "test", "tiered", np.array([0, 0, 1]))

    def test_failed_shard_creation_cleans_only_owned_collection(self):
        client = Mock()
        client.create_shard_key.side_effect = RuntimeError("shard failed")
        with self.assertRaisesRegex(RuntimeError, "shard failed"):
            with collections(client, np.ones((2, 4)), np.array([0, 1])):
                self.fail("shard creation should fail")
        client.delete_collection.assert_called_once_with(client.create_collection.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
