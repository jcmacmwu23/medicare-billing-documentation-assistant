import unittest
from unittest.mock import Mock

from billing_assistant.claims import get_synthetic_claim
from billing_assistant.fixtures import load_fixture
from billing_assistant.retrieval import search_bm25


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.passages = load_fixture("cms_ch18.json")["passages"]

    def test_exact_code_ranks_expected_citation(self):
        cases = {"Q0091": "30.5.A", "M83": "30.9", "Z12.4": "30.6.A"}
        for query, section in cases.items():
            with self.subTest(query=query):
                hits = search_bm25(query, self.passages, limit=1)
                self.assertEqual([hit["section"] for hit in hits], [section])

    def test_unmatched_query_returns_no_passages(self):
        self.assertEqual(search_bm25("telemetry", self.passages), [])


class ClaimTests(unittest.TestCase):
    def test_allowed_lookup_returns_only_projected_fields(self):
        raw = load_fixture("claims.json")
        raw["SYN-0001"]["unexpected_field"] = "internal"
        result = get_synthetic_claim("SYN-0001", allowed=True, load_claims=lambda: raw)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["claim"]["previous_covered_screening_month"], "2024-01")
        self.assertNotIn("unexpected_field", result["claim"])

    def test_permission_is_checked_before_the_store_is_read(self):
        loader = Mock()
        result = get_synthetic_claim("SYN-0001", allowed=False, load_claims=loader)
        self.assertEqual(result, {"status": "denied"})
        loader.assert_not_called()

    def test_invalid_ids_do_not_read_the_store(self):
        loader = Mock()
        for claim_id in ("SYN-*", "SYN-0001\n", "SYN-0001,SYN-0002"):
            with self.subTest(claim_id=claim_id):
                result = get_synthetic_claim(claim_id, allowed=True, load_claims=loader)
                self.assertEqual(result, {"status": "invalid_id"})
        loader.assert_not_called()

    def test_unknown_claim_is_distinct_from_missing_history(self):
        result = get_synthetic_claim("SYN-9999", allowed=True)
        self.assertEqual(result, {"status": "not_found"})
        result = get_synthetic_claim("SYN-0002", allowed=True)
        self.assertEqual(result["status"], "ok")
        self.assertIsNone(result["claim"]["previous_covered_screening_month"])
