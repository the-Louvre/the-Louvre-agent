import asyncio
import unittest

from app.evidence import build_packages, legacy_sources, merge_evidence, normalize_records


class EvidenceNormalizationTests(unittest.TestCase):
    def test_provider_records_keep_body_metadata_and_distinguish_summary(self):
        items = normalize_records([
            {"url": "https://brand.example/product", "title": "Product", "publisher": "Brand",
             "source_level": "official", "raw_content": "7天修护，受试者30人", "brand": "Brand",
             "product_name": "Cream", "published_at": "2026-09-01"},
            {"url": "https://brand.example/summary", "title": "Search result", "snippet": "可能含有某成分"},
            {"url": "https://social.example/post", "title": "User post", "snippet": "我觉得有效",
             "is_ugc": True, "is_official": True},
        ], "tavily")
        self.assertEqual([item["availability"] for item in items], ["body_available", "summary_only", "summary_only"])
        self.assertEqual(items[0]["source_type"], "brand")
        self.assertEqual(items[0]["publisher"], "Brand")
        self.assertEqual(items[0]["content_source"], "body")
        self.assertEqual(items[1]["content_source"], "provider_snippet")
        self.assertTrue(items[2]["is_ugc"])
        self.assertIn("ugc_official_conflict", items[2]["trust_basis"])
        self.assertEqual(len(legacy_sources(items)), 3)

    def test_same_provider_url_is_deduplicated_and_body_wins(self):
        items = normalize_records([
            {"url": "https://brand.example/a", "title": "A", "snippet": "short"},
            {"url": "https://brand.example/a", "title": "A", "raw_content": "long original text"},
        ], "bailian")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["availability"], "body_available")
        self.assertEqual(items[0]["original_excerpt"], "long original text")

    def test_different_urls_with_same_text_keep_separate_provenance(self):
        first = normalize_records([{
            "url": "https://brand.example/a", "title": "A", "raw_content": "同一段原文",
            "source_level": "official", "publisher": "Brand"
        }], "bailian")
        second = normalize_records([{
            "url": "https://mirror.example/a", "title": "Mirror", "raw_content": "同一段原文",
            "source_level": "platform", "publisher": "Platform"
        }], "tavily")
        merged = merge_evidence(first, second)
        self.assertEqual(len(merged), 2)
        self.assertEqual({item["url"] for item in merged}, {
            "https://brand.example/a", "https://mirror.example/a"
        })

    def test_official_label_is_preserved_but_not_trusted_without_publisher(self):
        items = normalize_records([{
            "url": "https://brand.example/product", "title": "Product",
            "source_level": "official", "raw_content": "官方页面"
        }], "tavily")
        self.assertTrue(items[0]["provider_marked_official"])
        self.assertIn("provider_label:official", items[0]["trust_basis"])
        self.assertFalse(legacy_sources(items)[0]["trusted"])

    def test_source_type_aliases_and_ingredient_scope_are_preserved(self):
        items = normalize_records([
            {"url": "https://research.example/study", "source_type": "论文", "raw_content": "成分研究"},
            {"url": "https://test.example/report", "source_type": "第三方检测", "raw_content": "检测结果"},
        ], "mcp")
        self.assertEqual([item["source_type"] for item in items], ["research", "testing"])
        items[0]["study_scope"] = "ingredient_study"
        packages = build_packages([{
            "claim_id": "input:claim:ingredient", "original_text": "改善细纹",
            "product_context": {"brand": "Brand", "product_name": "Cream"},
        }], items[:1], "input")
        self.assertEqual(packages[0]["candidates"][0]["product_match"], "uncertain")
        self.assertIn("成分研究不能直接证明成品功效", packages[0]["candidates"][0]["reason"])

    def test_claim_package_marks_product_and_condition_gaps(self):
        evidence = normalize_records([{
            "url": "https://brand.example/product", "title": "Brand Cream",
            "raw_content": "7天修护", "source_level": "official", "brand": "Brand",
            "product_name": "Cream"
        }], "bailian")
        evidence[0]["source_id"] = "s1"
        packages = build_packages([{
            "claim_id": "input:claim:1", "original_text": "7天修护，30人测试",
            "normalized_text": "7天修护", "product_context": {"brand": "Brand", "product_name": "Cream"},
            "conditions": {"time": {"value": 7, "unit": "天", "original_text": "7天"},
                           "sample_size": {"value": 30, "unit": "人", "original_text": "30人"}}
        }], evidence, "input")
        self.assertEqual(packages[0]["claim_id"], "input:claim:1")
        self.assertEqual(packages[0]["candidates"][0]["product_match"], "matched")
        self.assertIn("sample_size", packages[0]["gaps"])
        self.assertIn("time", packages[0]["candidates"][0]["covered_conditions"])


if __name__ == "__main__":
    unittest.main()
