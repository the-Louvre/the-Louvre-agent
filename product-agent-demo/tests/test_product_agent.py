import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas import ProductIdentification, EvidenceEvent
from app.model_client import MissingBailianKeyError, BailianVisionClient
from app.orchestrator import ProductAgentOrchestrator


class ProductAgentTests(unittest.TestCase):
    def test_identification_requires_structured_product_fields(self):
        result = ProductIdentification.model_validate({
            "brand": "L'Oreal Paris",
            "product_name": "Revitalift",
            "category": "护肤",
            "confidence": 0.92,
            "evidence": ["瓶身文字可见"],
            "needs_more_image": False,
        })
        self.assertEqual(result.brand, "L'Oreal Paris")
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_missing_bailian_key_is_explicit(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": ""}):
            client = BailianVisionClient(api_key=None)
            with self.assertRaises(MissingBailianKeyError):
                client.identify("/tmp/product.jpg")

    def test_orchestrator_emits_auditable_events(self):
        class FakeVision:
            def identify(self, image_path):
                return ProductIdentification(
                    brand="测试品牌",
                    product_name="测试产品",
                    category="护肤",
                    confidence=0.88,
                    evidence=["包装文字"],
                    needs_more_image=False,
                )

        class EmptyEvidence:
            def search(self, product):
                return []

        events = asyncio.run(ProductAgentOrchestrator(FakeVision(), EmptyEvidence()).run("/tmp/product.jpg"))
        self.assertEqual(events[0].type, "identification_started")
        self.assertEqual(events[-1].type, "report_ready")
        self.assertTrue(any(e.type == "evidence_summary" for e in events))


if __name__ == "__main__":
    unittest.main()

class EvidenceProviderTests(unittest.TestCase):
    def test_configured_providers_run_in_parallel_with_cap(self):
        import time
        from app.schemas import EvidenceItem
        from app.orchestrator import ConfiguredEvidenceProvider

        class SlowProvider:
            def __init__(self, name):
                self.name = name
            def search(self, product):
                time.sleep(0.15)
                return [EvidenceItem(source_name=self.name, source_level="official", claim="ok")]

        started = time.monotonic()
        result = ConfiguredEvidenceProvider([SlowProvider("a"), SlowProvider("b")]).search(
            ProductIdentification(brand="b", product_name="p", category="护肤", confidence=0.9)
        )
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.27)
        self.assertEqual({item.source_name for item in result}, {"a", "b"})

class ProductKbTests(unittest.TestCase):
    def test_product_kb_matches_alias_and_returns_official_evidence(self):
        import json
        import tempfile
        from app.product_kb import ProductKbProvider

        payload = [{
            "product_id": "loreal-001",
            "brand": "欧莱雅",
            "product_name": "紫熨斗眼霜",
            "aliases": ["Revitalift Filler", "紫熨斗"],
            "category": "护肤",
            "official_url": "https://example.com/product",
            "ingredients_url": "https://example.com/ingredients"
        }]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            f.flush()
            provider = ProductKbProvider(f.name)
            items = provider.search(ProductIdentification(
                brand="欧莱雅", product_name="Revitalift Filler", category="护肤", confidence=0.9
            ))
        self.assertEqual(len(items), 2)
        self.assertTrue(all(item.source_level == "official" for item in items))

class LiveEventTests(unittest.TestCase):
    def test_started_event_arrives_before_vision_call(self):
        class NotYetCalledVision:
            called = False
            def identify(self, path):
                self.called = True
                return ProductIdentification(brand="b", product_name="p", category="护肤", confidence=0.9)
        class EmptyEvidence:
            def search(self, product):
                return []
        vision = NotYetCalledVision()
        async def first_event():
            stream = ProductAgentOrchestrator(vision, EmptyEvidence()).stream("/tmp/a.png")
            event = await stream.__anext__()
            await stream.aclose()
            return event
        event = asyncio.run(first_event())
        self.assertEqual(event.type, "identification_started")
        self.assertFalse(vision.called)

class IdentificationGateTests(unittest.TestCase):
    def test_uncertain_identity_does_not_query_wrong_product(self):
        class UncertainVision:
            def identify(self, path):
                return ProductIdentification(brand="未知", product_name="候选产品", category="护肤", confidence=0.4)
        class EvidenceMustNotRun:
            def search(self, product):
                raise AssertionError("uncertain identity must not trigger external search")
        events = asyncio.run(ProductAgentOrchestrator(UncertainVision(), EvidenceMustNotRun()).run("/tmp/a.png"))
        self.assertTrue(any(event.type == "clarification_needed" for event in events))
        self.assertFalse(any(event.type == "evidence_search_started" for event in events))
        self.assertFalse(any(event.type == "error" for event in events))

class ProviderFailureTests(unittest.TestCase):
    def test_one_provider_failure_does_not_drop_other_evidence(self):
        from app.orchestrator import ConfiguredEvidenceProvider
        from app.schemas import EvidenceItem

        class GoodProvider:
            def search(self, product):
                return [EvidenceItem(source_name="official", source_level="official", claim="confirmed")]
        class BrokenProvider:
            def search(self, product):
                raise TimeoutError("upstream timeout")

        items = ConfiguredEvidenceProvider([GoodProvider(), BrokenProvider()]).search(
            ProductIdentification(brand="b", product_name="p", category="护肤", confidence=0.9)
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source_name, "official")

class RequestAuditTests(unittest.TestCase):
    def test_identification_event_carries_model_request_id(self):
        class AuditedVision:
            last_request_id = "req-demo-001"
            def identify(self, path):
                return ProductIdentification(brand="b", product_name="p", category="护肤", confidence=0.9)
        class EmptyEvidence:
            def search(self, product):
                return []
        events = asyncio.run(ProductAgentOrchestrator(AuditedVision(), EmptyEvidence()).run("/tmp/a.png"))
        completed = next(e for e in events if e.type == "identification_completed")
        self.assertEqual(completed.data["model_request_id"], "req-demo-001")


class ConfigurableAgentTests(unittest.TestCase):
    def test_openai_compatible_config_has_safe_defaults(self):
        from app.config import ModelConfig
        config = ModelConfig.from_dict({"api_key": "secret", "model": "qwen-vl"})
        self.assertEqual(config.base_url, "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(config.timeout_seconds, 20)
        self.assertEqual(config.temperature, 0.0)

    def test_report_contains_source_calls_and_claim_audits(self):
        from app.report import build_official_source_report
        report = build_official_source_report(
            product={"brand": "测试品牌", "product_name": "测试产品", "confidence": 0.9},
            claims=[{"claim_id": "c1", "text": "7天修护", "type": "efficacy_time"}],
            evidence=[{"source_name": "品牌官方页", "source_level": "official", "claim": "支持保湿"}],
            source_calls=[{"source_id": "brand_page", "status": "matched", "latency_ms": 120}],
        )
        self.assertEqual(report["report_type"], "official_source_report")
        self.assertEqual(report["source_calls"][0]["status"], "matched")
        self.assertEqual(report["claim_evidence_audit"][0]["evidence_status"], "unsupported")

    def test_lab_report_drops_facts_with_unknown_source_ids(self):
        from app.lab import validate_report
        result = validate_report({
            "summary": "x",
            "official_facts": [{"fact": "未经证实", "source_ids": ["unknown"]}],
            "claim_evidence_audit": [],
            "evidence_gaps": [],
        }, [{"source_id": "s1", "url": "https://example.com"}])
        self.assertEqual(result["official_facts"], [])
        self.assertTrue(result["evidence_gaps"])
