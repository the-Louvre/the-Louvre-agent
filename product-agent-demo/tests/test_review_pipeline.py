import asyncio
import json
import unittest
from unittest.mock import patch

import httpx

from app.lab import run_models, sources_from


class ReviewPipelineTests(unittest.TestCase):
    def test_tavily_series_evidence_produces_partial_verdict(self):
        record = {
            "url": "https://shop.example/revitalift",
            "title": "欧莱雅(L'OREAL)复颜抗皱紧致水乳套装",
            "raw_content": "导航" * 600 + "欧莱雅复颜系列抗皱紧致水乳套装，柔肤水130ml，乳液110ml。",
        }
        calls = []

        async def retrieve(*args, **kwargs):
            calls.append(args[1])
            kwargs["records_out"].append(record)
            return sources_from([record]), record["raw_content"]

        async def handler(request):
            body = json.loads(request.content)
            if body["model"] == "vision-model":
                identity = {"brand": "L'OREAL PARIS", "product_name": "复颜中秋团圆美礼",
                            "specification": "复颜柔肤水130ml+复颜紧致乳110ml",
                            "claims": ["抗皱紧致 养出好气色"], "confidence": .95}
                return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(identity)}}]})
            task = json.loads(body["messages"][1]["content"][1]["text"])
            candidate = task["claim_evidence_packages"][0]["candidates"][0]
            item = task["evidence_items"][0]
            self.assertEqual(candidate["product_match"], "related")
            self.assertEqual(candidate["content_relevance"], "related")
            self.assertIn("抗皱紧致", item["original_excerpt"])
            self.assertLessEqual(len(item["original_excerpt"]), 1600)
            claim_id = task["claims_structured"][0]["claim_id"]
            audit = {"claim_id": claim_id, "status": "部分支持", "supporting_evidence": [{
                "evidence_id": item["evidence_id"], "source_id": item["source_id"],
                "quote": "抗皱紧致", "supported_text": "抗皱紧致",
                "reason": "同系列页面提及抗皱紧致", "relation": "partial",
            }]}
            report = {"claim_evidence_audit": [audit]}
            chunk = {"choices": [{"delta": {"content": json.dumps(report, ensure_ascii=False)}}]}
            return httpx.Response(200, text="data: " + json.dumps(chunk, ensure_ascii=False) + "\n\ndata: [DONE]\n\n")

        real_client = httpx.AsyncClient
        with patch("app.lab.retrieve", side_effect=retrieve), patch(
            "app.lab.httpx.AsyncClient",
            side_effect=lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
        ):
            events = asyncio.run(self._collect())
        report = next(event["report"] for event in events if event["type"] == "report")
        self.assertEqual(report["claim_evidence_audit"][0]["status"], "部分支持")
        self.assertEqual(len(report["claim_evidence_audit"][0]["supporting_evidence"]), 1)
        self.assertTrue(calls)

    @staticmethod
    async def _collect():
        return [json.loads(event.removeprefix("data: ")) async for event in run_models({
            "models": [{"id": "review", "agent_role": "review", "model": "review-model",
                        "base_url": "https://model.example/v1", "api_key": "x"}],
            "vision_model": "vision-model", "search_enabled": True,
            "search": {"provider": "tavily", "api_key": "x"},
        }, "data:image/png;base64,a")]
