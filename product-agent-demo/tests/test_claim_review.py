import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.claim_review import adjudicate_claims


def claim(text_value="7天祛斑", claim_type="efficacy", conditions=None, claim_id="input:claim:1"):
    return {
        "claim_id": claim_id,
        "original_text": text_value,
        "claim_type": claim_type,
        "product_context": {"brand": "Brand", "product_name": "Cream", "version": None},
        "conditions": conditions or {},
    }


def evidence(source_type="research", excerpt="Brand Cream 研究支持 7天祛斑", is_ugc=False):
    return {
        "evidence_id": "ev-1", "source_id": "s1", "title": "研究正文",
        "source_type": source_type, "publisher": "Research Lab", "url": "https://research.example/study",
        "provider_marked_official": source_type == "brand",
        "availability": "body_available", "is_ugc": is_ugc,
        "original_excerpt": excerpt, "excerpt_locator": {"start": 0, "end": len(excerpt)},
        "content_fingerprint": "fixture-fingerprint",
    }


def package(**overrides):
    result = {
        "claim_id": "input:claim:1",
        "candidates": [{"evidence_id": "ev-1", "source_id": "s1", "product_match": "matched",
                        "content_relevance": "relevant", "covered_conditions": [], "uncovered_conditions": []}],
        "gaps": [],
    }
    result.update(overrides)
    return result


def audit(c, e, status="有资料支持", **overrides):
    result = {
        "claim_id": c["claim_id"], "status": status,
        "supporting_evidence": [{"evidence_id": e["evidence_id"], "source_id": e["source_id"],
                                 "quote": e["original_excerpt"], "supported_text": c["original_text"],
                                 "reason": "正文直接描述该声明", "relation": "supports",
                                 "condition_assessments": {}}],
    }
    result.update(overrides)
    return result


class ClaimReviewTests(unittest.TestCase):
    def adjudicate(self, c, e, p, model_audit=None, **kwargs):
        return adjudicate_claims([c], [e], [p], [model_audit or audit(c, e)], **kwargs)[0]

    def test_specification_page_cannot_support_efficacy(self):
        e = evidence("brand", "Brand Cream 规格 50mL")
        result = self.adjudicate(c=claim(), e=e, p=package(candidates=[{**package()["candidates"][0], "content_relevance": "irrelevant"}]),
                                 model_audit=audit(claim(), e, supported_text="50mL"))
        self.assertEqual(result["status"], "待核验")

    def test_full_support_requires_research_and_conditions(self):
        c = claim("7天祛斑，30人测试", conditions={"time": {"original_text": "7天"}, "sample_size": {"original_text": "30人"}})
        e = evidence(excerpt="Brand Cream 研究支持 7天祛斑，30人测试")
        p = package(candidates=[{**package()["candidates"][0], "covered_conditions": ["time", "sample_size"]}])
        a = audit(c, e)
        a["supporting_evidence"][0]["condition_assessments"] = {
            "time": {"status": "matched", "quote": "7天"},
            "sample_size": {"status": "matched", "quote": "30人"},
        }
        result = self.adjudicate(c, e, p, a)
        self.assertEqual(result["status"], "有资料支持")
        self.assertEqual(result["evidence_ids"], ["ev-1"])

    def test_brand_self_claim_is_partial_and_ugc_is_partial_with_limit(self):
        c = claim()
        brand = self.adjudicate(c, evidence("brand"), package(), audit(c, evidence("brand")))
        self.assertEqual(brand["status"], "部分支持")
        ugc = self.adjudicate(c, evidence("ugc", is_ugc=True), package(), audit(c, evidence("ugc", is_ugc=True)))
        self.assertEqual(ugc["status"], "待核验")
        self.assertTrue(ugc["limitations"])

    def test_unknown_ids_and_missing_claims_are_conservative(self):
        c = claim()
        result = adjudicate_claims([c], [evidence()], [package(candidates=[{"evidence_id": "missing", "product_match": "matched"}])],
                                   [{"claim_id": "missing-claim", "status": "有资料支持", "evidence_ids": ["ev-1"]}])[0]
        self.assertEqual(result["status"], "待核验")
        self.assertEqual(result["evidence_ids"], [])

    def test_explicit_contradiction_is_conflict_but_time_mismatch_is_not(self):
        c = claim()
        e = evidence(excerpt="Brand Cream 研究未观察到祛斑效果")
        a = {"claim_id": c["claim_id"], "status": "存在冲突", "contradicting_evidence": [{
            "evidence_id": "ev-1", "source_id": "s1", "quote": e["original_excerpt"],
            "supported_text": "未观察到祛斑效果", "reason": "同一产品反向结果", "relation": "contradicts",
            "comparable": True}],}
        conflict = self.adjudicate(c, e, package(candidates=[{**package()["candidates"][0], "content_relevance": "relevant"}]), a)
        self.assertEqual(conflict["status"], "存在冲突")
        c2 = claim("7天见效", conditions={"time": {"original_text": "7天"}})
        e2 = evidence(excerpt="Brand Cream 研究：28天观察到改善")
        a2 = audit(c2, e2)
        a2["supporting_evidence"][0]["condition_assessments"] = {"time": {"status": "unmatched", "quote": "28天"}}
        mismatch = self.adjudicate(c2, e2, package(candidates=[{**package()["candidates"][0], "uncovered_conditions": ["time"]}]), a2)
        self.assertNotEqual(mismatch["status"], "存在冲突")

    def test_failure_returns_one_pending_result_per_claim(self):
        claims = [claim(), claim("保湿", claim_id="input:claim:2")]
        results = adjudicate_claims(claims, [], [], [], retrieval_status="error", failure="模型超时")
        self.assertEqual(len(results), 2)
        self.assertTrue(all(item["status"] == "待核验" for item in results))
        self.assertTrue(all("模型超时" in item["gaps"][0] for item in results))


if __name__ == "__main__":
    unittest.main()
