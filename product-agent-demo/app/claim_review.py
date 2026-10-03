"""Conservative enforcement of model-proposed semantics against acquired evidence.

Candidate retrieval matches are hints, never verdicts. This validates provenance
and prerequisites; it does not claim to prove the model's semantic accuracy.
"""
from collections import Counter
import re
from urllib.parse import urlparse

STATUSES = ("有资料支持", "部分支持", "待核验", "存在冲突")
CONDITIONS = ("efficacy_metric", "value", "time", "audience", "usage_condition", "sample_size", "endorsements")


def text(value):
    return value.strip() if isinstance(value, str) else ""


def items(value):
    return value if isinstance(value, list) else []


def fold(value):
    return re.sub(r"\s+", "", text(value)).casefold()


def condition_text(value):
    if isinstance(value, list):
        return " ".join(condition_text(v) for v in value)
    if isinstance(value, dict):
        return text(value.get("original_text") or value.get("name"))
    return text(value)


def source_tier(item):
    """Return the evidentiary tier without turning metadata into a verdict.

    Search-provider citations often have useful answer text but no publisher or
    source label. They may support a bounded, summary-level result; they cannot
    satisfy the independent/body requirements for a full efficacy verdict.
    """
    host = urlparse(text(item.get("url"))).hostname or ""
    if item.get("is_ugc") or item.get("source_type") == "ugc":
        return ""
    basis = items(item.get("trust_basis"))
    if "ugc_official_conflict" in basis:
        return ""
    if item.get("source_type") == "registration":
        return "independent" if host.endswith((".gov.cn", ".nifdc.org.cn")) and "regulatory_domain" in basis else "context"
    if item.get("source_type") == "brand":
        return "brand" if text(item.get("publisher")) and item.get("provider_marked_official") else "context"
    if item.get("source_type") in {"research", "testing"}:
        return "independent"
    if item.get("source_type") in {"platform", "other"}:
        return "context"
    return ""


def usable_source(item):
    """Compatibility boolean for callers that only need source eligibility."""
    return bool(source_tier(item))


def product_matches(claim, item):
    expected = claim.get("product_context") or {}
    actual = item.get("product") or {}
    body = fold(item.get("original_excerpt") or item.get("original_text"))
    # A name is required. Brand alone cannot identify a finished product.
    if not text(expected.get("product_name")):
        return False
    # 2B may have only a normalized package-level match when the provider did
    # not expose structured product fields. Re-check explicit fields when they
    # are available; otherwise preserve that package match.
    if not any(text(actual.get(key)) for key in ("brand", "product_name", "specification", "version")):
        return True
    for key in ("brand", "product_name", "specification", "version"):
        want, have = fold(expected.get(key)), fold(actual.get(key))
        if want and have and want != have:
            return False
        if want and not have and (key not in {"brand", "product_name"} or want not in body):
            return False
    return True


def locate(item, quote):
    body = text(item.get("original_excerpt") or item.get("original_text"))
    locator = item.get("excerpt_locator") or item.get("locator")
    if not text(quote) or not isinstance(locator, dict):
        return None
    start = locator.get("start")
    if not isinstance(start, int) or isinstance(start, bool) or start < 0:
        return None
    at = body.find(quote)
    if at < 0 or not item.get("content_fingerprint"):
        return None
    return {"kind": locator.get("kind"), "start": start + at, "end": start + at + len(quote),
            "content_fingerprint": item["content_fingerprint"]}


def _conditions(claim, assessment, item):
    expected = claim.get("conditions") or {}
    matches = assessment.get("condition_assessments") or {}
    if not isinstance(matches, dict):
        matches = {}
    covered, missing = [], []
    for key in CONDITIONS:
        original = condition_text(expected.get(key))
        if not original:
            continue
        candidate = matches.get(key)
        quote = text(candidate.get("quote")) if isinstance(candidate, dict) else ""
        if (isinstance(candidate, dict) and candidate.get("status") == "matched"
                and locate(item, quote) and fold(original) in fold(quote)):
            covered.append(key)
        else:
            missing.append(key)
    return covered, missing


def review_context(claims, evidence_items, claim_packages, per_claim=2):
    """Send a small set of eligible candidates to the model; keep full evidence for validation."""
    evidence = {text(v.get("evidence_id")): v for v in items(evidence_items) if isinstance(v, dict)}
    claims_by_id = {text(v.get("claim_id")): v for v in items(claims) if isinstance(v, dict)}
    selected, wanted = [], {}
    for package in items(claim_packages):
        if not isinstance(package, dict):
            continue
        claim = claims_by_id.get(text(package.get("claim_id")), {})
        ranked = []
        for link in items(package.get("candidates")):
            if not isinstance(link, dict) or text(link.get("evidence_id")) not in evidence:
                continue
            pm, relevance = text(link.get("product_match")), text(link.get("content_relevance"))
            if pm not in {"matched", "related"} or relevance not in {"relevant", "related"}:
                continue
            item = evidence[link["evidence_id"]]
            score = (4 if pm == "matched" else 2) + (4 if relevance == "relevant" else 2)
            score += 1 if item.get("availability") == "body_available" else 0
            ranked.append((score, link))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        links = [link for _, link in ranked[:per_claim]]
        selected.append({**package, "candidates": links})
        for link in links:
            wanted.setdefault(link["evidence_id"], []).append(text(claim.get("original_text")))
    compact = []
    for eid, claim_texts in wanted.items():
        item = evidence[eid]
        body = text(item.get("original_excerpt") or item.get("original_text"))
        positions = []
        for claim_text in claim_texts:
            chunks = re.findall(r"[\u4e00-\u9fff]{2,4}", claim_text)
            positions.extend(body.find(chunk) for chunk in chunks if chunk in body)
        start = max(0, min(positions) - 300) if positions else 0
        excerpt = body[start:start + 1600]
        compact.append({key: item.get(key) for key in ("evidence_id", "source_id", "url", "title", "publisher",
                            "source_type", "availability", "study_scope", "product", "experiment_conditions")}
                       | {"original_excerpt": excerpt})
    return selected, compact


def adjudicate_claims(claims, evidence_items, claim_packages, model_audits, retrieval_status="ok", failure=None):
    evidence = {text(v.get("evidence_id")): v for v in items(evidence_items) if isinstance(v, dict)}
    packages = {text(v.get("claim_id")): v for v in items(claim_packages) if isinstance(v, dict)}
    inputs = {text(v.get("claim_id")): v for v in items(claims) if isinstance(v, dict) and text(v.get("claim_id"))}
    proposals = {}
    for audit in items(model_audits):
        if not isinstance(audit, dict):
            continue
        cid = text(audit.get("claim_id"))
        # Legacy text matching is only allowed when no ID was supplied.
        if not cid:
            matches = [k for k, v in inputs.items() if text(v.get("original_text")) == text(audit.get("claim"))]
            cid = matches[0] if len(matches) == 1 else ""
        if cid in inputs:
            proposals.setdefault(cid, []).append(audit)
    results = []
    for cid, claim in inputs.items():
        package = packages.get(cid, {})
        links = {text(v.get("evidence_id")): v for v in items(package.get("candidates")) if isinstance(v, dict)}
        gaps = [text(v) for v in items(package.get("gaps")) + items(package.get("retrieval_events")) if text(v)]
        limitations = list(items(claim.get("uncertainty_reasons")))
        support, against, covered = [], [], set()
        missing = {key for key in CONDITIONS if condition_text((claim.get("conditions") or {}).get(key))}
        full = False
        candidates = proposals.get(cid, [])
        audit = candidates[0] if len(candidates) == 1 else {}
        invalid = bool(failure or len(candidates) != 1 or audit.get("status") not in STATUSES)
        if failure:
            gaps.append("review_failure: " + str(failure)[:180])
        elif not candidates:
            gaps.append("model_result_missing")
        elif len(candidates) > 1:
            gaps.append("duplicate_model_results")
        elif audit.get("status") not in STATUSES:
            gaps.append("invalid_status")
        for key in ("evidence_ids", "source_ids"):
            valid_ids = set(links) if key == "evidence_ids" else {text(evidence[eid].get("source_id")) for eid in links if eid in evidence}
            if key in audit and (not isinstance(audit[key], list) or any(text(v) not in valid_ids for v in audit[key])):
                invalid = True
                gaps.append("invalid_" + key)
        assessments = [(v, "supports") for v in items(audit.get("supporting_evidence"))]
        assessments += [(v, "contradicts") for v in items(audit.get("contradicting_evidence"))]
        for assessment, direction in assessments:
            if not isinstance(assessment, dict):
                gaps.append("malformed_evidence_assessment")
                invalid = True
                continue
            eid = text(assessment.get("evidence_id"))
            item, link = evidence.get(eid), links.get(eid)
            if not item or not link or text(assessment.get("source_id")) != text(item.get("source_id")) or (
                    link.get("source_id") and link["source_id"] != item.get("source_id")):
                gaps.append("invalid_evidence_reference")
                invalid = True
                continue
            quote = text(assessment.get("quote"))
            locator = locate(item, quote)
            if not locator:
                gaps.append("unlocatable_quote")
                invalid = True
                continue
            tier = source_tier(item)
            product_match = text(link.get("product_match")).lower()
            relevance = text(link.get("content_relevance")).lower()
            exact_link = product_match in {"matched", "match", "相符"} and relevance in {"relevant", "直接相关", "matched"}
            related_link = product_match in {"matched", "match", "相符", "related"} and relevance in {"relevant", "直接相关", "matched", "related"}
            if (not tier or not product_matches(claim, item)
                    or not related_link):
                gaps.append("source_or_product_inapplicable")
                continue
            relation = assessment.get("relation")
            if relation not in ({"supports", "partial"} if direction == "supports" else {"contradicts"}):
                gaps.append("invalid_semantic_relation")
                invalid = True
                continue
            # Require a semantic explanation and a separable proposition. Retrieval
            # substring relevance alone cannot produce a positive result.
            proposition = text(assessment.get("supported_text"))
            explanation = text(assessment.get("reason"))
            if not proposition or not explanation:
                gaps.append("semantic_assessment_missing")
                continue
            is_effect = claim.get("claim_type") in {"efficacy", "endorsement"} or bool((claim.get("conditions") or {}).get("efficacy_metric"))
            scope = text(item.get("study_scope")).lower()
            if is_effect and (scope in {"ingredient", "ingredient_study", "成分研究"} or item.get("source_type") == "registration"):
                gaps.append("ingredient_or_registration_extrapolation")
                continue
            matched, absent = _conditions(claim, assessment, item)
            view = {"evidence_id": eid, "source_id": item["source_id"], "excerpt": quote, "locator": locator,
                    "source_type": item.get("source_type"), "publisher": item.get("publisher"),
                    "availability": item.get("availability"), "supported_text": proposition,
                    "reason": explanation, "covered_conditions": matched, "uncovered_conditions": absent,
                    "experiment_conditions": item.get("experiment_conditions") or {}}
            if direction == "contradicts":
                # A difference in time, audience, usage or sample is not a refutation.
                incomparable = set(absent) - {"value", "efficacy_metric"}
                if (not exact_link or assessment.get("comparable") is not True or incomparable
                        or item.get("availability") != "body_available"
                        or (is_effect and item.get("source_type") not in {"research", "testing"})):
                    gaps.append("counterevidence_not_comparable")
                    continue
                against.append(view)
            else:
                support.append(view)
                covered.update(matched)
                if not exact_link:
                    limitations.append("仅确认同系列或声明子命题，未确认当前产品/版本和完整表述")
                if exact_link and relation == "supports" and not absent and item.get("availability") == "body_available":
                    if tier in {"independent", "brand"} and (not is_effect or (item.get("source_type") in {"research", "testing"}
                                         and scope not in {"ingredient", "ingredient_study", "成分研究", "registration", "备案"})):
                        full = True
                if item.get("source_type") == "brand" and is_effect:
                    limitations.append("品牌页面如此声称，不等同于独立功效验证")
                if item.get("availability") != "body_available":
                    limitations.append("只有提供方原文摘要片段，未获取完整正文")
                if tier == "context":
                    limitations.append("来源类型或发布主体未充分确认，仅作为背景或摘要级材料")
                if is_effect and item.get("source_type") in {"research", "testing"} and scope not in {"finished_product", "product", "成品研究"}:
                    limitations.append("尚未确认研究适用于具体成品")
        if claim.get("parse_status") in {"unparsed", "partially_parsed", "budget_excluded"}:
            invalid = True
            gaps.append("claim_parse_uncertain")
        if invalid:
            status = "待核验"
            support, against = [], []
            covered = set()
            reason = "声明或模型输出无法完整校验，已保守降级；详见缺口"
        elif against:
            status = "存在冲突"
            reason = "存在可追溯且条件可比的明确反证；详见反对依据和冲突点"
        elif full:
            status = "有资料支持"
            reason = "语义候选与可核对正文、产品适用性及关键条件检查均通过；支持范围见依据"
        elif support:
            status = "部分支持"
            reason = "只支持列出的子命题或限定范围，未覆盖内容与材料限制见下方"
        else:
            status = "待核验"
            reason = "没有满足来源、产品、原文和适用条件要求的直接证据"
        missing -= covered
        if retrieval_status in {"error", "timeout", "empty", "disabled", "cancelled"}:
            gaps.append("retrieval_" + retrieval_status)
        limitations = [text(v) for v in limitations if text(v)]
        limitations.append("语义关系由模型提出，规则校验不能证明语义判断准确")
        refs = support + against
        results.append({"claim_id": cid, "claim": claim.get("original_text", ""), "original_text": claim.get("original_text", ""),
                        "status": status, "reason": reason, "evidence_ids": list(dict.fromkeys(v["evidence_id"] for v in refs)),
                        "source_ids": list(dict.fromkeys(v["source_id"] for v in refs)), "supporting_evidence": support,
                        "contradicting_evidence": against, "covered_conditions": sorted(covered), "uncovered_conditions": sorted(missing),
                        "limitations": list(dict.fromkeys(limitations)), "gaps": list(dict.fromkeys(gaps))})
    return results


def build_review_summary(verdicts):
    counts = Counter(v.get("status") for v in verdicts)
    if not verdicts:
        return "没有可处理的结构化声明。"
    return "，".join([f"共核验 {len(verdicts)} 条声明"] + [f"{s}{counts[s]}条" for s in STATUSES if counts[s]]) + "。支持范围、条件及限制详见逐声明依据。"
