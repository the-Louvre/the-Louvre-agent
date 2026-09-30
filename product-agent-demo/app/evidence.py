"""Traceable, claim-scoped candidates built from search-provider records.

This module deliberately does not decide whether a marketing claim is supported.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit


CONDITION_KEYS = ("efficacy_metric", "value", "time", "audience", "usage_condition", "sample_size", "endorsements")
SOURCE_TYPES = {"brand", "registration", "research", "testing", "platform", "ugc", "other"}
SOURCE_TYPE_ALIASES = {
    "official": "brand",
    "authority": "registration",
    "user": "ugc",
    "user_content": "ugc",
    "study": "research",
    "test": "testing",
    "官方": "brand",
    "官方来源": "brand",
    "品牌官方": "brand",
    "监管": "registration",
    "监管公开来源": "registration",
    "研究": "research",
    "论文": "research",
    "检测": "testing",
    "第三方检测": "testing",
    "平台": "platform",
    "用户": "ugc",
    "用户内容": "ugc",
}


def _text(value):
    return str(value or "").strip()


def _hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _fold(value):
    return re.sub(r"[\W_]+", "", _text(value).casefold(), flags=re.UNICODE)


def _url(value):
    try:
        parts = urlsplit(_text(value))
        if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
            return ""
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", parts.query, ""))
    except ValueError:
        return ""


def _source_type(raw):
    claimed = _text(raw.get("source_type") or raw.get("source_level") or raw.get("kind")).lower()
    kind = SOURCE_TYPE_ALIASES.get(claimed, claimed)
    if raw.get("is_ugc") or kind == "ugc":
        return "ugc"
    if kind in SOURCE_TYPES:
        return kind
    host = urlsplit(_url(raw.get("url"))).hostname or ""
    if host.endswith((".gov.cn", ".nifdc.org.cn")):
        return "registration"
    return "other"


def normalize_records(records, provider, fetched_at=None):
    """Map provider records to evidence, merging repeated URLs within one input run."""
    now = fetched_at or datetime.now(timezone.utc).isoformat()
    by_key = {}
    for raw in records or []:
        if not isinstance(raw, dict):
            continue
        url = _url(raw.get("url"))
        if not url:
            continue
        kind = _source_type(raw)
        is_ugc = bool(raw.get("is_ugc")) or kind == "ugc"
        raw_level = _text(raw.get("source_level") or raw.get("kind") or raw.get("source_type"))
        conflict = is_ugc and (raw.get("is_official") or raw_level.lower() in {"official", "authority", "registration"})
        body = _text(raw.get("raw_content") or raw.get("body") or raw.get("page_content"))
        snippet = _text(raw.get("snippet") or raw.get("content") or raw.get("excerpt"))
        summary = _text(raw.get("model_summary") or raw.get("answer"))
        quote = body or snippet
        status = "body_available" if body else "summary_only" if snippet else "url_only"
        if raw.get("body_unavailable"):
            status = "body_unavailable" if not quote else status
        content_source = "body" if body else "provider_snippet" if snippet else "none"
        locator = {"kind": "body" if body else "provider_snippet", "start": 0, "end": len(quote)} if quote else None
        proof = []
        if kind == "registration" and (urlsplit(url).hostname or "").endswith((".gov.cn", ".nifdc.org.cn")):
            proof.append("regulatory_domain")
        if raw_level:
            proof.append(f"provider_label:{raw_level}")
        if raw.get("publisher") or raw.get("issuer"):
            proof.append("publisher_metadata")
        if conflict:
            proof.append("ugc_official_conflict")
        source_key = f"{provider}|{url}"
        item = {
            "evidence_id": f"ev-{_hash(source_key)}", "source_id": f"src-{_hash(source_key)}", "url": url,
            "title": _text(raw.get("title")) or (urlsplit(url).hostname or url),
            "publisher": _text(raw.get("publisher") or raw.get("issuer") or raw.get("author")),
            "source_type": kind, "trust_basis": proof, "is_ugc": is_ugc,
            "provider_marked_official": bool(raw.get("is_official")) or raw_level.lower() in {"official", "authority", "registration", "官方", "官方来源", "监管公开来源"},
            "credibility_basis": proof,
            "retrieved_at": now, "published_at": _text(raw.get("published_at") or raw.get("published_date") or raw.get("date")) or None,
            "availability": status, "content_source": content_source, "original_excerpt": quote[:5000] or None,
            "excerpt_locator": locator, "content_fingerprint": _hash(quote) if quote else None,
            "fetch_status": status, "original_text": quote[:5000] or None, "locator": locator,
            "product": {key: _text(raw.get(key)) or None for key in ("brand", "product_name", "specification", "version")},
            "study_scope": _text(raw.get("study_scope") or raw.get("product_scope")) or None,
            "experiment_conditions": raw.get("experiment_conditions") if isinstance(raw.get("experiment_conditions"), dict) else {},
            "conditions": raw.get("experiment_conditions") if isinstance(raw.get("experiment_conditions"), dict) else {},
            "provider": provider, "provider_summary": summary[:2000] or None,
        }
        previous = by_key.get(source_key)
        if not previous or (status == "body_available" and previous["availability"] != "body_available") or (
            quote and len(quote) > len(previous.get("original_excerpt") or "")
        ):
            by_key[source_key] = item
    return list(by_key.values())


def merge_evidence(existing, additions):
    """Preserve the first source ID; merge only repeated canonical URLs.

    Two different URLs can publish the same wording. They remain separate
    sources so the evidence package does not lose provenance.
    """
    merged = list(existing)
    positions = {item["url"]: i for i, item in enumerate(merged)}
    for item in additions:
        at = positions.get(item["url"])
        if at is None:
            item = dict(item, source_id=f"s{len(merged) + 1}")
            positions[item["url"]] = len(merged)
            merged.append(item)
        else:
            old = merged[at]
            stronger = item["availability"] == "body_available" and old["availability"] != "body_available"
            longer = len(item.get("original_excerpt") or "") > len(old.get("original_excerpt") or "")
            if stronger or longer:
                merged[at] = {**item, "source_id": old["source_id"], "evidence_id": old["evidence_id"]}
                positions[merged[at]["url"]] = at
    return merged


def legacy_sources(evidence):
    result = []
    for index, item in enumerate(evidence):
        kind = item["source_type"]
        trusted = not item["is_ugc"] and kind == "registration" and "regulatory_domain" in item["trust_basis"]
        # A provider's "official" label is metadata, not proof by itself.
        # A brand source is only carried as trusted for the legacy consumer
        # when the label is accompanied by publisher metadata.
        if (not item["is_ugc"] and kind == "brand" and
                item.get("provider_marked_official") and "publisher_metadata" in item["trust_basis"]):
            trusted = True
        result.append({"source_id": item.get("source_id") or f"s{index + 1}", "title": item["title"], "url": item["url"],
                       "domain": urlsplit(item["url"]).hostname or "", "snippet": (item["original_excerpt"] or "")[:2500],
                       "kind": kind, "trusted": trusted, "source_type": kind, "trust_basis": item["trust_basis"],
                       "is_ugc": item["is_ugc"], "availability": item["availability"]})
    return result


def _condition_text(value):
    if isinstance(value, list):
        return " ".join(_condition_text(item) for item in value)
    if isinstance(value, dict):
        return _text(value.get("original_text") or value.get("name") or value.get("value"))
    return _text(value)


def _product_match(claim, evidence):
    expected = claim.get("product_context") or {}
    actual = evidence["product"]
    content = _fold(" ".join((evidence.get("title") or "", evidence.get("original_excerpt") or "")))
    reasons = []
    known = False
    for key in ("brand", "product_name", "specification", "version"):
        want, have = _fold(expected.get(key)), _fold(actual.get(key))
        if want and have:
            known = True
            if want != have:
                return "mismatch", f"{key} 与当前产品不一致"
        elif want and key in {"brand", "product_name"} and want in content:
            known = True
        elif want:
            reasons.append(f"未确认 {key}")
    scope = _text(evidence.get("study_scope")).lower()
    if scope in {"ingredient", "ingredient_study", "成分研究"}:
        return "uncertain", "成分研究不能直接证明成品功效"
    return ("matched" if known and not reasons else "uncertain"), ("；".join(reasons) or "产品身份字段相符")


def build_packages(claims, evidence, input_id):
    packages = []
    for claim in claims or []:
        claim_id = _text(claim.get("claim_id"))
        conditions = claim.get("conditions") or {}
        wanted = {key: _condition_text(conditions.get(key)) for key in CONDITION_KEYS if _condition_text(conditions.get(key))}
        links = []
        for index, item in enumerate(evidence):
            pm, pm_reason = _product_match(claim, item)
            excerpt = _fold(item.get("original_excerpt"))
            metric = _fold(wanted.get("efficacy_metric") or claim.get("normalized_text") or claim.get("original_text"))
            relevant = bool(metric and metric in excerpt)
            relevance = "relevant" if relevant else "uncertain" if not excerpt else "irrelevant"
            covered = [key for key, value in wanted.items() if _fold(value) and _fold(value) in excerpt]
            uncovered = [key for key in wanted if key not in covered]
            links.append({"claim_id": claim_id, "evidence_id": item["evidence_id"], "source_id": item.get("source_id") or f"s{index + 1}",
                          "product_match": pm, "content_relevance": relevance,
                          "covered_conditions": covered, "uncovered_conditions": uncovered,
                          "reason": pm_reason + ("；原文相关" if relevant else "；未找到对应原文表述")})
        direct = [link for link in links if link["product_match"] == "matched" and
                  link["content_relevance"] == "relevant" and not link["uncovered_conditions"] and
                  next((item for item in evidence if item["evidence_id"] == link["evidence_id"]), {}).get("availability") == "body_available"]
        gaps = []
        if not links:
            gaps.append("no_result")
        if links and not any(link["product_match"] == "matched" for link in links):
            gaps.append("product_identity")
        if links and not any(link["content_relevance"] == "relevant" and link["product_match"] == "matched" for link in links):
            gaps.append("claim_text")
        if links and not any(item["availability"] == "body_available" for item in evidence):
            gaps.append("body_unavailable")
        for key in wanted:
            if not any(link["product_match"] == "matched" and key in link["covered_conditions"] for link in links):
                gaps.append(key)
        if not direct and not gaps:
            gaps.append("direct_evidence")
        packages.append({"claim_id": claim_id, "input_id": input_id, "candidates": links,
                         "gaps": list(dict.fromkeys(gaps)), "retrieval_events": []})
    return packages


def followup_query(claim, gaps):
    if not gaps:
        return ""
    product = claim.get("product_context") or {}
    pieces = [product.get("brand"), product.get("product_name"), product.get("specification"),
              product.get("version"), claim.get("normalized_text") or claim.get("original_text")]
    conditions = claim.get("conditions") or {}
    pieces.extend(_condition_text(conditions.get(key)) for key in CONDITION_KEYS if key in gaps)
    pieces.extend(_condition_text(conditions.get("endorsements")) for _ in [0] if "endorsements" in gaps)
    return " ".join(dict.fromkeys(_text(piece) for piece in pieces if _text(piece)))[:350]
