from __future__ import annotations

from typing import Any, Dict, List


def build_official_source_report(
    product: Dict[str, Any],
    claims: List[Dict[str, Any]],
    evidence: List[Dict[str, Any]],
    source_calls: List[Dict[str, Any]],
) -> Dict[str, Any]:
    trusted_levels = {"official", "registration", "authority"}
    trusted_evidence = [item for item in evidence if item.get("source_level") in trusted_levels and not item.get("is_ugc")]
    evidence_text = " ".join(str(item.get("claim", "")) for item in trusted_evidence).lower()
    audits = []
    for claim in claims:
        text = str(claim.get("text", ""))
        supported = text.lower() in evidence_text if text else False
        audits.append({
            "claim_id": claim.get("claim_id"),
            "claim": text,
            "evidence_status": "supported" if supported else "unsupported",
            "matched_evidence": trusted_evidence if supported else [],
            "reason": "找到官方资料中的对应表述" if supported else "当前官方资料未覆盖该声明",
        })
    return {
        "report_type": "official_source_report",
        "product": product,
        "claims": claims,
        "official_facts": trusted_evidence,
        "claim_evidence_audit": audits,
        "source_calls": source_calls,
        "limitations": [] if trusted_evidence else ["当前没有可用的官方证据；用户内容仅作为待核验线索"],
    }
