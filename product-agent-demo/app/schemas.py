from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ProductIdentification(BaseModel):
    brand: str = Field(min_length=1)
    product_name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    series: Optional[str] = None
    specification: Optional[str] = None
    variant: Optional[str] = None
    barcode: Optional[str] = None
    candidate_products: List[Dict[str, Any]] = Field(default_factory=list)
    visual_evidence: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)
    needs_more_image: bool = False
    ocr_text: str = ""
    visible_claims: List[Dict[str, Any]] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    source_name: str
    source_level: Literal["official", "registration", "authority", "platform", "user"]
    claim: str
    url: Optional[str] = None
    retrieved_at: Optional[str] = None
    availability: Literal["available", "unavailable", "error"] = "available"


class EvidenceEvent(BaseModel):
    type: Literal[
        "identification_started",
        "ocr_completed",
        "claims_extracted",
        "identification_completed",
        "clarification_needed",
        "evidence_search_started",
        "evidence_source_completed",
        "evidence_summary",
        "report_ready",
        "error",
    ]
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)


class ProductReport(BaseModel):
    product: ProductIdentification
    evidence: List[EvidenceItem] = Field(default_factory=list)
    conclusion: str
    limitations: List[str] = Field(default_factory=list)
