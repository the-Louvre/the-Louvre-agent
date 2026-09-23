from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


Probability = Field(ge=0.0, le=1.0)


class ProductContext(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    product_id: Optional[str] = Field(default=None, max_length=128)
    brand: Optional[str] = Field(default=None, max_length=128)
    product_name: Optional[str] = Field(default=None, max_length=256)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reference_image_urls: list[str] = Field(default_factory=list, max_length=5)


class InputMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    mime_type: Literal["image/png", "image/jpeg", "image/webp"]
    width: int = Field(ge=64, le=12000)
    height: int = Field(ge=64, le=12000)
    animated: bool = False


class ModelError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=240)
    retryable: bool = False


class AideModelResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "error", "timeout", "unavailable"]
    ai_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    real_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    model_version: str = Field(min_length=1, max_length=80)
    latency_ms: Optional[int] = Field(default=None, ge=0)
    error: Optional[ModelError] = None


class ImlVitModelResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "error", "timeout", "unavailable"]
    tamper_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    mask_area_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    model_version: str = Field(min_length=1, max_length=80)
    latency_ms: Optional[int] = Field(default=None, ge=0)
    error: Optional[ModelError] = None


class VisualRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region_id: str = Field(min_length=1, max_length=64)
    type: Literal["suspected_manipulation"]
    bbox: tuple[int, int, int, int]
    normalized_bbox: tuple[float, float, float, float]
    score: float = Field(ge=0.0, le=1.0)
    area_ratio: float = Field(ge=0.0, le=1.0)
    reason_code: Literal["high_tamper_probability"]


class ArtifactLinks(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probability_mask_url: Optional[str] = None
    binary_mask_url: Optional[str] = None
    overlay_url: Optional[str] = None


class VisualEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1, max_length=64)
    level: Literal["L6"]
    source: Literal["AIDE", "IML-ViT", "fusion-rules-v1"]
    finding_code: Literal[
        "global_aigc_probability_high",
        "global_aigc_probability_low",
        "localized_manipulation_detected",
        "no_localized_manipulation_detected",
        "model_results_conflict",
        "model_result_degraded",
    ]
    finding: str = Field(min_length=1, max_length=240)
    confidence: float = Field(ge=0.0, le=1.0)


class TaskError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component: Literal["validation", "aide", "iml_vit", "postprocessing", "fusion", "storage", "system"]
    code: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=240)
    retryable: bool = False


class ModelResults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aide: AideModelResult
    iml_vit: ImlVitModelResult


class VisualReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = "1.0.0"
    review_id: str = Field(min_length=23, max_length=35, pattern=r"^vr_[A-Za-z0-9]{20,32}$")
    task_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    status: Literal["completed", "degraded", "failed"]
    stage: Literal["completed"] = "completed"
    source_type: Literal[
        "likely_ai_generated",
        "suspected_ai_assisted_edit",
        "edited_photo",
        "no_anomaly_detected",
        "inconclusive",
    ]
    risk: Literal["high", "medium", "low", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)
    input: InputMetadata
    model_results: ModelResults
    regions: list[VisualRegion] = Field(default_factory=list, max_length=20)
    artifacts: ArtifactLinks
    evidence: list[VisualEvidence] = Field(default_factory=list, max_length=20)
    uncertainties: list[str] = Field(default_factory=list, max_length=20)
    errors: list[TaskError] = Field(default_factory=list, max_length=20)
    fusion_version: Literal["fusion-rules-v1"] = "fusion-rules-v1"
    created_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = Field(default=None, ge=0)


class ReviewAccepted(BaseModel):
    review_id: str
    task_id: str
    status: Literal["queued"]
    created_at: datetime
    status_url: str


class ReviewProgress(BaseModel):
    review_id: str
    task_id: str
    status: Literal["queued", "running"]
    stage: Literal["validating", "queued", "model_inference", "postprocessing", "fusion", "storing_artifacts"]
    progress: float = Field(ge=0.0, le=1.0)
    created_at: datetime
    updated_at: datetime


class HealthModel(BaseModel):
    status: Literal["ready", "degraded", "unavailable"]
    model_version: str
    latency_ms: Optional[int] = None


class HealthResult(BaseModel):
    status: Literal["ready", "degraded", "unavailable"]
    models: dict[str, HealthModel]
    fusion_version: Literal["fusion-rules-v1"] = "fusion-rules-v1"


class RawMaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "error", "timeout", "unavailable"]
    model_version: str
    width: Optional[int] = Field(default=None, ge=1)
    height: Optional[int] = Field(default=None, ge=1)
    probability_mask: Optional[list[list[float]]] = None
    latency_ms: Optional[int] = Field(default=None, ge=0)
    error: Optional[ModelError] = None
