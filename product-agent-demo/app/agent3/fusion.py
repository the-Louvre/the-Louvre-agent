from __future__ import annotations

from dataclasses import dataclass

from .config import Agent3Config
from .schemas import AideModelResult, ImlVitModelResult


@dataclass(frozen=True)
class FusionDecision:
    source_type: str
    risk: str
    confidence: float


def decide(aide: AideModelResult, iml: ImlVitModelResult, config: Agent3Config) -> FusionDecision:
    if aide.status == "ok" and iml.status == "ok":
        ai, tamper, area = aide.ai_probability or 0.0, iml.tamper_probability or 0.0, iml.mask_area_ratio or 0.0
        if ai >= config.aide_high and area >= config.global_area_min:
            return FusionDecision("likely_ai_generated", "high", ai)
        if ai >= config.aide_high and tamper >= config.tamper_high and area >= config.local_area_min:
            return FusionDecision("suspected_ai_assisted_edit", "high", 0.6 * ai + 0.4 * tamper)
        if ai >= config.aide_high:
            return FusionDecision("likely_ai_generated", "medium", ai)
        if ai <= config.aide_low and tamper >= config.tamper_high and area >= config.local_area_min:
            return FusionDecision("edited_photo", "medium", 0.7 * tamper + 0.3 * (1.0 - ai))
        if ai <= config.aide_low and tamper < config.tamper_high and area < config.local_area_min:
            return FusionDecision("no_anomaly_detected", "low", 0.5 * (1.0 - ai) + 0.5 * (1.0 - tamper))
        return FusionDecision("inconclusive", "unknown", 0.5)
    if aide.status == "ok":
        ai = aide.ai_probability or 0.0
        return FusionDecision("likely_ai_generated", "medium", ai) if ai >= config.aide_high else FusionDecision("inconclusive", "unknown", 0.5)
    if iml.status == "ok" and (iml.tamper_probability or 0.0) >= config.tamper_high and (iml.mask_area_ratio or 0.0) >= config.local_area_min:
        tamper = iml.tamper_probability or 0.0
        return FusionDecision("edited_photo", "medium", tamper)
    return FusionDecision("inconclusive", "unknown", 0.5)
