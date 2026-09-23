from __future__ import annotations

import base64
import hashlib
import io
import uuid
from datetime import datetime, timezone
from typing import Optional

from PIL import Image, ImageDraw

from .config import CONFIG, Agent3Config
from .detectors.aide_client import RemoteAideDetector
from .detectors.fake import FakeAideDetector, FakeImlVitLocalizer
from .detectors.iml_vit_client import RemoteImlVitLocalizer
from .fusion import decide
from .postprocess import summarize_mask
from .repository import ReviewRecord, ReviewRepository, utc_now
from .schemas import (
    AideModelResult, ArtifactLinks, InputMetadata, ImlVitModelResult, ModelError,
    ModelResults, RawMaskResult, ReviewAccepted, TaskError, VisualEvidence,
    VisualRegion, VisualReviewResult,
)


def _review_id() -> str:
    return "vr_" + uuid.uuid4().hex[:24]


def _mime_and_image(data: bytes):
    try:
        image = Image.open(io.BytesIO(data))
        fmt = (image.format or "").upper()
        mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}.get(fmt)
        if mime is None:
            raise ValueError("unsupported media type")
        image.verify()
        image = Image.open(io.BytesIO(data))
        width, height = image.size
    except Exception as exc:
        raise ValueError("invalid image") from exc
    if width < 64 or height < 64 or width > 12000 or height > 12000 or width * height > 40_000_000:
        raise OverflowError("image dimensions exceed limits")
    return mime, width, height, image


class VisualReviewService:
    def __init__(self, repository: Optional[ReviewRepository] = None, config: Agent3Config = CONFIG, aide=None, iml_vit=None):
        self.config = config
        self.repository = repository or ReviewRepository(config.runtime_dir)
        self.aide = aide or (
            RemoteAideDetector(
                base_url=config.aide_worker_url,
                timeout_seconds=config.model_timeout_seconds,
                model_version=config.aide_model_version,
            )
            if config.aide_worker_url
            else FakeAideDetector()
        )
        self.iml_vit = iml_vit or (
            RemoteImlVitLocalizer(
                base_url=config.iml_vit_worker_url,
                timeout_seconds=config.model_timeout_seconds,
                model_version=config.iml_vit_model_version,
            )
            if config.iml_vit_worker_url
            else FakeImlVitLocalizer()
        )
        self.queue_count = 0

    async def create_review(self, task_id: str, image_bytes: bytes, product_context: Optional[dict] = None) -> ReviewAccepted:
        if len(image_bytes) > self.config.max_upload_bytes:
            raise OverflowError("image too large")
        mime, width, height, image = _mime_and_image(image_bytes)
        review_id, created = _review_id(), utc_now()
        metadata = InputMetadata(sha256=hashlib.sha256(image_bytes).hexdigest(), mime_type=mime, width=width, height=height, animated=False)
        record = ReviewRecord(review_id, task_id, image_bytes, metadata, created, product_context)
        await self.repository.add(record)
        self.repository.task_dir(review_id).joinpath("original.bin").write_bytes(image_bytes)
        self.repository.write_json(review_id, "input.json", {"task_id": task_id, "input": metadata.model_dump(mode="json")})
        return ReviewAccepted(review_id=review_id, task_id=task_id, status="queued", created_at=created, status_url=f"/api/v1/agent3/reviews/{review_id}")

    async def process_review(self, review_id: str) -> None:
        record = await self.repository.get(review_id)
        if record is None:
            return
        await self.repository.update(record, status="running", stage="model_inference", progress=0.2)
        aide_result, raw_mask = await self._run_models(record)
        await self.repository.update(record, stage="postprocessing", progress=0.75)
        postprocess_error = None
        try:
            iml_result, regions, final_mask = self._postprocess(raw_mask, record.input_metadata.width, record.input_metadata.height)
        except Exception as exc:
            postprocess_error = TaskError(component="postprocessing", code="postprocessing_failed", message=str(exc)[:180], retryable=True)
            iml_result = ImlVitModelResult(status="error", model_version=raw_mask.model_version, error=ModelError(code="postprocessing_failed", message="Mask 后处理失败", retryable=True))
            regions, final_mask = [], [[0.0]]
        await self.repository.update(record, stage="fusion", progress=0.85)
        decision = decide(aide_result, iml_result, self.config)
        storage_error = None
        try:
            artifacts = self._write_artifacts(record, final_mask)
        except Exception as exc:
            storage_error = TaskError(component="storage", code="artifact_write_failed", message=str(exc)[:180], retryable=True)
            artifacts = ArtifactLinks()
        await self.repository.update(record, stage="storing_artifacts", progress=0.95)
        errors = [TaskError(component="aide", code=aide_result.error.code, message=aide_result.error.message, retryable=aide_result.error.retryable) for _ in [aide_result] if aide_result.error]
        errors += [TaskError(component="iml_vit", code=iml_result.error.code, message=iml_result.error.message, retryable=iml_result.error.retryable) for _ in [iml_result] if iml_result.error]
        if postprocess_error:
            errors.append(postprocess_error)
        if storage_error:
            errors.append(storage_error)
        evidence = self._evidence(aide_result, iml_result, decision, bool(errors))
        both_failed = aide_result.status != "ok" and iml_result.status != "ok"
        result = VisualReviewResult(
            review_id=record.review_id, task_id=record.task_id, status="failed" if both_failed else ("completed" if not errors else "degraded"), source_type=decision.source_type, risk=decision.risk, confidence=max(0.0, min(1.0, decision.confidence)), input=record.input_metadata,
            model_results=ModelResults(aide=aide_result, iml_vit=iml_result), regions=regions, artifacts=artifacts, evidence=evidence,
            uncertainties=["模型判断属于辅助证据，不能单独证明图片来源", "篡改区域不等同于确定的AI生成区域", "截图、压缩和二次编辑可能影响检测结果"], errors=errors,
            created_at=record.created_at, completed_at=utc_now(), duration_ms=int((utc_now() - record.created_at).total_seconds() * 1000),
        )
        record.result = result
        await self.repository.update(record, status=result.status, stage="completed", progress=1.0)
        self.repository.write_json(review_id, "result.json", result.model_dump(mode="json"))

    async def _run_models(self, record):
        import asyncio
        async def safe(detector, fallback):
            try:
                return await detector.predict(record.image_bytes, record.review_id)
            except TimeoutError:
                return fallback("timeout", "model_timeout", "模型推理超时", True)
            except Exception as exc:
                return fallback("error", "inference_failed", str(exc)[:180], True)
        aide, mask = await asyncio.gather(
            safe(self.aide, lambda status, code, message, retryable: AideModelResult(status=status, model_version=self.config.aide_model_version if self.config.aide_worker_url else "aide-fake-v1", error=ModelError(code=code, message=message, retryable=retryable))),
            safe(self.iml_vit, lambda status, code, message, retryable: RawMaskResult(status=status, model_version="iml-vit-fake-v1", error=ModelError(code=code, message=message, retryable=retryable))),
        )
        return aide, mask

    def _postprocess(self, raw: RawMaskResult, width: int, height: int):
        if raw.status != "ok" or raw.probability_mask is None:
            error = raw.error or ModelError(code="mask_unavailable", message="Mask unavailable", retryable=True)
            return ImlVitModelResult(status=raw.status, model_version=raw.model_version, latency_ms=raw.latency_ms, error=error), [], [[0.0]]
        if raw.width != len(raw.probability_mask[0]) or raw.height != len(raw.probability_mask):
            raise ValueError("mask dimensions do not match payload")
        tamper, area, found, mask = summarize_mask(raw.probability_mask)
        sx, sy = width / raw.width, height / raw.height
        regions = [VisualRegion(region_id=f"region_{i:03d}", type="suspected_manipulation", bbox=(round(x * sx), round(y * sy), round(w * sx), round(h * sy)), normalized_bbox=(round(x / raw.width, 6), round(y / raw.height, 6), round(w / raw.width, 6), round(h / raw.height, 6)), score=round(region.score, 6), area_ratio=round(region.area_ratio, 6), reason_code="high_tamper_probability") for i, region in enumerate(found, 1) for x, y, w, h in [region.bbox]]
        return ImlVitModelResult(status="ok", tamper_probability=tamper, mask_area_ratio=area, model_version=raw.model_version, latency_ms=raw.latency_ms), regions, mask

    def _write_artifacts(self, record, mask):
        directory = self.repository.task_dir(record.review_id) / "artifacts"
        directory.mkdir(parents=True, exist_ok=True)
        image = Image.open(io.BytesIO(record.image_bytes)).convert("RGBA")
        if len(mask) == 1 and len(mask[0]) == 1:
            mask = [[0.0 for _ in range(image.width)] for _ in range(image.height)]
        probability = Image.new("L", (len(mask[0]), len(mask)), 0)
        probability.putdata([round(value * 255) for row in mask for value in row])
        binary = probability.point(lambda value: 255 if value >= 128 else 0)
        probability = probability.resize(image.size, Image.Resampling.NEAREST)
        binary = binary.resize(image.size, Image.Resampling.NEAREST)
        overlay = image.copy()
        red = Image.new("RGBA", image.size, (220, 30, 50, 0))
        red.putalpha(binary.point(lambda value: round(value * 0.45)))
        overlay = Image.alpha_composite(overlay, red).convert("RGB")
        probability.save(directory / "probability-mask.png")
        binary.save(directory / "binary-mask.png")
        overlay.save(directory / "overlay.png")
        prefix = f"/api/v1/agent3/reviews/{record.review_id}/artifacts/"
        return ArtifactLinks(probability_mask_url=prefix + "probability-mask.png", binary_mask_url=prefix + "binary-mask.png", overlay_url=prefix + "overlay.png")

    def _evidence(self, aide, iml, decision, degraded):
        evidence = []
        if aide.status == "ok":
            evidence.append(VisualEvidence(evidence_id="ev_aide", level="L6", source="AIDE", finding_code="global_aigc_probability_high" if (aide.ai_probability or 0) >= self.config.aide_high else "global_aigc_probability_low", finding="整图AIGC概率较高" if (aide.ai_probability or 0) >= self.config.aide_high else "整图AIGC概率未达到高风险阈值", confidence=aide.ai_probability or 0.0))
        if iml.status == "ok":
            evidence.append(VisualEvidence(evidence_id="ev_iml_vit", level="L6", source="IML-ViT", finding_code="localized_manipulation_detected" if (iml.mask_area_ratio or 0) >= self.config.local_area_min else "no_localized_manipulation_detected", finding="检测到局部疑似篡改区域" if (iml.mask_area_ratio or 0) >= self.config.local_area_min else "未检测到达到阈值的局部区域", confidence=iml.tamper_probability or 0.0))
        if degraded:
            evidence.append(VisualEvidence(evidence_id="ev_degraded", level="L6", source="fusion-rules-v1", finding_code="model_result_degraded", finding="一个模型不可用，当前结果为降级结果", confidence=0.5))
        return evidence

    async def get(self, review_id: str):
        record = await self.repository.get(review_id)
        if record is None:
            return None
        return record.result or self.repository.progress(record)
