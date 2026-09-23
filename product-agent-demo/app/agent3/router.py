from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from pydantic import ValidationError
from fastapi.responses import FileResponse

from .config import CONFIG
from .repository import ReviewRepository
from .schemas import HealthModel, HealthResult, ProductContext, ReviewAccepted, ReviewProgress, VisualReviewResult
from .service import VisualReviewService


router = APIRouter(prefix="/api/v1/agent3", tags=["agent3"])
repository = ReviewRepository(CONFIG.runtime_dir)
service = VisualReviewService(repository=repository, config=CONFIG)
ARTIFACTS = {"probability-mask.png", "binary-mask.png", "overlay.png"}


@router.post("/reviews", response_model=ReviewAccepted, status_code=202)
async def create_review(background_tasks: BackgroundTasks, image: UploadFile = File(...), task_id: str = Form(...), product_context: str | None = Form(default=None)):
    if not task_id or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", task_id) is None:
        raise HTTPException(status_code=422, detail={"code": "validation_error", "message": "task_id 格式不合法"})
    data = await image.read(CONFIG.max_upload_bytes + 1)
    try:
        context = ProductContext.model_validate_json(product_context).model_dump(mode="json") if product_context else None
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail={"code": "validation_error", "message": "product_context 格式不合法"}) from exc
    try:
        accepted = await service.create_review(task_id, data, context)
    except OverflowError as exc:
        raise HTTPException(status_code=413, detail={"code": "image_too_large", "message": str(exc)}) from exc
    except ValueError as exc:
        message = "图片格式不支持或无法解码"
        raise HTTPException(status_code=415 if "media type" in str(exc) else 400, detail={"code": "unsupported_media_type" if "media type" in str(exc) else "invalid_image", "message": message}) from exc
    background_tasks.add_task(service.process_review, accepted.review_id)
    return accepted


@router.get("/reviews/{review_id}", response_model=ReviewProgress | VisualReviewResult)
async def get_review(review_id: str):
    result = await service.get(review_id)
    if result is None:
        raise HTTPException(status_code=404, detail={"code": "review_not_found", "message": "review 不存在"})
    return result


@router.get("/reviews/{review_id}/artifacts/{artifact_name}")
async def get_artifact(review_id: str, artifact_name: str):
    if artifact_name not in ARTIFACTS:
        raise HTTPException(status_code=404, detail={"code": "artifact_not_found", "message": "artifact 不存在"})
    record = await service.repository.get(review_id)
    if record is None:
        raise HTTPException(status_code=404, detail={"code": "review_not_found", "message": "review 不存在"})
    path = service.repository.task_dir(review_id) / "artifacts" / artifact_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail={"code": "artifact_not_found", "message": "artifact 尚未生成"})
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, no-store"})


@router.get("/health", response_model=HealthResult)
async def health():
    aide_version = CONFIG.aide_model_version if CONFIG.aide_worker_url else "aide-fake-v1"
    iml_version = CONFIG.iml_vit_model_version if CONFIG.iml_vit_worker_url else "iml-vit-fake-v1"
    return HealthResult(status="ready", models={"aide": HealthModel(status="ready", model_version=aide_version), "iml_vit": HealthModel(status="ready", model_version=iml_version)})
