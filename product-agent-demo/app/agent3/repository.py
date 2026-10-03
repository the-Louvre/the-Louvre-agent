from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .schemas import ReviewProgress, VisualReviewResult


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ReviewRecord:
    review_id: str
    task_id: str
    image_bytes: bytes
    input_metadata: Any
    created_at: datetime
    product_context: Optional[dict[str, Any]] = None
    status: str = "queued"
    stage: str = "queued"
    progress: float = 0.1
    updated_at: datetime = field(default_factory=utc_now)
    result: Optional[VisualReviewResult] = None


class ReviewRepository:
    def __init__(self, runtime_dir: Path):
        self.runtime_dir = runtime_dir
        self.records: dict[str, ReviewRecord] = {}
        self.lock = asyncio.Lock()

    async def add(self, record: ReviewRecord) -> None:
        async with self.lock:
            self.records[record.review_id] = record

    async def get(self, review_id: str) -> Optional[ReviewRecord]:
        async with self.lock:
            return self.records.get(review_id)

    async def update(self, record: ReviewRecord, *, status: Optional[str] = None, stage: Optional[str] = None, progress: Optional[float] = None) -> None:
        async with self.lock:
            if status is not None:
                record.status = status
            if stage is not None:
                record.stage = stage
            if progress is not None:
                record.progress = progress
            record.updated_at = utc_now()

    def task_dir(self, review_id: str) -> Path:
        path = self.runtime_dir / review_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, review_id: str, name: str, payload: dict[str, Any]) -> None:
        path = self.task_dir(review_id) / name
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        temp.replace(path)

    def progress(self, record: ReviewRecord) -> ReviewProgress:
        return ReviewProgress(review_id=record.review_id, task_id=record.task_id, status=record.status, stage=record.stage, progress=record.progress, created_at=record.created_at, updated_at=record.updated_at)
