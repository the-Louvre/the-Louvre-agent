from __future__ import annotations

from typing import Protocol

from ..schemas import AideModelResult, RawMaskResult


class AigcDetector(Protocol):
    async def predict(self, image_bytes: bytes, request_id: str) -> AideModelResult: ...


class TamperLocalizer(Protocol):
    async def predict(self, image_bytes: bytes, request_id: str) -> RawMaskResult: ...
