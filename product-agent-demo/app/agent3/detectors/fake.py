from __future__ import annotations

import time

from ..schemas import AideModelResult, RawMaskResult


class FakeAideDetector:
    model_version = "aide-fake-v1"

    async def predict(self, image_bytes: bytes, request_id: str) -> AideModelResult:
        started = time.perf_counter()
        # Deterministic fixture: the first byte selects a useful fusion branch.
        marker = image_bytes[0] if image_bytes else 0
        ai_probability = {0x01: 0.90, 0x02: 0.90, 0x03: 0.10}.get(marker, 0.10)
        return AideModelResult(
            status="ok",
            ai_probability=ai_probability,
            real_probability=1.0 - ai_probability,
            model_version=self.model_version,
            latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
        )


class FakeImlVitLocalizer:
    model_version = "iml-vit-fake-v1"

    async def predict(self, image_bytes: bytes, request_id: str) -> RawMaskResult:
        started = time.perf_counter()
        marker = image_bytes[0] if image_bytes else 0
        # A small 10x10 probability map is enough to exercise post-processing.
        mask = [[0.0 for _ in range(10)] for _ in range(10)]
        if marker in (0x01, 0x02):
            for y in range(2, 8):
                for x in range(2, 8):
                    mask[y][x] = 0.9
        return RawMaskResult(
            status="ok",
            model_version=self.model_version,
            width=10,
            height=10,
            probability_mask=mask,
            latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
        )
