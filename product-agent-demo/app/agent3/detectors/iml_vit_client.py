from __future__ import annotations

import base64
import io
import time

import httpx
from PIL import Image

from ..schemas import ModelError, RawMaskResult


def _decode_probability_mask(encoded: str) -> tuple[int, int, list[list[float]]]:
    raw = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        if image.mode not in {"I;16", "I", "L"}:
            raise ValueError("worker mask must be a grayscale PNG")
        width, height = image.size
        values = list(image.convert("I").getdata())
    scale = 65535.0 if max(values, default=0) > 255 else 255.0
    mask = [
        [max(0.0, min(1.0, values[row * width + column] / scale)) for column in range(width)]
        for row in range(height)
    ]
    return width, height, mask


class RemoteImlVitLocalizer:
    def __init__(self, base_url: str, timeout_seconds: float = 30.0, model_version: str = "iml-vit-v1"):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.model_version = model_version

    async def predict(self, image_bytes: bytes, request_id: str) -> RawMaskResult:
        started = time.perf_counter()
        try:
            timeout = httpx.Timeout(self.timeout_seconds)
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                response = await client.post(
                    f"{self.base_url}/internal/v1/predict",
                    files={"image": ("input.bin", image_bytes, "application/octet-stream")},
                    data={"request_id": request_id},
                )
                response.raise_for_status()
                payload = response.json()
            if payload.get("status") != "ok":
                error = payload.get("error") or {}
                return RawMaskResult(
                    status=payload.get("status", "error"),
                    model_version=payload.get("model_version", self.model_version),
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    error=ModelError(
                        code=error.get("code", "worker_error"),
                        message=error.get("message", "IML-ViT Worker returned an error"),
                        retryable=bool(error.get("retryable", True)),
                    ),
                )
            width, height, mask = _decode_probability_mask(payload["probability_mask"])
            if width != payload.get("mask_width") or height != payload.get("mask_height"):
                raise ValueError("worker mask dimensions do not match payload")
            return RawMaskResult(
                status="ok",
                model_version=payload.get("model_version", self.model_version),
                width=width,
                height=height,
                probability_mask=mask,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        except httpx.TimeoutException as exc:
            raise TimeoutError("IML-ViT Worker request timed out") from exc
        except Exception as exc:
            raise RuntimeError(f"IML-ViT Worker request failed: {str(exc)[:160]}") from exc
