from __future__ import annotations

import math
import time

import httpx

from ..schemas import AideModelResult, ModelError


class RemoteAideDetector:
    """Client for the isolated AIDE global-AIGC worker."""

    def __init__(self, base_url: str, timeout_seconds: float = 30.0, model_version: str = "aide-genimage-v1"):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.model_version = model_version

    async def predict(self, image_bytes: bytes, request_id: str) -> AideModelResult:
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
                return AideModelResult(
                    status=payload.get("status", "error"),
                    model_version=payload.get("model_version", self.model_version),
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    error=ModelError(
                        code=error.get("code", "worker_error"),
                        message=error.get("message", "AIDE Worker returned an error"),
                        retryable=bool(error.get("retryable", True)),
                    ),
                )
            ai_probability = float(payload["ai_probability"])
            real_probability = float(payload["real_probability"])
            if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (ai_probability, real_probability)):
                raise ValueError("worker returned invalid probabilities")
            return AideModelResult(
                status="ok",
                ai_probability=ai_probability,
                real_probability=real_probability,
                model_version=payload.get("model_version", self.model_version),
                latency_ms=int(payload.get("latency_ms", int((time.perf_counter() - started) * 1000))),
            )
        except httpx.TimeoutException as exc:
            raise TimeoutError("AIDE Worker request timed out") from exc
        except Exception as exc:
            raise RuntimeError(f"AIDE Worker request failed: {str(exc)[:160]}") from exc
