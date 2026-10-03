from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Agent3Config:
    runtime_dir: Path = Path(os.getenv("AGENT3_RUNTIME_DIR", "runtime/agent3"))
    max_upload_bytes: int = int(os.getenv("AGENT3_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    max_queue_size: int = int(os.getenv("AGENT3_MAX_QUEUE_SIZE", "16"))
    concurrency: int = int(os.getenv("AGENT3_CONCURRENCY", "1"))
    aide_high: float = float(os.getenv("AGENT3_AIDE_HIGH", "0.75"))
    aide_low: float = float(os.getenv("AGENT3_AIDE_LOW", "0.35"))
    tamper_high: float = float(os.getenv("AGENT3_TAMPER_HIGH", "0.60"))
    local_area_min: float = float(os.getenv("AGENT3_LOCAL_AREA_MIN", "0.02"))
    global_area_min: float = float(os.getenv("AGENT3_GLOBAL_AREA_MIN", "0.60"))
    aide_worker_url: str = os.getenv("AIDE_WORKER_URL", "").strip()
    iml_vit_worker_url: str = os.getenv("IML_VIT_WORKER_URL", "").strip()
    model_timeout_seconds: float = float(os.getenv("AGENT3_MODEL_TIMEOUT_SECONDS", "30"))
    aide_model_version: str = os.getenv("AIDE_MODEL_VERSION", "aide-genimage-v1")
    iml_vit_model_version: str = os.getenv("IML_VIT_MODEL_VERSION", "iml-vit-v1")


CONFIG = Agent3Config()
