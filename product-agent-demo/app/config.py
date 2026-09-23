from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict


def load_local_env() -> None:
    """Load the local .env file without requiring an extra dotenv dependency."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


load_local_env()


@dataclass
class ModelConfig:
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3-vl-32b-instruct"
    temperature: float = 0.0
    timeout_seconds: int = 20

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ModelConfig":
        return cls(
            api_key=str(payload.get("api_key") or os.getenv("DASHSCOPE_API_KEY", "")),
            base_url=str(payload.get("base_url") or cls.base_url).rstrip("/"),
            model=str(payload.get("model") or cls.model),
            temperature=float(payload.get("temperature", cls.temperature)),
            timeout_seconds=int(payload.get("timeout_seconds", cls.timeout_seconds)),
        )

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"
