from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


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
            api_key=str(payload.get("api_key", "")),
            base_url=str(payload.get("base_url") or cls.base_url).rstrip("/"),
            model=str(payload.get("model") or cls.model),
            temperature=float(payload.get("temperature", cls.temperature)),
            timeout_seconds=int(payload.get("timeout_seconds", cls.timeout_seconds)),
        )

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"
