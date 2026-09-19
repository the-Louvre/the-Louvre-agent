from __future__ import annotations

import base64
import json
import mimetypes
import os
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import ProductIdentification
from .config import ModelConfig


class MissingBailianKeyError(RuntimeError):
    pass


class BailianVisionClient:
    """Real DashScope client; no fallback or mock result is generated."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 base_url: Optional[str] = None, prompt: Optional[str] = None,
                 skill: Optional[str] = None, timeout_seconds: int = 20,
                 temperature: float = 0.0):
        self.config = ModelConfig.from_dict({
            "api_key": api_key if api_key is not None else os.getenv("DASHSCOPE_API_KEY", ""),
            "model": model or os.getenv("BAILIAN_VL_MODEL", "qwen3-vl-32b-instruct"),
            "base_url": base_url or os.getenv("BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "timeout_seconds": timeout_seconds,
            "temperature": temperature,
        })
        self.api_key = self.config.api_key
        self.model = self.config.model
        self.base_url = self.config.base_url
        self.prompt_override = prompt
        self.skill = skill or ""
        self.workspace_id = os.getenv("BAILIAN_WORKSPACE_ID")
        self.region = os.getenv("BAILIAN_REGION", "cn-beijing")
        self.last_request_id = None

    def _image_uri(self, image_path: str) -> str:
        path = Path(image_path)
        mime, _ = mimetypes.guess_type(path.name)
        if not mime or not mime.startswith("image/"):
            raise ValueError("仅支持可识别的图片格式")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def identify(self, image_path: str) -> ProductIdentification:
        if not self.api_key:
            raise MissingBailianKeyError(
                "未配置 DASHSCOPE_API_KEY。请在本机环境变量中配置百炼 API Key 后重试。"
            )
        if not Path(image_path).is_file():
            raise FileNotFoundError(image_path)

        prompt = self.prompt_override or """
你是美妆产品身份识别器。只根据图片中可见信息识别产品，不猜测功效，不把营销文案当成事实。
返回严格 JSON，不要 Markdown，不要解释：
{
  "brand": "品牌",
  "product_name": "完整产品名",
  "category": "护肤/洗护发/彩妆/香水身体护理/其他",
  "series": "系列，无法确认则 null",
  "specification": "规格或色号，无法确认则 null",
  "variant": "规格或色号，无法确认则 null",
  "barcode": "可见条码，无法确认则 null",
  "candidate_products": [],
  "visual_evidence": ["包装文字、瓶型、色号等可见证据"],
  "uncertainties": [],
  "confidence": 0.0,
  "evidence": ["包装文字、瓶型、色号等可见证据"],
  "ocr_text": "图片中可读文字",
  "visible_claims": [{"claim_id":"c1","text":"图片中的营销声明","type":"efficacy"}],
  "needs_more_image": false
}
若无法确定具体产品，confidence 必须低于 0.70，并将 needs_more_image 设为 true。
""".strip()
        if self.skill:
            prompt = f"{prompt}\n\n当前 Skill 约束：\n{self.skill}"
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self._image_uri(image_path)}},
                    {"type": "text", "text": "请识别这张产品图片中的具体商品，并提取图片中可见的营销声明。"},
                ],
            },
        ]
        response = requests.post(
            self.config.chat_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages, "temperature": self.config.temperature,
                  "response_format": {"type": "json_object"}},
            timeout=self.config.timeout_seconds,
        )
        if response.status_code // 100 != 2:
            raise RuntimeError(f"模型调用失败: HTTP {response.status_code}: {response.text[:500]}")
        payload = response.json()
        self.last_request_id = response.headers.get("x-request-id") or payload.get("id") or payload.get("request_id")
        text = self._extract_openai_text(payload)
        return ProductIdentification.model_validate(self._parse_json(text))

    def complete_text(self, prompt: str, system: str = "") -> str:
        if not self.api_key:
            raise MissingBailianKeyError("未配置 API Key")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = requests.post(
            self.config.chat_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages, "temperature": self.config.temperature},
            timeout=self.config.timeout_seconds,
        )
        if response.status_code // 100 != 2:
            raise RuntimeError(f"模型调用失败: HTTP {response.status_code}: {response.text[:500]}")
        self.last_request_id = response.headers.get("x-request-id") or response.json().get("id")
        return self._extract_openai_text(response.json())

    @staticmethod
    def _extract_text(payload: Dict[str, Any]) -> str:
        choices = payload.get("output", {}).get("choices", [])
        if not choices:
            raise RuntimeError("百炼返回中没有 choices")
        content = choices[0].get("message", {}).get("content", [])
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                return item["text"]
        raise RuntimeError("百炼返回中没有文本识别结果")

    @staticmethod
    def _extract_openai_text(payload: Dict[str, Any]) -> str:
        choices = payload.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        return BailianVisionClient._extract_text(payload)

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").removeprefix("json").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"百炼未返回合法 JSON: {text[:300]}") from exc
