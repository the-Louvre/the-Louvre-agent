from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import requests

from .schemas import EvidenceItem, ProductIdentification


class StreamableHttpMcpProvider:
    """Small Streamable HTTP MCP client for configured Bailian/custom MCP tools."""

    def __init__(self, url: Optional[str] = None, tool_name: Optional[str] = None,
                 api_key: Optional[str] = None, timeout_seconds: int = 20):
        self.url = url or os.getenv("BAILIAN_MCP_URL")
        self.tool_name = tool_name or os.getenv("BAILIAN_MCP_TOOL", "web_search")
        self.api_key = api_key if api_key is not None else os.getenv("DASHSCOPE_API_KEY")
        self.timeout_seconds = timeout_seconds
        self.session_id: Optional[str] = None

    def search(self, product: ProductIdentification) -> List[EvidenceItem]:
        if not self.url:
            return []
        if not self.api_key:
            raise RuntimeError("配置了 BAILIAN_MCP_URL，但未配置 DASHSCOPE_API_KEY")
        query = f"{product.brand} {product.product_name} 官方成分 登记 报告"
        result = self._call_tool(self.tool_name, {"query": query})
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        text = self._content_text(result)
        return [EvidenceItem(
            source_name=f"MCP/{self.tool_name}",
            source_level="authority",
            claim=text[:4000] or "MCP 返回为空",
            retrieved_at=now,
            availability="available",
        )]

    def _call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self._rpc("initialize", {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "loreal-product-agent", "version": "0.1.0"},
        })
        return self._rpc("tools/call", {"name": name, "arguments": arguments})

    def _rpc(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": method, "params": params}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        response = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        if response.headers.get("Mcp-Session-Id"):
            self.session_id = response.headers["Mcp-Session-Id"]
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            for line in response.text.splitlines():
                if line.startswith("data:"):
                    return json.loads(line.removeprefix("data:").strip())
            raise RuntimeError("MCP SSE 响应没有 data 事件")
        return response.json()

    @staticmethod
    def _content_text(payload: Dict[str, Any]) -> str:
        if payload.get("error"):
            raise RuntimeError(f"MCP 工具调用失败：{payload['error']}")
        content = payload.get("result", {}).get("content", [])
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])
        return "\n".join(parts)
