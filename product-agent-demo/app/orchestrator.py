from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, List, Optional, Protocol

from .schemas import EvidenceEvent, EvidenceItem, ProductIdentification, ProductReport
from .report import build_official_source_report


class VisionClient(Protocol):
    def identify(self, image_path: str) -> ProductIdentification: ...


class EvidenceProvider(Protocol):
    def search(self, product: ProductIdentification) -> List[EvidenceItem]: ...


class ProductAgentOrchestrator:
    def __init__(self, vision: VisionClient, evidence: EvidenceProvider):
        self.vision = vision
        self.evidence = evidence

    async def run(self, image_path: str) -> List[EvidenceEvent]:
        return [event async for event in self.stream(image_path)]

    async def stream(self, image_path: str) -> AsyncIterator[EvidenceEvent]:
        yield EvidenceEvent(type="identification_started", message="正在调用百炼视觉模型识别产品")
        try:
            product = await asyncio.to_thread(self.vision.identify, image_path)
            yield EvidenceEvent(
                type="identification_completed",
                message=f"识别完成：{product.brand} {product.product_name}",
                data={"product": product.model_dump(), "model_request_id": getattr(self.vision, "last_request_id", None)},
            )
            if product.ocr_text:
                yield EvidenceEvent(type="ocr_completed", message="OCR 文本已提取", data={"text": product.ocr_text})
            if product.visible_claims:
                yield EvidenceEvent(type="claims_extracted", message=f"已提取 {len(product.visible_claims)} 条可核验声明", data={"claims": product.visible_claims})
            if product.needs_more_image or product.confidence < 0.70:
                product.needs_more_image = True
                yield EvidenceEvent(
                    type="clarification_needed",
                    message="产品身份尚不确定，请补充清晰的正面包装、背面名称或条码图片；暂不检索候选产品。",
                    data={"product": product.model_dump(), "model_request_id": getattr(self.vision, "last_request_id", None)},
                )
                return
            yield EvidenceEvent(
                type="evidence_search_started",
                message="已锁定产品身份，开始按信源等级检索资料",
                data={"source_order": ["official", "registration", "authority", "platform"]},
            )
            items = await asyncio.to_thread(self.evidence.search, product)
            source_calls = []
            for error in getattr(self.evidence, "last_errors", []):
                yield EvidenceEvent(
                    type="evidence_source_completed",
                    message=f"外部信源不可用：{error}",
                    data={"availability": "error", "error": error},
                )
                source_calls.append({"source_id": "provider", "status": "error", "error": error})
            for item in items:
                yield EvidenceEvent(
                    type="evidence_source_completed",
                    message=f"已获取：{item.source_name}",
                    data={"evidence": item.model_dump()},
                )
                source_calls.append({"source_id": item.source_name, "source_name": item.source_name,
                                     "status": item.availability, "url": item.url})
            yield EvidenceEvent(
                type="evidence_summary",
                message=f"检索完成，获得 {len(items)} 条可引用证据",
                data={"count": len(items)},
            )
            official_report = build_official_source_report(
                product=product.model_dump(),
                claims=product.visible_claims,
                evidence=[item.model_dump() for item in items],
                source_calls=source_calls,
            )
            report = ProductReport(
                product=product,
                evidence=items,
                conclusion=(
                    "报告仅基于当前已获取证据生成；官方、登记与权威资料优先，"
                    "平台内容只作为辅助验证。"
                ),
                limitations=(
                    ["产品身份置信度不足，请补充正面包装、背面成分表或条码图片"]
                    if product.needs_more_image else []
                ),
            )
            yield EvidenceEvent(
                type="report_ready",
                message="产品核验报告已生成",
                data={"report": report.model_dump(), "official_source_report": official_report},
            )
        except Exception as exc:
            yield EvidenceEvent(type="error", message=str(exc))


class ConfiguredEvidenceProvider:
    """Evidence adapter boundary; real MCP/SDK providers plug in here."""

    def __init__(self, providers: Optional[List[EvidenceProvider]] = None):
        self.providers = providers or []
        self.last_errors: List[str] = []

    def search(self, product: ProductIdentification) -> List[EvidenceItem]:
        self.last_errors = []
        if not self.providers:
            return []
        # Keep external fan-out bounded at three concurrent providers.
        with ThreadPoolExecutor(max_workers=min(3, len(self.providers))) as pool:
            futures = [pool.submit(provider.search, product) for provider in self.providers]
            result: List[EvidenceItem] = []
            for provider, future in zip(self.providers, futures):
                try:
                    result.extend(future.result())
                except Exception as exc:
                    self.last_errors.append(f"{provider.__class__.__name__}: {exc}")
            return result
