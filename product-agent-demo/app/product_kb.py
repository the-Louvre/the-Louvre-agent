from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import EvidenceItem, ProductIdentification


class ProductKbProvider:
    """Read-only provider for a team-maintained JSON Product KB.

    The file is an input data source, not generated demo data. If it is absent,
    this provider returns no evidence so the report cannot silently invent facts.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or os.getenv("PRODUCT_KB_PATH", "product-kb.json"))

    def search(self, product: ProductIdentification) -> List[EvidenceItem]:
        if not self.path.is_file():
            return []
        records = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("PRODUCT_KB_PATH 必须是 JSON 数组")
        matched: List[EvidenceItem] = []
        for record in records:
            if not isinstance(record, dict) or not self._matches(record, product):
                continue
            product_id = str(record.get("product_id", ""))
            name = str(record.get("product_name", product.product_name))
            matched.append(EvidenceItem(
                source_name="Product KB / official product record",
                source_level="official",
                claim=f"产品记录 {product_id}: {record.get('brand', '')} {name}",
                url=record.get("official_url"),
                availability="available",
            ))
            if record.get("ingredients_url"):
                matched.append(EvidenceItem(
                    source_name="Product KB / ingredient record",
                    source_level="official",
                    claim=f"成分表入口：{name}",
                    url=record["ingredients_url"],
                    availability="available",
                ))
        return matched

    @staticmethod
    def _norm(value: Any) -> str:
        return re.sub(r"[\s\-_/（）()]+", "", str(value or "")).lower()

    def _matches(self, record: Dict[str, Any], product: ProductIdentification) -> bool:
        target_brand = self._norm(product.brand)
        target_name = self._norm(product.product_name)
        brand = self._norm(record.get("brand"))
        names = [record.get("product_name", "")] + list(record.get("aliases", []) or [])
        return brand == target_brand and any(
            target_name == self._norm(name)
            or target_name in self._norm(name)
            or self._norm(name) in target_name
            for name in names
        )
