from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    bbox: tuple[int, int, int, int]
    score: float
    area_ratio: float


def summarize_mask(mask: list[list[float]], threshold: float = 0.50, min_area_ratio: float = 0.005, max_regions: int = 20):
    if not mask or not mask[0]:
        return 0.0, 0.0, [], []
    height, width = len(mask), len(mask[0])
    if any(len(row) != width for row in mask):
        raise ValueError("mask rows have inconsistent widths")
    if any(value < 0.0 or value > 1.0 for row in mask for value in row):
        raise ValueError("mask probability must be in [0,1]")
    binary = {(x, y) for y, row in enumerate(mask) for x, value in enumerate(row) if value >= threshold}
    min_pixels = width * height * min_area_ratio
    regions: list[Region] = []
    kept: set[tuple[int, int]] = set()
    remaining = set(binary)
    while remaining:
        seed = remaining.pop()
        component = {seed}
        queue = deque([seed])
        while queue:
            x, y = queue.popleft()
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1),
                           (x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1), (x + 1, y + 1)):
                point = (nx, ny)
                if point in remaining:
                    remaining.remove(point)
                    component.add(point)
                    queue.append(point)
        if len(component) < min_pixels:
            continue
        xs, ys = zip(*component)
        score = sum(mask[y][x] for x, y in component) / len(component)
        regions.append(Region((min(xs), min(ys), max(xs) - min(xs) + 1, max(ys) - min(ys) + 1), score, len(component) / (width * height)))
        kept.update(component)
    regions.sort(key=lambda item: item.score, reverse=True)
    regions = regions[:max_regions]
    kept = {point for region in regions for point in _points_for_bbox(region.bbox)}
    final_mask = [[1.0 if (x, y) in kept else 0.0 for x in range(width)] for y in range(height)]
    values = sorted(value for row in mask for value in row)
    top_count = max(1, (len(values) + 19) // 20)
    tamper_probability = sum(values[-top_count:]) / top_count
    area_ratio = sum(sum(row) for row in final_mask) / (width * height)
    return tamper_probability, area_ratio, regions, final_mask


def _points_for_bbox(bbox: tuple[int, int, int, int]):
    x, y, width, height = bbox
    return {(px, py) for py in range(y, y + height) for px in range(x, x + width)}
