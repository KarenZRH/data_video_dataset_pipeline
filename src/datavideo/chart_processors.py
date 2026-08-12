"""Chart-type dispatch helpers."""

from __future__ import annotations

from typing import Any

CHART_FAMILIES: dict[str, str] = {
    "bar": "bar",
    "combined": "bar",
    "line": "line",
    "area": "line",
    "timeline": "line",
    "pie": "pie",
    "donut": "pie",
    "map": "map",
    "pictograph": "pictograph",
    "treemap": "treemap",
    "scatter": "scatter",
}

SUPPORTED_PROCESSORS: set[str] = {"bar", "line"}


def normalize_chart_type(value: Any) -> str:
    return str(value or "").strip().lower()


def resolve_processor(chart_type: Any) -> str:
    return CHART_FAMILIES.get(normalize_chart_type(chart_type), "unknown")


def detect_chart_type(declared: Any, recovered: Any) -> tuple[str, str, bool]:
    declared_type = normalize_chart_type(declared)
    recovered_type = normalize_chart_type(recovered)
    recovered_processor = resolve_processor(recovered_type)
    processor = recovered_processor if recovered_processor != "unknown" else resolve_processor(declared_type)
    consistent = (
        processor == "unknown"
        or not recovered_type
        or resolve_processor(declared_type) == processor
    )
    return processor, declared_type, consistent
