"""Chart-type dispatch skeleton.

Every chart family gets its own processor; the dispatcher routes a clip to
the right one after validating the chart type. Only the ``bar`` processor is
implemented today; the other families are registered but routed to an
"unsupported" path so the pipeline still writes a report (flagged for
review) instead of silently treating e.g. a line chart as bars.

Adding a new family means implementing a processor (geometry detection,
value estimation, semantic render) and flipping its flag in
``SUPPORTED_PROCESSORS``; the shared pipeline steps (keyframes, narration,
title resolution, dataset packaging) stay untouched.
"""

from __future__ import annotations

from typing import Any


# Chart family -> processor name. Sub-type detection (horizontal vs vertical
# bars, multi-line vs single-line, ...) happens inside each processor.
CHART_FAMILIES: dict[str, str] = {
    "bar": "bar",
    "combined": "bar",  # combined clips frequently contain bar charts
    "line": "line",
    "area": "line",
    "timeline": "line",
    "pie": "pie",
    "donut": "pie",
    "map": "map",
    "pictograph": "pictograph",
    "treemap": "treemap",
    "scatter": "scatter",
    "sankey": "sankey",
}

SUPPORTED_PROCESSORS: set[str] = {"bar", "line"}


def normalize_chart_type(chart_type: Any) -> str:
    return str(chart_type or "").strip().lower()


def resolve_processor(chart_type: Any) -> str:
    """Map a declared/recovered chart type to its processor name."""
    return CHART_FAMILIES.get(normalize_chart_type(chart_type), "unknown")


def detect_chart_type(declared: Any, recovered: Any) -> tuple[str, str, bool]:
    """Resolve the processor and validate the declared vs recovered type.

    Returns ``(processor, declared_type, consistent)``. The recovered
    (VLM-read) type is authoritative for dispatch when it names a known
    family; a mismatch between the CSV declaration and the frame verdict is
    reported so the clip can be flagged for review.
    """
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
