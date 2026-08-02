from __future__ import annotations

import json
import re
from typing import Any

from datavideo.qwen_vl import QwenVLClient


KEYFRAME_SCORE_PROMPT = """You are scoring one video frame as a keyframe for a data-video clip.

The goal is to choose one frame that best represents the chart as a static data visualization for SVG tracing and data extraction.

Clip context:
__CLIP_CONTEXT__

The chart may be any of these types: map, bar, line, donut, area, pictograph, pie, timeline, treemap, scatter, sankey, or combined.

A good keyframe must satisfy these conditions:
1. It belongs to the same target chart and visual scene.
2. It shows the chart as complete as possible for the target chart type.
3. The data marks, labels, legends, axes, title, annotations, and category/value text are visible when present in the video.
4. It is not a transition, scene cut, occluded frame, cropped close-up, decorative interstitial, or unrelated illustration.
5. If the clip shows an animation from an initial state to a final state, prefer the frame that best preserves the complete chart structure and readable data, not necessarily the earliest frame.

Type-specific guidance:
- bar/line/area/scatter: prefer full axes, labels, legends, and all visible series/points/marks.
- pie/donut/treemap: prefer frames where segments, labels, percentages or values, and legend are most readable.
- map: prefer frames where geographic regions, encoded colors/symbols, legend, and labels are readable.
- timeline: prefer frames with visible time scale, events, labels, and ordering.
- sankey: prefer frames with visible nodes, flows, labels, and flow values if shown.
- pictograph: prefer frames with visible icon units, categories, labels, and legend.
- combined: prefer the frame where the combined chart encodings are simultaneously most readable.

Return strict JSON only:
{
  "same_chart": boolean,
  "scene_change": boolean,
  "complete_chart": boolean,
  "data_marks_readable": boolean,
  "labels_readable": boolean,
  "legend_or_axes_readable": boolean,
  "staticness": number,
  "completeness": number,
  "chart_identity_consistency": number,
  "data_extraction_suitability": number,
  "motion_score": number,
  "reason": string
}
"""

DATA_PROMPT = """Recover the data from this data visualization keyframe.

Chart context:
__CHART_CONTEXT__

Rules:
- Return only strict JSON.
- Do not invent numbers, labels, categories, coordinates, or percentages.
- If the chart has no concrete readable numeric values, percentages, coordinates, dates, counts, ranks, or clearly labeled categorical measurements, set has_extractable_data=false and rows=[].
- If some values are readable and others are not, include only the readable or explicitly labeled values and describe uncertainty.
- Preserve units exactly as shown when visible.
- For map charts, extract region-level values only when a legend/value/label lets you read concrete data; otherwise rows=[].
- For pictographs, only infer values when the unit-per-icon rule is visible or explicitly labeled.

Schema:
{
  "has_extractable_data": boolean,
  "chart_type": string,
  "title": null,
  "unit": null,
  "x_axis": null,
  "y_axis": null,
  "series": [],
  "rows": [
    {
      "label": null,
      "series": null,
      "x": null,
      "y": null,
      "value": null,
      "unit": null,
      "raw_text": null
    }
  ],
  "uncertain_fields": [],
  "skip_reason": null,
  "notes": ""
}
"""


def _json_from_text(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.S)
    payload = match.group(0) if match else text
    return json.loads(payload)


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _normalize_keyframe_score(result: dict[str, Any]) -> dict[str, Any]:
    complete_chart = _as_bool(result.get("complete_chart", False))
    return {
        "same_chart": _as_bool(result.get("same_chart", False)),
        "scene_change": _as_bool(result.get("scene_change", False)),
        "complete_chart": complete_chart,
        "data_marks_readable": _as_bool(result.get("data_marks_readable", complete_chart)),
        "labels_readable": _as_bool(result.get("labels_readable", False)),
        "legend_or_axes_readable": _as_bool(result.get("legend_or_axes_readable", False)),
        "staticness": _clamp01(result.get("staticness", 0.0)),
        "completeness": _clamp01(result.get("completeness", 0.0)),
        "chart_identity_consistency": _clamp01(result.get("chart_identity_consistency", 0.0)),
        "data_extraction_suitability": _clamp01(result.get("data_extraction_suitability", 0.0)),
        "motion_score": _clamp01(result.get("motion_score", 1.0), default=1.0),
        "reason": str(result.get("reason", "") or ""),
    }


def _normalize_chart_data(result: dict[str, Any], chart_type: str) -> dict[str, Any]:
    rows = result.get("rows") if isinstance(result.get("rows"), list) else []
    normalized_rows = []
    for row in rows:
        if isinstance(row, dict):
            normalized_rows.append(
                {
                    "label": row.get("label"),
                    "series": row.get("series"),
                    "x": row.get("x"),
                    "y": row.get("y"),
                    "value": row.get("value"),
                    "unit": row.get("unit"),
                    "raw_text": row.get("raw_text"),
                }
            )
    has_extractable = _as_bool(result.get("has_extractable_data", bool(normalized_rows)))
    if not normalized_rows:
        has_extractable = False
    return {
        "has_extractable_data": has_extractable,
        "chart_type": str(result.get("chart_type", chart_type) or chart_type),
        "title": result.get("title"),
        "unit": result.get("unit"),
        "x_axis": result.get("x_axis"),
        "y_axis": result.get("y_axis"),
        "series": result.get("series") if isinstance(result.get("series"), list) else [],
        "rows": normalized_rows if has_extractable else [],
        "uncertain_fields": result.get("uncertain_fields") if isinstance(result.get("uncertain_fields"), list) else [],
        "skip_reason": result.get("skip_reason") if not has_extractable else None,
        "notes": str(result.get("notes", "") or ""),
    }


class MultichartQwenClient:
    def __init__(self, cfg: dict[str, Any]):
        self.base = QwenVLClient(cfg)
        self.cfg = cfg

    @property
    def model_path(self) -> str | None:
        return self.base.model_path

    def score_keyframe_candidate(self, image_path: str, clip_context: dict[str, Any]) -> dict[str, Any]:
        if self.base.load():
            try:
                prompt = KEYFRAME_SCORE_PROMPT.replace(
                    "__CLIP_CONTEXT__", json.dumps(clip_context, ensure_ascii=False, indent=2)
                )
                raw = self.base._generate([image_path], prompt, max_new_tokens=224)
                result = _normalize_keyframe_score(_json_from_text(raw))
                return {"result": result, "raw_response": raw, "model_status": "qwen", "failure_reason": None}
            except Exception as exc:
                return self._unavailable_keyframe_score(f"qwen inference failed: {exc}")
        return self._unavailable_keyframe_score(self.base.load_error or "qwen unavailable")

    def recover_chart_data(self, image_path: str, chart_context: dict[str, Any]) -> dict[str, Any]:
        chart_type = str(chart_context.get("chart_type", "unknown") or "unknown")
        if self.base.load():
            try:
                prompt = DATA_PROMPT.replace(
                    "__CHART_CONTEXT__", json.dumps(chart_context, ensure_ascii=False, indent=2)
                )
                raw = self.base._generate([image_path], prompt, max_new_tokens=768)
                data = _normalize_chart_data(_json_from_text(raw), chart_type)
                return {"data": data, "raw_response": raw, "model_status": "qwen", "failure_reason": None}
            except Exception as exc:
                return self._unknown_data(chart_type, f"qwen inference failed: {exc}")
        return self._unknown_data(chart_type, self.base.load_error or "qwen unavailable")

    def _unavailable_keyframe_score(self, reason: str) -> dict[str, Any]:
        result = {
            "same_chart": False,
            "scene_change": True,
            "complete_chart": False,
            "data_marks_readable": False,
            "labels_readable": False,
            "legend_or_axes_readable": False,
            "staticness": 0.0,
            "completeness": 0.0,
            "chart_identity_consistency": 0.0,
            "data_extraction_suitability": 0.0,
            "motion_score": 1.0,
            "reason": f"Qwen multichart keyframe scoring unavailable: {reason}",
        }
        return {"result": result, "raw_response": None, "model_status": "qwen_unavailable", "failure_reason": reason}

    def _unknown_data(self, chart_type: str, reason: str) -> dict[str, Any]:
        data = {
            "has_extractable_data": False,
            "chart_type": chart_type,
            "title": None,
            "unit": None,
            "x_axis": None,
            "y_axis": None,
            "series": [],
            "rows": [],
            "uncertain_fields": [],
            "skip_reason": f"Could not reliably recover values: {reason}",
            "notes": "",
        }
        return {"data": data, "raw_response": None, "model_status": "unavailable", "failure_reason": reason}
