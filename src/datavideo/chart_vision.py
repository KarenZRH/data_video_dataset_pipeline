"""Type-aware single vision read of the selected keyframe.

After chart-type dispatch and keyframe selection, one vision call returns a
chart-type-specific spec:

* bar family  -> ``match_chart_style`` (style + series label/value/color +
  title + orientation + ticks + value_labels)
* line family -> ``read_line_analysis`` (title/unit/x_labels/ticks/series
  points)
* map family  -> ``read_map_spec`` (regions name/color/value + legend + unit)

The renderer consumes the matching spec; CV geometry stays auxiliary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datavideo.cv_align import _call_vision, _extract_json_object

_BAR_FAMILY = {"bar", "stacked", "diverging", "pyramid", "combined"}
_LINE_FAMILY = {"line", "area", "timeline"}
_MAP_FAMILY = {"map", "choropleth"}

_MAP_PROMPT = (
    "这是视频截图里的一个地图/区域可视化。请一次性读出结构，只返回 JSON，不要解释：\n"
    '{"title": 标题原文（没有则空字符串）, '
    '"regions": [{"name": 区域名, "color": 填充色 hex, "value": 数值或 null}], '
    '"legend": 图例说明文字, "value_unit": 数值单位}'
)


def read_chart_spec(
    chart_type: str | None,
    image_path: str | Path,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """One type-aware vision call on the keyframe -> spec (or None)."""
    ct = str(chart_type or "").strip().lower()
    try:
        if ct in _LINE_FAMILY:
            from datavideo.line_processor import read_line_analysis

            spec = read_line_analysis(str(image_path), cfg)
            if spec:
                spec["chart_type"] = "line"
            return spec
        if ct in _MAP_FAMILY:
            return read_map_spec(str(image_path), cfg)
        if ct in _BAR_FAMILY or not ct:
            from datavideo.semantic_render import match_chart_style

            spec = match_chart_style(str(image_path), cfg)
            if spec:
                spec["chart_type"] = "bar"
            return spec
    except Exception:
        return None
    return None


def read_map_spec(
    image_path: str | Path,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Single vision read for map/region visualizations."""
    cfg = cfg or {}
    try:
        text = _call_vision(str(image_path), _MAP_PROMPT, cfg, temperature=0.0)
    except Exception:
        return None
    obj = _extract_json_object(text)
    if not isinstance(obj, dict):
        return None
    regions = []
    for r in obj.get("regions") or []:
        if not isinstance(r, dict):
            continue
        item = {"name": str(r.get("name") or "").strip(), "color": str(r.get("color") or "").strip()}
        try:
            item["value"] = float(r["value"])
        except (KeyError, TypeError, ValueError):
            item["value"] = None
        if item["name"]:
            regions.append(item)
    return {
        "title": str(obj.get("title") or "").strip(),
        "regions": regions,
        "legend": str(obj.get("legend") or "").strip(),
        "value_unit": str(obj.get("value_unit") or "").strip(),
        "chart_type": "map",
    }
