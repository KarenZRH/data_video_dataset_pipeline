"""Data-driven semantic SVG renderer.

Renders a chart's semantic.svg / semantic_components.json from the recovered
data table (entities + values) instead of from VLM-predicted bounding boxes.
Bar geometry is computed deterministically from the values, so coordinates are
exact by construction.
"""

from __future__ import annotations

import html
import re
import shutil
from pathlib import Path
from typing import Any

from .schemas import ensure_dir, write_json


W, H = 1280, 720
LEFT, RIGHT, TOP, BOTTOM = 120, 1220, 140, 600


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return slug or "item"


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().replace(",", "").replace("$", "").replace("%", "")
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def entities_from_metadata(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Prefer chart_metadata.series; fall back to entities (skip placeholders)."""
    series = metadata.get("series") if isinstance(metadata.get("series"), list) else []
    if series:
        entities: list[dict[str, Any]] = []
        for item in series:
            if not isinstance(item, dict):
                continue
            label = str(item.get("name") or "").strip()
            if not label:
                continue
            values = item.get("values") if isinstance(item.get("values"), list) else []
            value = _to_float(values[0]) if values else None
            if value is None:
                continue
            entities.append({"label": label, "value": value})
        if entities:
            return entities

    entities = []
    seen = set()
    for item in metadata.get("entities") if isinstance(metadata.get("entities"), list) else []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label or label.startswith("entity_"):
            continue
        value = _to_float(item.get("value"))
        if value is None:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        entities.append({"label": label, "value": value})
    return entities


def _nice_ticks(maxv: float) -> list[float]:
    if maxv <= 0:
        return [0.0]
    import math

    raw_step = 10 ** (math.floor(math.log10(maxv)) - 1)
    step = raw_step
    for mult in (1, 2, 5, 10):
        candidate = mult * raw_step
        count = int(maxv / candidate) + 1
        if 3 <= count <= 9:
            step = candidate
            break
    ticks = []
    v = 0.0
    while v <= maxv * 1.02 and len(ticks) < 10:
        ticks.append(round(v, 6))
        v += step
    return ticks


_YEAR_RE = re.compile(r"^\d{2,4}$")


def _timestamp_evidenced(state_key: Any, visible_text: Any) -> bool:
    """Whether a timestamp-like state key (e.g. "2019") is actually visible in
    the video frame.  A year/period must never be added to the SVG title just
    because the model guessed it; it has to appear in the frame's visible
    text (``visible_text`` from clip recovery)."""
    key = str(state_key or "").strip()
    if not key or key == "state":
        return True
    if not _YEAR_RE.match(key):
        return False
    norm = re.sub(r"[^a-z0-9]+", "", key.lower())
    tokens = visible_text if isinstance(visible_text, list) else []
    for token in tokens:
        if norm and norm in re.sub(r"[^a-z0-9]+", "", str(token).lower()):
            return True
    return False


def resolve_render_title(original_title: Any, auto_title: Any) -> str:
    """Prefer the VLM-read chart title, unless it carries a year that
    conflicts with the state actually being rendered (then keep the
    evidence-based auto title, e.g. "Illiteracy Rate (2017)" instead of a
    stale "Illiteracy Rate 1990" while the bars show the 2017 values)."""
    original = str(original_title or "").strip()
    auto = str(auto_title or "").strip()
    if not original:
        return auto
    years_auto = set(re.findall(r"\d{4}", auto))
    years_orig = set(re.findall(r"\d{4}", original))
    # Only override the VLM title when the original itself carries a year
    # that contradicts the state being rendered. An original without any
    # year (e.g. "Monthly price of Sovaldi, hepatitis C drug") must be kept,
    # even when the auto title appends the state year ("Price (2017)").
    if years_orig and years_auto and years_orig != years_auto:
        return auto
    return original


def frame_title_status(title: Any, visible_text: Any) -> str:
    """Classify how well ``title`` matches the frame's visible text.

    Returns one of:
      * "visible"  - the title (or a heavily overlapping variant) is printed
                     in the frame;
      * "candidate" - the title is not in the frame but a longer visible line
                     looks like the real chart title;
      * "none"     - neither, so a vision read of the frame is needed.
    """
    title_text = str(title or "").strip()
    if not title_text:
        return "none"
    tokens = [str(token) for token in visible_text] if isinstance(visible_text, list) else []
    title_norm = re.sub(r"[^a-z0-9]+", "", title_text.lower())
    for token in tokens:
        if title_norm and title_norm in re.sub(r"[^a-z0-9]+", "", token.lower()):
            return "visible"
    candidates = [
        token for token in tokens
        if re.search(r"[a-zA-Z]", token) and len(token) > 15
    ]
    if not candidates:
        return "none"
    title_tokens = set(re.findall(r"[a-z0-9]+", title_text.lower()))
    if any(
        len(title_tokens & set(re.findall(r"[a-z0-9]+", candidate.lower())))
        / max(1, len(title_tokens))
        >= 0.5
        for candidate in candidates
    ):
        return "visible"
    return "candidate"


def prefer_frame_visible_title(title: Any, visible_text: Any) -> str:
    """Prefer the real chart title printed in the frame over a VLM title.

    The VLM occasionally reports the *video* title instead of the chart title
    (e.g. "Why drugs cost more in America" while the frame says "Adults who
    skipped prescriptions or doses because of cost"). When the resolved title
    does not appear in the frame text and no visible candidate overlaps it
    heavily (a partially-truncated version of the same title), the longest
    visible text line is used instead.
    """
    title_text = str(title or "").strip()
    status = frame_title_status(title_text, visible_text)
    if status == "candidate":
        tokens = [str(token) for token in visible_text] if isinstance(visible_text, list) else []
        candidates = [
            token for token in tokens
            if re.search(r"[a-zA-Z]", token) and len(token) > 15
        ]
        if candidates:
            return max(candidates, key=len)
    return title_text


def _sanitize_unit(unit: Any, metadata: dict[str, Any]) -> str:
    visible = metadata.get("visible_text")
    if isinstance(visible, list) and any("%" in str(t) or "percent" in str(t).lower() for t in visible):
        return "%"
    u = str(unit or "").strip()
    if len(u) <= 4 and u.lower() not in ("none", "unknown", "unit"):
        return u
    return ""


def _infer_unit(rows: list[dict[str, Any]], visible_text: Any = None) -> str:
    """Infer the render unit from recovered rows, falling back to visible text.

    The unit must come from the original chart (printed labels/axis), never a
    hard-coded default: e.g. SAT scores must not render as "890%".
    """
    for row in rows:
        if not isinstance(row, dict):
            continue
        unit = str(row.get("unit") or "").strip()
        if unit.lower() not in ("", "none", "unknown", "unit"):
            return unit
    tokens = [str(token) for token in visible_text] if isinstance(visible_text, list) else []
    if any("%" in token or "percent" in token.lower() for token in tokens):
        return "%"
    return ""


def _format_value(value: Any, unit: Any) -> str:
    """Format a numeric value with its unit; currency uses a "$" prefix."""
    unit = str(unit or "").strip()
    try:
        rendered = f"{float(value):g}"
    except (TypeError, ValueError):
        rendered = str(value)
    if unit == "$":
        return f"${rendered}"
    return f"{rendered}{unit}"


def _layout(
    entities: list[dict[str, Any]],
    orientation: str = "vertical",
) -> list[dict[str, Any]]:
    if not entities:
        return []
    maxv = max(e["value"] for e in entities) or 1.0
    if orientation == "horizontal":
        slot = (BOTTOM - TOP) / len(entities)
        bar_h = max(14.0, slot * 0.52)
        plot_w = RIGHT - LEFT
        out = []
        for i, e in enumerate(entities):
            wgt = max(0.0, e["value"] / maxv * plot_w)
            cy = TOP + slot * (i + 0.5)
            out.append({**e, "x": float(LEFT), "y": cy - bar_h / 2, "w": wgt, "h": bar_h})
        return out
    slot = (RIGHT - LEFT) / len(entities)
    bar_w = slot * 0.52
    plot_h = BOTTOM - TOP
    out = []
    for i, e in enumerate(entities):
        hgt = e["value"] / maxv * plot_h
        cx = LEFT + slot * (i + 0.5)
        x = cx - bar_w / 2
        y = BOTTOM - hgt
        out.append({**e, "x": x, "y": y, "w": bar_w, "h": hgt})
    return out


def _color(index: int) -> str:
    palette = ["#FFD700", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]
    return palette[index % len(palette)]


def _build_components(
    clip_id: str,
    layout: list[dict[str, Any]],
    title: str,
    unit: str,
    orientation: str = "vertical",
) -> dict[str, Any]:
    objects = []
    groups = []
    horizontal = orientation == "horizontal"
    for i, e in enumerate(layout):
        eid = _slug(e["label"])
        x, y, w, hgt = e["x"], e["y"], e["w"], e["h"]
        objects.append(
            {
                "id": f"{eid}-bar",
                "entity_id": eid,
                "type": "bar",
                "label": e["label"],
                "text": None,
                "text_status": "not_applicable",
                "bbox_px": [round(x), round(y), round(x + w), round(y + hgt)],
                "dominant_color": _color(i),
                "confidence": 1.0,
                "reason": "data-driven",
                "animation_axis": "x" if horizontal else "y",
                "anchor": "left" if horizontal else "bottom",
            }
        )
        value_label_bbox = (
            [round(x + w + 4), round(y + hgt / 2 - 14), round(x + w + 4 + max(60.0, len(str(e["value"])) * 14)), round(y + hgt / 2 + 14)]
            if horizontal
            else [round(x), round(y - 34), round(x + w), round(y - 4)]
        )
        category_label_bbox = (
            [round(x), round(y - 34), round(x + w), round(y - 4)]
            if horizontal
            else [round(x), round(BOTTOM + 6), round(x + w), round(BOTTOM + 36)]
        )
        objects.append(
            {
                "id": f"{eid}-value-label",
                "entity_id": eid,
                "type": "value_label",
                "label": _format_value(e["value"], unit),
                "text": _format_value(e["value"], unit),
                "text_status": "readable",
                "bbox_px": value_label_bbox,
                "dominant_color": _color(i),
                "confidence": 1.0,
                "reason": "data-driven",
                "animation_axis": None,
                "anchor": None,
            }
        )
        objects.append(
            {
                "id": f"{eid}-label",
                "entity_id": eid,
                "type": "category_label",
                "label": e["label"],
                "text": e["label"],
                "text_status": "readable",
                "bbox_px": category_label_bbox,
                "dominant_color": _color(i),
                "confidence": 1.0,
                "reason": "data-driven",
                "animation_axis": None,
                "anchor": None,
            }
        )
        groups.append(
            {
                "entity_id": eid,
                "label": e["label"],
                "component_ids": [f"{eid}-bar", f"{eid}-value-label", f"{eid}-label"],
                "confidence": 1.0,
            }
        )
    objects.append(
        {
            "id": "chart-title",
            "entity_id": None,
            "type": "title",
            "label": title,
            "text": title,
            "text_status": "readable",
            "bbox_px": [round(W / 2 - 300), 30, round(W / 2 + 300), 80],
            "dominant_color": "#222222",
            "confidence": 1.0,
            "reason": "data-driven",
            "animation_axis": None,
            "anchor": None,
        }
    )
    return {
        "clip_id": clip_id,
        "source_keyframe": "",
        "image_width": W,
        "image_height": H,
        "annotation_method": "data_driven_semantic_render_v1",
        "automation_level": "deterministic",
        "uses_chart_metadata": True,
        "metadata_title": title,
        "metadata_chart_type": "bar",
        "chart_type": "vertical_bar",
        "needs_review": False,
        "objects": objects,
        "entity_groups": groups,
        "reconciliation_actions": [],
        "warnings": [],
    }


def _build_svg(
    layout: list[dict[str, Any]],
    title: str,
    unit: str,
    orientation: str = "vertical",
) -> str:
    horizontal = orientation == "horizontal"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" data-role="semantic-chart" data-generator="datavideo.semantic_render_v1">',
        f'<rect id="scene-background-fill" data-role="background-fill" x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text id="chart-title" data-role="title" x="{W / 2}" y="70" text-anchor="middle" font-family="Arial, sans-serif" font-size="36" font-weight="700" fill="#222222">{html.escape(title)}</text>',
        '<g id="chart-plot" data-role="plot">',
        f'<line data-role="axis" x1="{LEFT}" y1="{BOTTOM}" x2="{RIGHT}" y2="{BOTTOM}" stroke="#666666" stroke-width="3"/>',
        f'<line data-role="axis" x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{BOTTOM}" stroke="#666666" stroke-width="3"/>',
    ]
    maxv = max((e["value"] for e in layout), default=1.0) or 1.0
    for tv in _nice_ticks(maxv):
        if horizontal:
            tx = LEFT + tv / maxv * (RIGHT - LEFT)
            lines.append(f'<line data-role="tick" x1="{tx:.1f}" y1="{BOTTOM}" x2="{tx:.1f}" y2="{BOTTOM + 8}" stroke="#666666" stroke-width="2"/>')
            lines.append(f'<text data-role="tick-label" x="{tx:.1f}" y="{BOTTOM + 30:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" fill="#444444">{html.escape(_format_value(tv, unit))}</text>')
        else:
            ty = BOTTOM - tv / maxv * (BOTTOM - TOP)
            lines.append(f'<line data-role="tick" x1="{LEFT - 8}" y1="{ty:.1f}" x2="{LEFT}" y2="{ty:.1f}" stroke="#666666" stroke-width="2"/>')
            lines.append(f'<text data-role="tick-label" x="{LEFT - 16}" y="{ty + 6:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="22" fill="#444444">{html.escape(_format_value(tv, unit))}</text>')
    for i, e in enumerate(layout):
        eid = _slug(e["label"])
        color = _color(i)
        lines.append(f'<g id="entity-{eid}" data-role="entity" data-entity-id="{eid}" data-label="{html.escape(e["label"])}">')
        lines.append(
            f'<rect id="{eid}-bar" data-role="bar" data-entity-id="{eid}" data-value="{e["value"]:g}" '
            f'x="{e["x"]:.1f}" y="{e["y"]:.1f}" width="{e["w"]:.1f}" height="{e["h"]:.1f}" fill="{color}" '
            f'data-animation-property="{"width" if horizontal else "height"}" data-anchor="{"left" if horizontal else "bottom"}" data-animation-axis="{"x" if horizontal else "y"}" data-orientation="{orientation}"/>'
        )
        value_label_x = (e["x"] + e["w"] + 14) if horizontal else (e["x"] + e["w"] / 2)
        value_label_y = (e["y"] + e["h"] / 2 + 8) if horizontal else (e["y"] - 14)
        value_anchor = "start" if horizontal else "middle"
        category_label_x = (e["x"] + 2) if horizontal else (e["x"] + e["w"] / 2)
        category_label_y = (e["y"] - 14) if horizontal else (BOTTOM + 30)
        category_anchor = "start" if horizontal else "middle"
        lines.append(
            f'<text id="{eid}-value-label" data-role="value-label" data-entity-id="{eid}" '
            f'x="{value_label_x:.1f}" y="{value_label_y:.1f}" text-anchor="{value_anchor}" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#222222">{html.escape(_format_value(e["value"], unit))}</text>'
        )
        lines.append(
            f'<text id="{eid}-label" data-role="category-label" data-entity-id="{eid}" '
            f'x="{category_label_x:.1f}" y="{category_label_y:.1f}" text-anchor="{category_anchor}" font-family="Arial, sans-serif" font-size="20" fill="#333333">{html.escape(e["label"])}</text>'
        )
        lines.append("</g>")
    lines.append("</g>")
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _estimate_text_width(text: str, font_size: float) -> float:
    """Rough text width estimate (average glyph ~0.55em) for box sizing."""
    return max(10.0, len(str(text)) * 0.55 * font_size)


def _build_components_svg(
    layout: list[dict[str, Any]],
    title: str,
    unit: str,
    orientation: str = "vertical",
) -> str:
    """Data-driven boxed component diagram (semantic_components.svg).

    Mirrors the annotation style used for review: white boxes around the title,
    each printed value and each category name, plus red-bordered bars.
    """
    horizontal = orientation == "horizontal"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" data-role="semantic-components" data-generator="datavideo.semantic_render_v1">',
        f'<rect id="scene-background-fill" data-role="background-fill" x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>',
    ]
    title_w = max(200.0, _estimate_text_width(title, 34) + 44)
    title_x = W / 2 - title_w / 2
    lines.append(
        f'<rect id="chart-title-box" data-role="title-box" x="{title_x:.1f}" y="28" '
        f'width="{title_w:.1f}" height="58" fill="#ffffff" stroke="#333333" stroke-width="3"/>'
    )
    lines.append(
        f'<text id="chart-title" data-role="title" x="{W / 2}" y="67" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="34" font-weight="700" fill="#222222">{html.escape(title)}</text>'
    )
    lines.append('<g id="chart-plot" data-role="plot">')
    lines.append(
        f'<line data-role="axis" x1="{LEFT}" y1="{BOTTOM}" x2="{RIGHT}" y2="{BOTTOM}" stroke="#666666" stroke-width="3"/>'
    )
    lines.append(
        f'<line data-role="axis" x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{BOTTOM}" stroke="#666666" stroke-width="3"/>'
    )
    maxv = max((e["value"] for e in layout), default=1.0) or 1.0
    for tv in _nice_ticks(maxv):
        if horizontal:
            tx = LEFT + tv / maxv * (RIGHT - LEFT)
            lines.append(
                f'<line data-role="tick" x1="{tx:.1f}" y1="{BOTTOM}" x2="{tx:.1f}" y2="{BOTTOM + 8}" stroke="#666666" stroke-width="2"/>'
            )
            lines.append(
                f'<text data-role="tick-label" x="{tx:.1f}" y="{BOTTOM + 30:.1f}" text-anchor="middle" '
                f'font-family="Arial, sans-serif" font-size="22" fill="#444444">{tv:g}{html.escape(unit)}</text>'
            )
        else:
            ty = BOTTOM - tv / maxv * (BOTTOM - TOP)
            lines.append(
                f'<line data-role="tick" x1="{LEFT - 8}" y1="{ty:.1f}" x2="{LEFT}" y2="{ty:.1f}" stroke="#666666" stroke-width="2"/>'
            )
            lines.append(
                f'<text data-role="tick-label" x="{LEFT - 16}" y="{ty + 6:.1f}" text-anchor="end" '
                f'font-family="Arial, sans-serif" font-size="22" fill="#444444">{tv:g}{html.escape(unit)}</text>'
            )
    for i, e in enumerate(layout):
        eid = _slug(e["label"])
        color = _color(i)
        lines.append(f'<g id="entity-{eid}" data-role="entity" data-entity-id="{eid}" data-label="{html.escape(e["label"])}">')
        lines.append(
            f'<rect id="{eid}-bar" data-role="bar" data-entity-id="{eid}" data-value="{e["value"]:g}" '
            f'x="{e["x"]:.1f}" y="{e["y"]:.1f}" width="{e["w"]:.1f}" height="{e["h"]:.1f}" fill="{color}" '
            f'stroke="#d62728" stroke-width="3" data-animation-property="{"width" if horizontal else "height"}" data-anchor="{"left" if horizontal else "bottom"}" data-animation-axis="{"x" if horizontal else "y"}" data-orientation="{orientation}"/>'
        )
        value_text = _format_value(e["value"], unit)
        value_w = max(56.0, _estimate_text_width(value_text, 22) + 22)
        value_h = 30.0
        if horizontal:
            value_x = e["x"] + e["w"] + 10
            value_y = e["y"] + e["h"] / 2 - value_h / 2
            value_text_x = value_x + 8
            value_text_anchor = "start"
        else:
            value_x = e["x"] + e["w"] / 2 - value_w / 2
            value_y = e["y"] - value_h - 8
            value_text_x = e["x"] + e["w"] / 2
            value_text_anchor = "middle"
        lines.append(
            f'<rect id="{eid}-value-box" data-role="value-box" data-entity-id="{eid}" '
            f'x="{value_x:.1f}" y="{value_y:.1f}" width="{value_w:.1f}" height="{value_h:.1f}" '
            'fill="#fafafa" stroke="#333333" stroke-width="2"/>'
        )
        lines.append(
            f'<text id="{eid}-value-label" data-role="value-label" data-entity-id="{eid}" '
            f'x="{value_text_x:.1f}" y="{value_y + 21:.1f}" text-anchor="{value_text_anchor}" '
            f'font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#222222">{html.escape(value_text)}</text>'
        )
        label_w = max(e["w"], _estimate_text_width(e["label"], 18) + 26)
        label_h = 30.0
        if horizontal:
            label_x = e["x"]
            label_y = e["y"] - label_h - 8
            label_text_x = label_x + 8
            label_text_anchor = "start"
        else:
            label_x = e["x"] + e["w"] / 2 - label_w / 2
            label_y = BOTTOM + 12
            label_text_x = e["x"] + e["w"] / 2
            label_text_anchor = "middle"
        lines.append(
            f'<rect id="{eid}-label-box" data-role="category-box" data-entity-id="{eid}" '
            f'x="{label_x:.1f}" y="{label_y:.1f}" width="{label_w:.1f}" height="{label_h:.1f}" '
            'fill="#fafafa" stroke="#333333" stroke-width="2"/>'
        )
        lines.append(
            f'<text id="{eid}-label" data-role="category-label" data-entity-id="{eid}" '
            f'x="{label_text_x:.1f}" y="{label_y + 21:.1f}" text-anchor="{label_text_anchor}" '
            f'font-family="Arial, sans-serif" font-size="18" fill="#333333">{html.escape(e["label"])}</text>'
        )
        lines.append("</g>")
    lines.append("</g>")
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _render_components_preview(
    layout: list[dict[str, Any]],
    title: str,
    unit: str,
    out: Path,
    orientation: str = "vertical",
) -> bool:
    """Render the boxed component diagram as a PNG for quick visual review."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return False
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    try:
        font_t = ImageFont.truetype("arial.ttf", 32)
        font_v = ImageFont.truetype("arial.ttf", 21)
        font_l = ImageFont.truetype("arial.ttf", 18)
        font_a = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font_t = font_v = font_l = font_a = ImageFont.load_default()
    horizontal = orientation == "horizontal"
    title_w = max(200.0, d.textlength(title, font=font_t) + 44)
    d.rectangle([W / 2 - title_w / 2, 28, W / 2 + title_w / 2, 88], fill="white", outline=(51, 51, 51), width=3)
    d.text((W / 2 - d.textlength(title, font=font_t) / 2, 34), title, fill=(34, 34, 34), font=font_t)
    d.line([(LEFT, BOTTOM), (RIGHT, BOTTOM)], fill=(100, 100, 100), width=3)
    d.line([(LEFT, TOP), (LEFT, BOTTOM)], fill=(100, 100, 100), width=3)
    maxv = max((e["value"] for e in layout), default=1.0) or 1.0
    for tv in _nice_ticks(maxv):
        if horizontal:
            tx = LEFT + tv / maxv * (RIGHT - LEFT)
            d.line([(tx, BOTTOM), (tx, BOTTOM + 8)], fill=(100, 100, 100), width=2)
            d.text((tx - 20, BOTTOM + 12), _format_value(tv, unit), fill=(80, 80, 80), font=font_a)
        else:
            ty = BOTTOM - tv / maxv * (BOTTOM - TOP)
            d.line([(LEFT - 8, ty), (LEFT, ty)], fill=(100, 100, 100), width=2)
            d.text((LEFT - 70, ty - 12), _format_value(tv, unit), fill=(80, 80, 80), font=font_a)
    for i, e in enumerate(layout):
        d.rectangle([e["x"], e["y"], e["x"] + e["w"], e["y"] + e["h"]], fill=_color(i), outline=(214, 39, 40), width=3)
        value_text = _format_value(e["value"], unit)
        value_w = max(56.0, d.textlength(value_text, font=font_v) + 22)
        if horizontal:
            value_x = e["x"] + e["w"] + 8
            value_y = e["y"] + e["h"] / 2 - 15
            value_text_x = value_x + 6
            value_text_anchor = "la"
        else:
            value_x = e["x"] + e["w"] / 2 - value_w / 2
            value_y = e["y"] - 38
            value_text_x = e["x"] + e["w"] / 2 - d.textlength(value_text, font=font_v) / 2
            value_text_anchor = "la"
        d.rectangle([value_x, value_y, value_x + value_w, value_y + 30], fill=(250, 250, 250), outline=(51, 51, 51), width=2)
        d.text((value_text_x, value_y + 4), value_text, fill=(34, 34, 34), font=font_v, anchor=value_text_anchor)
        label_w = max(e["w"], d.textlength(e["label"], font=font_l) + 26)
        if horizontal:
            label_x = e["x"]
            label_y = e["y"] - 38
            label_text_x = label_x + 6
            label_text_anchor = "la"
        else:
            label_x = e["x"] + e["w"] / 2 - label_w / 2
            label_y = BOTTOM + 12
            label_text_x = e["x"] + e["w"] / 2 - d.textlength(e["label"], font=font_l) / 2
            label_text_anchor = "la"
        d.rectangle([label_x, label_y, label_x + label_w, label_y + 30], fill=(250, 250, 250), outline=(51, 51, 51), width=2)
        d.text((label_text_x, label_y + 5), e["label"], fill=(51, 51, 51), font=font_l, anchor=label_text_anchor)
    img.save(out)
    return out.exists()


def _render_preview(
    layout: list[dict[str, Any]],
    title: str,
    unit: str,
    out: Path,
    orientation: str = "vertical",
) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return False
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    try:
        font_t = ImageFont.truetype("arial.ttf", 34)
        font_v = ImageFont.truetype("arial.ttf", 24)
        font_l = ImageFont.truetype("arial.ttf", 20)
        font_a = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font_t = font_v = font_l = font_a = ImageFont.load_default()
    horizontal = orientation == "horizontal"
    d.text((W / 2 - d.textlength(title, font=font_t) / 2, 30), title, fill=(30, 30, 30), font=font_t)
    d.line([(LEFT, BOTTOM), (RIGHT, BOTTOM)], fill=(100, 100, 100), width=3)
    d.line([(LEFT, TOP), (LEFT, BOTTOM)], fill=(100, 100, 100), width=3)
    maxv = max((e["value"] for e in layout), default=1.0) or 1.0
    for tv in _nice_ticks(maxv):
        if horizontal:
            tx = LEFT + tv / maxv * (RIGHT - LEFT)
            d.line([(tx, BOTTOM), (tx, BOTTOM + 8)], fill=(100, 100, 100), width=2)
            d.text((tx - 20, BOTTOM + 12), _format_value(tv, unit), fill=(80, 80, 80), font=font_a)
        else:
            ty = BOTTOM - tv / maxv * (BOTTOM - TOP)
            d.line([(LEFT - 8, ty), (LEFT, ty)], fill=(100, 100, 100), width=2)
            d.text((LEFT - 70, ty - 12), _format_value(tv, unit), fill=(80, 80, 80), font=font_a)
    for i, e in enumerate(layout):
        d.rectangle([e["x"], e["y"], e["x"] + e["w"], e["y"] + e["h"]], fill=_color(i), outline=(0, 0, 0))
        text = _format_value(e["value"], unit)
        if horizontal:
            d.text((e["x"] + e["w"] + 10, e["y"] + e["h"] / 2 - 16), text, fill=(30, 30, 30), font=font_v)
            d.text((e["x"] + 2, e["y"] - 28), e["label"], fill=(50, 50, 50), font=font_l)
        else:
            d.text((e["x"] + e["w"] / 2 - d.textlength(text, font=font_v) / 2, e["y"] - 28), text, fill=(30, 30, 30), font=font_v)
            d.text((e["x"] + e["w"] / 2 - d.textlength(e["label"], font=font_l) / 2, BOTTOM + 8), e["label"], fill=(50, 50, 50), font=font_l)
    img.save(out)
    return out.exists()


def render_data_driven(
    clip_id: str,
    metadata: dict[str, Any],
    out_dir: str | Path,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    entities = entities_from_metadata(metadata)
    orientation = str(metadata.get("orientation") or "vertical")
    layout = _layout(entities, orientation)
    title = str(metadata.get("title") or "Data Chart").replace("\r", " ")
    # A VLM title may join the main title and the source line with a newline
    # ("Monthly price of Humira, arthritis drug\nCommonwealth Fund, 2017");
    # render only the main title line.
    title = title.split("\n", 1)[0].strip() or "Data Chart"
    unit = _sanitize_unit(metadata.get("unit"), metadata)
    components = _build_components(clip_id, layout, title, unit, orientation)
    svg_path = out_dir / "semantic.svg"
    components_svg_path = out_dir / "semantic_components.svg"
    comp_path = out_dir / "semantic_components.json"
    scene_path = out_dir / "semantic_scene.json"
    preview_path = out_dir / "semantic_preview.png"
    components_preview_path = out_dir / "semantic_components_preview.png"
    svg_path.write_text(_build_svg(layout, title, unit, orientation), encoding="utf-8")
    components_svg_path.write_text(_build_components_svg(layout, title, unit, orientation), encoding="utf-8")
    write_json(comp_path, components)
    write_json(
        scene_path,
        {
            "clip_id": clip_id,
            "source_keyframe": "",
            "image_width": W,
            "image_height": H,
            "annotation_source": "semantic_components.json",
            "generator": "datavideo.semantic_render_v1",
            "contains_data_values": True,
            "entities": [
                {
                    "entity_id": _slug(e["label"]),
                    "label": e["label"],
                    "value": e["value"],
                    "bbox_px": [round(e["x"]), round(e["y"]), round(e["x"] + e["w"]), round(e["y"] + e["h"])],
                }
                for e in layout
            ],
            "non_entity_components": [],
        },
    )
    preview_success = _render_preview(layout, title, unit, preview_path, orientation)
    components_preview_success = _render_components_preview(layout, title, unit, components_preview_path, orientation)
    return {
        "tool": "semantic_render",
        "generator": "datavideo.semantic_render_v1",
        "input": "",
        "annotation": str(comp_path),
        "semantic_svg": str(svg_path),
        "semantic_components_svg": str(components_svg_path),
        "semantic_scene": str(scene_path),
        "semantic_preview": str(preview_path),
        "semantic_components_preview": str(components_preview_path),
        "success": bool(entities) and svg_path.exists(),
        "failure_reason": None if entities else "no_recoverable_entities",
        "preview_success": preview_success,
        "preview_failure_reason": None,
        "components_preview_success": components_preview_success,
        "entity_count": len(entities),
    }


def render_dynamic_states(
    clip_id: str,
    dynamic: dict[str, Any],
    out_dir: str | Path,
    visible_text: Any = None,
) -> list[dict[str, Any]]:
    """Render one data-driven SVG per recovered state (state_key)."""
    states = dynamic.get("states") if isinstance(dynamic, dict) else []
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in states if isinstance(states, list) else []:
        if not isinstance(row, dict):
            continue
        eid = str(row.get("entity_id") or "")
        if eid in ("", "unknown"):
            continue
        value = _to_float(row.get("value"))
        if value is None:
            continue
        key = str(row.get("state_key") or row.get("state_label") or row.get("state_id") or "state")
        groups.setdefault(key, []).append(
            {"id": eid, "label": str(row.get("entity") or eid), "value": value}
        )
    unit = _infer_unit(states if isinstance(states, list) else [], visible_text)
    reports = []
    state_root = ensure_dir(Path(out_dir) / "semantic_states")
    keep = {
        re.sub(r"[^a-z0-9]+", "-", key.lower()).strip("-") or "state"
        for key in groups
    }
    for child in state_root.iterdir():
        if child.is_dir() and child.name not in keep:
            shutil.rmtree(child, ignore_errors=True)
    for key, entities in groups.items():
        if not entities:
            continue
        safe = re.sub(r"[^a-z0-9]+", "-", key.lower()).strip("-") or "state"
        sub = ensure_dir(state_root / safe)
        metadata = {
            "title": (
                f"{clip_id} state {key}"
                if _timestamp_evidenced(key, visible_text)
                else f"{clip_id} state"
            ),
            "unit": unit,
            "series": [
                {"name": e["label"], "values": [e["value"]]}
                for e in entities
            ],
        }
        report = render_data_driven(clip_id, metadata, sub)
        reports.append({"state_key": key, "state_dir": str(sub), **report})
    return reports


def metadata_from_dynamic(
    dynamic: dict[str, Any],
    visible_text: Any = None,
) -> dict[str, Any] | None:
    """Build a chart metadata dict from dynamic states (frame-corrected values).

    Used after CV alignment/reconciliation so the primary semantic.svg is
    rendered from the frame-truth numbers instead of the VLM-recovered ones.
    Prefers a state group that contains CV-aligned (visual_frame_align) rows,
    then the group with the most entities; within the chosen group duplicate
    entity labels are collapsed to a single value (CV-aligned first).
    """
    states = dynamic.get("states") if isinstance(dynamic, dict) else []
    if not isinstance(states, list):
        return None
    groups: dict[str, list[dict[str, Any]]] = {}
    metric = "Value"
    for row in states:
        if not isinstance(row, dict):
            continue
        eid = str(row.get("entity_id") or "")
        if eid in ("", "unknown"):
            continue
        value = _to_float(row.get("value"))
        if value is None:
            continue
        if row.get("metric"):
            metric = str(row["metric"])
        key = str(row.get("state_key") or row.get("state_label") or row.get("state_id") or "state")
        groups.setdefault(key, []).append(
            {
                "name": str(row.get("entity") or eid),
                "values": [value],
                "source_type": str(row.get("source_type") or ""),
                "confidence": _to_float(row.get("confidence")) or 0.0,
                "unit": row.get("unit"),
            }
        )
    if not groups:
        return None
    aligned_keys = [
        k
        for k, rows in groups.items()
        if any(r.get("source_type") == "visual_frame_align" for r in rows)
    ]
    candidates = aligned_keys or list(groups)
    key = max(candidates, key=lambda k: len(groups[k]))

    rows = groups[key]
    # When CV alignment verified this state, its entity set is authoritative:
    # drop labels that were not confirmed by the frame (e.g. hallucinated
    # "cycling" next to real "cyclists"/"drivers").
    vfa_labels = {
        _slug(r["name"]) for r in rows if r.get("source_type") == "visual_frame_align"
    }
    if vfa_labels:
        rows = [r for r in rows if _slug(r["name"]) in vfa_labels]

    # Collapse duplicate entity labels in the chosen state to a single row,
    # preferring the CV-aligned (visual_frame_align) observation.
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        norm = _slug(row["name"])
        if not norm or norm == "unknown":
            continue
        cur = best.get(norm)
        if cur is None:
            best[norm] = row
            continue
        cur_rank = (
            cur.get("source_type") == "visual_frame_align",
            float(cur.get("confidence") or 0.0),
        )
        row_rank = (
            row.get("source_type") == "visual_frame_align",
            float(row.get("confidence") or 0.0),
        )
        if row_rank > cur_rank:
            best[norm] = row
    series = [{"name": r["name"], "values": r["values"]} for r in best.values()]
    series.sort(key=lambda e: -float(e["values"][0]))
    if not series:
        return None
    unit = _infer_unit(list(best.values()), visible_text)
    if _slug(metric) in {_slug(r["name"]) for r in best.values()}:
        metric = "Value"
    metric_text = str(metric or "").strip()
    # When the VLM metric is a generic placeholder (value/price/cost...) or
    # missing, fall back to the longest visible text line (the chart title
    # is usually the longest printed text, e.g. "Retail prescription drug
    # spending per capita" instead of "value").
    if not re.search(r"[a-zA-Z]{3,}", metric_text) or metric_text.lower() in {"value", "price", "cost", "amount", "metric"}:
        tokens = [str(token) for token in visible_text] if isinstance(visible_text, list) else []
        candidates = [
            token for token in tokens
            if re.search(r"[a-zA-Z]", token) and len(token) > 15
        ]
        if candidates:
            metric_text = max(candidates, key=len)
    title = (
        f"{metric_text} ({key})"
        if key != "state" and _timestamp_evidenced(key, visible_text)
        else metric_text
    )
    return {
        "title": title,
        "chart_type": "bar",
        "unit": unit,
        "x_axis": "",
        "y_axis": metric,
        "series": series,
        "entities": [
            {"label": e["name"], "value": e["values"][0], "unit": unit}
            for e in series
        ],
        "visible_text": [],
        "needs_manual_data": False,
        "model_status": "cv_aligned",
        "failure_reason": None,
        "skip_reason": None,
    }
