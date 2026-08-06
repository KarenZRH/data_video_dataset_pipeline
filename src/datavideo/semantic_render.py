"""Data-driven semantic SVG renderer.

Renders a chart's semantic.svg / semantic_components.json from the recovered
data table (entities + values) instead of from VLM-predicted bounding boxes.
Bar geometry is computed deterministically from the values, so coordinates are
exact by construction.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

from .schemas import ensure_dir, write_json


W, H = 1280, 720
LEFT, RIGHT, TOP, BOTTOM = 120, 1220, 140, 600


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return slug or "item"


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
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


def _sanitize_unit(unit: Any, metadata: dict[str, Any]) -> str:
    visible = metadata.get("visible_text")
    if isinstance(visible, list) and any("%" in str(t) or "percent" in str(t).lower() for t in visible):
        return "%"
    u = str(unit or "").strip()
    if len(u) <= 4 and u.lower() not in ("none", "unknown", "unit"):
        return u
    return ""


def _layout(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not entities:
        return []
    maxv = max(e["value"] for e in entities) or 1.0
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
) -> dict[str, Any]:
    objects = []
    groups = []
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
                "animation_axis": "y",
                "anchor": "bottom",
            }
        )
        objects.append(
            {
                "id": f"{eid}-value-label",
                "entity_id": eid,
                "type": "value_label",
                "label": f"{e['value']:g}{unit}",
                "text": f"{e['value']:g}{unit}",
                "text_status": "readable",
                "bbox_px": [round(x), round(y - 34), round(x + w), round(y - 4)],
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
                "bbox_px": [round(x), round(BOTTOM + 6), round(x + w), round(BOTTOM + 36)],
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


def _build_svg(layout: list[dict[str, Any]], title: str, unit: str) -> str:
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
        ty = BOTTOM - tv / maxv * (BOTTOM - TOP)
        lines.append(f'<line data-role="tick" x1="{LEFT - 8}" y1="{ty:.1f}" x2="{LEFT}" y2="{ty:.1f}" stroke="#666666" stroke-width="2"/>')
        lines.append(f'<text data-role="tick-label" x="{LEFT - 16}" y="{ty + 6:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="22" fill="#444444">{tv:g}{html.escape(unit)}</text>')
    for i, e in enumerate(layout):
        eid = _slug(e["label"])
        color = _color(i)
        lines.append(f'<g id="entity-{eid}" data-role="entity" data-entity-id="{eid}" data-label="{html.escape(e["label"])}">')
        lines.append(
            f'<rect id="{eid}-bar" data-role="bar" data-entity-id="{eid}" data-value="{e["value"]:g}" '
            f'x="{e["x"]:.1f}" y="{e["y"]:.1f}" width="{e["w"]:.1f}" height="{e["h"]:.1f}" fill="{color}" '
            'data-animation-property="height" data-anchor="bottom" data-animation-axis="y"/>'
        )
        lines.append(
            f'<text id="{eid}-value-label" data-role="value-label" data-entity-id="{eid}" '
            f'x="{e["x"] + e["w"] / 2:.1f}" y="{e["y"] - 14:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#222222">{e["value"]:g}{html.escape(unit)}</text>'
        )
        lines.append(
            f'<text id="{eid}-label" data-role="category-label" data-entity-id="{eid}" '
            f'x="{e["x"] + e["w"] / 2:.1f}" y="{BOTTOM + 30:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" fill="#333333">{html.escape(e["label"])}</text>'
        )
        lines.append("</g>")
    lines.append("</g>")
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _render_preview(layout: list[dict[str, Any]], title: str, unit: str, out: Path) -> bool:
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
    d.text((W / 2 - d.textlength(title, font=font_t) / 2, 30), title, fill=(30, 30, 30), font=font_t)
    d.line([(LEFT, BOTTOM), (RIGHT, BOTTOM)], fill=(100, 100, 100), width=3)
    d.line([(LEFT, TOP), (LEFT, BOTTOM)], fill=(100, 100, 100), width=3)
    maxv = max((e["value"] for e in layout), default=1.0) or 1.0
    for tv in _nice_ticks(maxv):
        ty = BOTTOM - tv / maxv * (BOTTOM - TOP)
        d.line([(LEFT - 8, ty), (LEFT, ty)], fill=(100, 100, 100), width=2)
        d.text((LEFT - 70, ty - 12), f"{tv:g}{unit}", fill=(80, 80, 80), font=font_a)
    for i, e in enumerate(layout):
        d.rectangle([e["x"], e["y"], e["x"] + e["w"], BOTTOM], fill=_color(i), outline=(0, 0, 0))
        text = f"{e['value']:g}{unit}"
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
    layout = _layout(entities)
    title = str(metadata.get("title") or "Data Chart")
    unit = _sanitize_unit(metadata.get("unit"), metadata)
    components = _build_components(clip_id, layout, title, unit)
    svg_path = out_dir / "semantic.svg"
    comp_path = out_dir / "semantic_components.json"
    scene_path = out_dir / "semantic_scene.json"
    preview_path = out_dir / "semantic_preview.png"
    svg_path.write_text(_build_svg(layout, title, unit), encoding="utf-8")
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
    preview_success = _render_preview(layout, title, unit, preview_path)
    return {
        "tool": "semantic_render",
        "generator": "datavideo.semantic_render_v1",
        "input": "",
        "annotation": str(comp_path),
        "semantic_svg": str(svg_path),
        "semantic_scene": str(scene_path),
        "semantic_preview": str(preview_path),
        "success": bool(entities) and svg_path.exists(),
        "failure_reason": None if entities else "no_recoverable_entities",
        "preview_success": preview_success,
        "preview_failure_reason": None,
        "entity_count": len(entities),
    }


def render_dynamic_states(
    clip_id: str,
    dynamic: dict[str, Any],
    out_dir: str | Path,
) -> list[dict[str, Any]]:
    """Render one data-driven SVG per recovered state (state_key)."""
    states = dynamic.get("states") if isinstance(dynamic, dict) else []
    groups: dict[str, list[dict[str, Any]]] = {}
    unit = "%"
    for row in states if isinstance(states, list) else []:
        if not isinstance(row, dict):
            continue
        eid = str(row.get("entity_id") or "")
        if eid in ("", "unknown"):
            continue
        value = _to_float(row.get("value"))
        if value is None:
            continue
        if row.get("unit") not in (None, ""):
            unit = str(row["unit"])
        key = str(row.get("state_key") or row.get("state_label") or row.get("state_id") or "state")
        groups.setdefault(key, []).append(
            {"id": eid, "label": str(row.get("entity") or eid), "value": value}
        )
    reports = []
    for key, entities in groups.items():
        if not entities:
            continue
        safe = re.sub(r"[^a-z0-9]+", "-", key.lower()).strip("-") or "state"
        sub = ensure_dir(Path(out_dir) / "semantic_states" / safe)
        metadata = {
            "title": f"{clip_id} state {key}",
            "unit": unit,
            "series": [
                {"name": e["label"], "values": [e["value"]]}
                for e in entities
            ],
        }
        report = render_data_driven(clip_id, metadata, sub)
        reports.append({"state_key": key, "state_dir": str(sub), **report})
    return reports


def metadata_from_dynamic(dynamic: dict[str, Any]) -> dict[str, Any] | None:
    """Build a chart metadata dict from dynamic states (frame-corrected values).

    Used after CV alignment/reconciliation so the primary semantic.svg is
    rendered from the frame-truth numbers instead of the VLM-recovered ones.
    Prefers the state group with the most entities (usually the keyframe state
    that CV alignment verified).
    """
    states = dynamic.get("states") if isinstance(dynamic, dict) else []
    if not isinstance(states, list):
        return None
    groups: dict[str, list[dict[str, Any]]] = {}
    unit = "%"
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
        if row.get("unit") not in (None, ""):
            unit = str(row["unit"])
        if row.get("metric"):
            metric = str(row["metric"])
        key = str(row.get("state_key") or row.get("state_label") or row.get("state_id") or "state")
        groups.setdefault(key, []).append(
            {"name": str(row.get("entity") or eid), "values": [value]}
        )
    if not groups:
        return None
    key = max(groups, key=lambda k: len(groups[k]))
    series = groups[key]
    series.sort(key=lambda e: -float(e["values"][0]))
    title = metric if key == "state" else f"{metric} ({key})"
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
