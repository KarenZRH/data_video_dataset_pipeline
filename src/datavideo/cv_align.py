"""CV-based bar detection + entity matching + value reading for aligned SVGs.

Produces, from a keyframe and the recovered data table:
  - aligned_overlay.png : the keyframe with boxes drawn on the real bars
  - semantic_aligned.svg: bars placed at the real (pixel) bar coordinates
  - aligned_report.json : boxes, matched entities, and value-ratio consistency

Values are read by cropping the value-label region above each detected bar and
asking the vision model to read the printed number (focused OCR is far more
reliable than whole-frame reading).
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .schemas import ensure_dir, write_json


def detect_bars(image_path: str | Path) -> list[dict[str, Any]]:
    """Detect colored bar regions in a chart keyframe via color segmentation."""
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    H, W = img.shape[:2]

    # estimate background hue: most common hue among saturated pixels
    sel = (s > 40) & (v > 40)
    if sel.sum() > 0:
        hist, _ = np.histogram(h[sel], bins=180, range=(0, 180))
        bg_hue = int(np.argmax(hist))
    else:
        bg_hue = 130
    bg_lo = max(0, bg_hue - 18)
    bg_hi = min(180, bg_hue + 18)
    bg = ((h >= bg_lo) & (h <= bg_hi)).astype(np.uint8)
    fg = ((s > 50) & (v > 50) & (bg == 0)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, 8)

    boxes: list[dict[str, Any]] = []
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        if y < 100 or w < 25 or hh < 25:
            continue
        if x < 5 or y < 5 or x + w > W - 5 or y + hh > H - 5:
            continue
        region = fg[y : y + hh, x : x + w]
        if region.sum() < area * 0.2:
            continue
        boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(hh)})
    boxes.sort(key=lambda b: b["x"])
    return boxes


def match_entities(
    boxes: list[dict[str, Any]],
    entities: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Match boxes to entities by left-to-right order; report mismatches."""
    aligned = []
    warnings: list[str] = []
    for i, box in enumerate(boxes):
        entity = entities[i] if i < len(entities) else None
        if entity is None:
            warnings.append(f"no entity for box #{i + 1} at x={box['x']}")
            continue
        aligned.append({**box, "entity_id": entity["id"], "label": entity["label"]})
    if len(boxes) > len(entities):
        warnings.append(f"detected {len(boxes)} bars but only {len(entities)} entities recovered")
    return aligned, warnings


def _crop_value_region(img: np.ndarray, box: dict[str, Any]) -> np.ndarray:
    x1 = max(0, box["x"] - 20)
    x2 = min(img.shape[1], box["x"] + box["w"] + 20)
    y1 = max(0, box["y"] - 60)
    y2 = min(img.shape[0], box["y"] + 8)
    if y2 <= y1 or x2 <= x1:
        return img[max(0, box["y"] - 40) : box["y"] + 8, x1:x2]
    return img[y1:y2, x1:x2]


def read_bar_values(
    client: Any,
    image_path: str | Path,
    aligned: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Read each bar's printed value from a focused crop above the bar."""
    if client is None:
        return []
    img = cv2.imread(str(image_path))
    out = []
    for i, item in enumerate(aligned):
        crop = _crop_value_region(img, item)
        crop_path = Path(image_path).with_name(f"value_crop_{i:02d}.png")
        cv2.imwrite(str(crop_path), crop)
        try:
            raw = client._generate(
                [str(crop_path)],
                "Read the number or percentage printed in this image crop. "
                "Return ONLY the number with unit, e.g. 36.1% or 6.9.",
                max_new_tokens=16,
            )
            match = re.search(r"-?\d+(?:\.\d+)?\s*%?", raw)
            value_text = match.group(0).strip() if match else None
        except Exception as exc:
            value_text = None
        out.append({**item, "value_text": value_text})
    return out


def _ratio_consistency(aligned: list[dict[str, Any]]) -> tuple[bool, str]:
    if len(aligned) < 2:
        return True, "too few bars to check"
    heights = [a["h"] for a in aligned]
    maxh = max(heights)
    ratios = [h / maxh for h in heights]
    errors = []
    for a, r in zip(aligned, ratios):
        v = a.get("value")
        if v is None or v == 0:
            continue
        expected = 1.0 / max(heights) * a["h"]  # ratio to max bar
        # expected value ratio vs observed height ratio
        val_ratio = a["value"] / max(v for item in aligned if item.get("value"))
        if abs(val_ratio - expected) > 0.12:
            errors.append(f"{a['label']}: value ratio {val_ratio:.2f} vs height ratio {expected:.2f}")
    if errors:
        return False, "; ".join(errors)
    return True, "ok"


def _render_overlay(image_path: str | Path, aligned: list[dict[str, Any]], out: Path) -> bool:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    color = (230, 25, 75)
    for a in aligned:
        d.rectangle([a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]], outline=color, width=4)
        label = f"{a.get('label','')} {a.get('value_text') or ''}".strip()
        d.text((a["x"], max(0, a["y"] - 26)), label, fill=color, font=font)
    img.save(out)
    return out.exists()


def _render_aligned_svg(aligned: list[dict[str, Any]], out: Path) -> bool:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720" data-role="semantic-chart" data-generator="datavideo.cv_align_v1">',
    ]
    for i, a in enumerate(aligned):
        eid = re.sub(r"[^a-z0-9]+", "-", str(a.get("entity_id") or "").lower()).strip("-") or f"bar-{i}"
        value = a.get("value_text") or ""
        lines.append(f'<g id="entity-{eid}" data-role="entity" data-entity-id="{eid}" data-label="{html.escape(str(a.get("label") or ""))}">')
        lines.append(
            f'<rect id="{eid}-bar" data-role="bar" data-entity-id="{eid}" data-value="{html.escape(value)}" '
            f'x="{a["x"]}" y="{a["y"]}" width="{a["w"]}" height="{a["h"]}" '
            'data-animation-property="height" data-anchor="bottom" fill="#3cb44b"/>'
        )
        if value:
            lines.append(
                f'<text data-role="value-label" x="{a["x"] + a["w"] / 2}" y="{max(12, a["y"] - 10)}" '
                f'text-anchor="middle" font-size="22" font-weight="700">{html.escape(value)}</text>'
            )
        lines.append("</g>")
    lines.append("</svg>")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out.exists()


def run_cv_align(
    clip_id: str,
    image_path: str | Path,
    entities: list[dict[str, Any]],
    out_dir: str | Path,
    client: Any = None,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    boxes = detect_bars(image_path)
    aligned, warnings = match_entities(boxes, entities)
    values = read_bar_values(client, image_path, aligned) if client is not None else aligned
    for item in values:
        if item.get("value_text"):
            try:
                item["value"] = float(re.sub(r"[^0-9.]", "", item["value_text"]))
            except ValueError:
                item["value"] = None
    consistent, message = _ratio_consistency(values)
    overlay = out_dir / "aligned_overlay.png"
    svg = out_dir / "semantic_aligned.svg"
    overlay_ok = _render_overlay(image_path, values, overlay)
    svg_ok = _render_aligned_svg(values, svg)
    report = {
        "clip_id": clip_id,
        "detected_bar_count": len(boxes),
        "matched_count": len(values),
        "warnings": warnings,
        "value_geometry_consistent": consistent,
        "consistency_message": message,
        "bars": values,
        "overlay_png": str(overlay),
        "aligned_svg": str(svg),
        "success": bool(values) and overlay_ok and svg_ok,
    }
    write_json(out_dir / "aligned_report.json", report)
    return report
