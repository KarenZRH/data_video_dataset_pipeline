"""CV-based bar detection + entity matching + vision value reading.

Produces, from a keyframe and the recovered data table:
  - aligned_overlay.png : the keyframe with boxes drawn on the real bars
  - semantic_aligned.svg: bars placed at the real (pixel) bar coordinates
  - aligned_report.json : boxes, matched entities, value-ratio consistency,
                          vision-based entity order and alignment verification

Values and label order are read by the external vision model (vision.js),
which is far more reliable than whole-frame OCR on the local 3B model.
"""

from __future__ import annotations

import html
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .schemas import ensure_dir, write_json


VISION_DEFAULTS = {
    "node_path": r"D:\node.exe",
    "script": r"C:\Users\ly200\.codex\skills\claude-vision-skill\vision.js",
    "proxy": None,
}


def _call_vision(
    image_path: str | Path,
    prompt: str,
    cfg: dict[str, Any] | None = None,
    *,
    temperature: float | None = None,
) -> str:
    """Call the external vision model (vision.js) on one image and return text.

    Tries a direct connection first (DashScope is reachable without a proxy in
    most environments); if a proxy is configured in ``cfg["cv_align"]["proxy"]``
    it is retried through that proxy when the direct attempt fails.
    """
    cfg = cfg or {}
    v = {**VISION_DEFAULTS, **(cfg.get("cv_align") or {})}
    attempts = [None]
    if v.get("proxy"):
        attempts.append(v["proxy"])
    last_error = "vision call failed"
    for proxy in attempts:
        env = {
            **os.environ,
            "PATH": str(Path(v["node_path"]).parent) + ";" + os.environ.get("PATH", ""),
        }
        if proxy:
            env["HTTPS_PROXY"] = proxy
            env["HTTP_PROXY"] = proxy
        else:
            env.pop("HTTPS_PROXY", None)
            env.pop("HTTP_PROXY", None)
        try:
            cmd = [v["node_path"], v["script"], str(image_path), prompt]
            if temperature is not None:
                cmd.extend(["--temp", str(temperature)])
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=120,
                env=env,
            )
            text = (result.stdout or "").strip() or (result.stderr or "").strip()
            if result.returncode == 0 and text:
                return text
            last_error = text or f"vision.js exit code {result.returncode}"
        except Exception as exc:
            last_error = str(exc)
    raise RuntimeError(last_error)


def detect_bars(image_path: str | Path) -> list[dict[str, Any]]:
    """Detect colored bar regions in a chart keyframe via color segmentation.

    Short bars (e.g. a 1% bar only a few pixels tall) are kept, so the height
    threshold is low; components that do not sit on the dominant baseline of
    the tall bars (typically merged category-label text) are dropped.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    H, W = img.shape[:2]

    sel = (s > 40) & (v > 40)
    if sel.sum() > 0:
        hist, _ = np.histogram(h[sel], bins=180, range=(0, 180))
        bg_hue = int(np.argmax(hist))
    else:
        bg_hue = 130
    bg = ((h >= max(0, bg_hue - 18)) & (h <= min(180, bg_hue + 18))).astype(np.uint8)
    fg = ((s > 50) & (v > 50) & (bg == 0)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, 8)

    candidates: list[dict[str, Any]] = []
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        if y < 100 or w < 25 or hh < 3:
            continue
        if x < 5 or y < 5 or x + w > W - 5 or y + hh > H - 5:
            continue
        region = fg[y : y + hh, x : x + w]
        if region.sum() < area * 0.2:
            continue
        candidates.append({"x": int(x), "y": int(y), "w": int(w), "h": int(hh)})

    if candidates:
        tall = [b for b in candidates if b["h"] >= 25]
        if tall:
            baseline = float(np.median([b["y"] + b["h"] for b in tall]))
            candidates = [
                b for b in candidates if abs((b["y"] + b["h"]) - baseline) <= 15
            ]
    candidates.sort(key=lambda b: b["x"])
    return candidates


def _clean_vision_label(part: str) -> str:
    part = str(part).strip().strip("*。·\t ")
    part = re.sub(r"^\d+[.)、:：]\s*", "", part)
    return part.strip()


def _normalize_label(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _labeled_value_pairs(text: str) -> list[tuple[str, str]]:
    """Extract ``Label: value`` pairs from a vision response."""
    pairs: list[tuple[str, str]] = []
    for match in re.finditer(r"([A-Za-z][A-Za-z &'\-\.]*?):\s*(-?\d+(?:\.\d+)?\s*%?)", text):
        label = _clean_vision_label(match.group(1))
        value_match = re.search(r"-?\d+(?:\.\d+)?\s*%?", match.group(2))
        if label and value_match:
            pairs.append((label, value_match.group(0).strip()))
    return pairs


def match_entities(
    boxes: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    vision_order: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Match boxes to entities by left-to-right order (vision-confirmed if available).

    When the vision model reports a category label that is not present in the
    recovered entities, a new entity is created from the frame label so the
    bar can still be aligned and reconciled into the data table.
    """
    aligned: list[dict[str, Any]] = []
    warnings: list[str] = []

    for i, box in enumerate(boxes):
        entity = None
        if vision_order and i < len(vision_order):
            want = _clean_vision_label(vision_order[i])
            norm_want = _normalize_label(want)
            if norm_want:
                for e in entities:
                    label = _normalize_label(e.get("label"))
                    if label and (norm_want in label or label in norm_want):
                        entity = e
                        break
            if entity is None and norm_want:
                entity = {
                    "id": re.sub(r"[^a-z0-9]+", "-", want.lower()).strip("-") or f"bar-{i + 1}",
                    "label": want,
                    "entity_source": "frame",
                }
                warnings.append(f"created entity from frame label: {want}")
        if entity is None and i < len(entities):
            entity = entities[i]
        if entity is None:
            warnings.append(f"no entity for box #{i + 1} at x={box['x']}")
            continue
        aligned.append(
            {
                **box,
                "entity_id": entity["id"],
                "label": entity["label"],
                "entity_source": entity.get("entity_source", "recovered"),
            }
        )
    if len(boxes) > len(entities):
        warnings.append(f"detected {len(boxes)} bars but only {len(entities)} entities recovered")
    return aligned, warnings


def read_entity_order(image_path: str | Path, cfg: dict[str, Any] | None = None) -> list[str]:
    """Ask the vision model for the left-to-right category labels of the bars."""
    text = _call_vision(
        image_path,
        "这是柱状图的一帧。请从左到右依次列出每根柱子底部的类别标签，只要名字本身，用逗号分隔，不要额外解释。",
        cfg,
        temperature=0.0,
    )
    labels = []
    skip = {"从左到右", "依次为", "第一根", "第二根", "第三根", "第四根", "柱子", "类别", "标签"}
    for part in re.split(r"[,，\n;；、]", text):
        part = _clean_vision_label(part)
        if part and part.lower() not in skip:
            labels.append(part)
    return labels


def verify_alignment(image_path: str | Path, cfg: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Ask the vision model whether the overlaid labels match the bars."""
    text = _call_vision(
        image_path,
        "检查这张图上叠加的文字标签是否与柱子的真实归属匹配（例如最高的柱子上写的是哪个实体名，它应该对应柱底的真实标签）。"
        "用一句话回答：匹配 或 不匹配，并说明原因。",
        cfg,
        temperature=0.0,
    )
    bad = any(k in text for k in ("不匹配", "错位", "不一致", "错误", "不符"))
    if bad:
        return False, text[:300]
    if any(k in text for k in ("匹配", "一致", "正确", "对应")):
        return True, text[:300]
    return False, text[:300]


def _crop_value_region(img: np.ndarray, box: dict[str, Any]) -> np.ndarray:
    x1 = max(0, box["x"] - 20)
    x2 = min(img.shape[1], box["x"] + box["w"] + 20)
    y1 = max(0, box["y"] - 60)
    y2 = min(img.shape[0], box["y"] + 8)
    if y2 <= y1 or x2 <= x1:
        return img[max(0, box["y"] - 40) : box["y"] + 8, x1:x2]
    return img[y1:y2, x1:x2]


def read_bar_values(
    image_path: str | Path,
    aligned: list[dict[str, Any]],
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Read each bar's printed value via one full-frame vision call.

    The vision model reads all printed values left-to-right, which is far more
    reliable than per-bar crops (a short bar's crop can include unrelated text).
    Bars with no value from the full-frame call fall back to a focused crop.
    """
    values: list[str | None] = [None] * len(aligned)
    value_prompt = (
        "这是一个柱状图。请按从左到右的顺序，用「标签: 数值」的格式列出每根柱子的"
        "底部标签和顶部数值，例如 Sub-Saharan Africa: 36.1%。不要解释。"
    )
    attempts: list[str] = []
    for _ in range(3):
        try:
            text = _call_vision(image_path, value_prompt, cfg, temperature=0.0)
        except Exception:
            continue
        attempts.append(text)

    per_bar: list[list[str]] = [[] for _ in aligned]
    for text in attempts:
        assigned: set[int] = set()
        for label, value in _labeled_value_pairs(text):
            norm = _normalize_label(label)
            if not norm:
                continue
            for idx, item in enumerate(aligned):
                if idx in assigned:
                    continue
                item_label = _normalize_label(item.get("label"))
                if item_label and (norm in item_label or item_label in norm):
                    per_bar[idx].append(value)
                    assigned.add(idx)
                    break
        if not assigned:
            tokens = re.findall(r"-?\d+(?:\.\d+)?\s*%?", text)
            percentages = [token for token in tokens if "%" in token]
            sequence = percentages if len(percentages) >= len(aligned) else tokens
            for idx, item in enumerate(aligned):
                if idx >= len(sequence):
                    break
                per_bar[idx].append(sequence[idx])

    for idx, candidates in enumerate(per_bar):
        if candidates:
            values[idx] = max(set(candidates), key=candidates.count)

    img = cv2.imread(str(image_path))
    out: list[dict[str, Any]] = []
    for i, item in enumerate(aligned):
        value_text = values[i]
        if value_text is None and img is not None:
            crop = _crop_value_region(img, item)
            crop_path = Path(image_path).with_name(f"value_crop_{i:02d}.png")
            cv2.imwrite(str(crop_path), crop)
            try:
                crop_text = _call_vision(
                    crop_path,
                    "读出这个裁剪图里的数字或百分比，只返回数字和单位，如 36.1% 或 6.9。",
                    cfg,
                    temperature=0.0,
                )
                match = re.search(r"-?\d+(?:\.\d+)?\s*%?", crop_text)
                value_text = match.group(0).strip() if match else None
            except Exception:
                value_text = None
        out.append({**item, "value_text": value_text})
    return out


def _value_plausibility(item: dict[str, Any], aligned: list[dict[str, Any]]) -> tuple[bool, str]:
    """Check a frame-read value against the chart's printed-value conventions."""
    value = item.get("value")
    if value is None:
        return False, "no value"
    if not (0 <= value <= 100):
        return False, f"value {value:g} outside plausible 0-100 range"
    heights = [float(a["h"]) for a in aligned if a["h"] > 0]
    values = [
        float(a["value"])
        for a in aligned
        if isinstance(a.get("value"), (int, float)) and a["value"] > 0
    ]
    if len(heights) >= 2 and len(values) >= 2:
        max_h = max(heights)
        max_v = max(values)
        value_ratio = value / max_v
        height_ratio = float(item["h"]) / max_h
        tolerance = 0.15 if value < 5 else 0.12
        if abs(value_ratio - height_ratio) > tolerance:
            return False, (
                f"value ratio {value_ratio:.3f} vs height ratio {height_ratio:.3f} "
                "differs beyond tolerance"
            )
    return True, "ok"


def _ratio_consistency(aligned: list[dict[str, Any]]) -> tuple[bool, str]:
    if len(aligned) < 2:
        return True, "too few bars to check"
    heights = [float(a["h"]) for a in aligned]
    maxh = max(heights)
    if maxh <= 0:
        return True, "degenerate bar heights"
    values = [
        float(a["value"])
        for a in aligned
        if isinstance(a.get("value"), (int, float)) and a.get("value")
    ]
    if not values:
        return True, "no numeric values to check"
    maxv = max(values)
    errors = []
    for a in aligned:
        v = a.get("value")
        if not isinstance(v, (int, float)) or v == 0:
            continue
        expected = float(a["h"]) / maxh
        val_ratio = float(v) / maxv
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
        label = f"{a.get('label', '')} {a.get('value_text') or ''}".strip()
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
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    boxes = detect_bars(image_path)
    try:
        vision_order = read_entity_order(image_path, cfg)
    except Exception:
        vision_order = None
    aligned, warnings = match_entities(boxes, entities, vision_order)
    values = read_bar_values(image_path, aligned, cfg)
    for item in values:
        if item.get("value_text"):
            try:
                item["value"] = float(re.sub(r"[^0-9.]", "", item["value_text"]))
            except ValueError:
                item["value"] = None
        else:
            item["value"] = None
    for item in values:
        item["value_plausible"], item["plausibility_message"] = _value_plausibility(item, values)
    implausible = [
        {"entity_id": item.get("entity_id"), "label": item.get("label"), "value_text": item.get("value_text"), "reason": item["plausibility_message"]}
        for item in values
        if not item["value_plausible"]
    ]
    consistent, message = _ratio_consistency(values)
    overlay = out_dir / "aligned_overlay.png"
    svg = out_dir / "semantic_aligned.svg"
    overlay_ok = _render_overlay(image_path, values, overlay)
    svg_ok = _render_aligned_svg(values, svg)
    verified = False
    verify_message = ""
    if overlay_ok:
        try:
            verified, verify_message = verify_alignment(overlay, cfg)
        except Exception as exc:
            verify_message = f"vision verify failed: {exc}"
    report = {
        "clip_id": clip_id,
        "detected_bar_count": len(boxes),
        "matched_count": len(values),
        "warnings": warnings,
        "value_geometry_consistent": consistent,
        "consistency_message": message,
        "implausible_bars": implausible,
        "value_read_method": "vision_full_frame",
        "vision_entity_order": vision_order or [],
        "alignment_verified": verified,
        "alignment_verify_message": verify_message,
        "bars": values,
        "overlay_png": str(overlay),
        "aligned_svg": str(svg),
        "success": bool(values) and overlay_ok and svg_ok,
    }
    write_json(out_dir / "aligned_report.json", report)
    return report
