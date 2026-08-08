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
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .schemas import ensure_dir, write_json


VISION_DEFAULTS = {
    "node_path": os.environ.get("DATAVIDEO_VISION_NODE", "node"),
    "script": os.environ.get("DATAVIDEO_VISION_SCRIPT", "vision.js"),
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
        node_parent = str(Path(v["node_path"]).parent)
        path_prefix = "" if node_parent == "." else node_parent + os.pathsep
        env = {
            **os.environ,
            "PATH": path_prefix + os.environ.get("PATH", ""),
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


def _hue_near(hue: np.ndarray, center: int, tolerance: int) -> np.ndarray:
    """Boolean mask for pixels whose circular hue distance to ``center`` is <= tolerance."""
    # Cast away uint8 first: under NEP 50 (NumPy >= 2) an unsigned subtraction
    # wraps around instead of promoting, which silently truncated the hue band.
    delta = np.abs(hue.astype(np.int32) - center)
    delta = np.minimum(delta, 180 - delta)
    return delta <= tolerance


def _estimate_background_mask(hsv: np.ndarray) -> np.ndarray:
    """Estimate which pixels belong to the chart background.

    The previous heuristic took the dominant hue among *saturated* pixels as
    the background hue and erased everything within +-18 degrees of it. That
    silently deletes the bars whenever the bars themselves are the largest
    saturated region of the frame (e.g. a light-gray chart background with
    solid colored bars, the most common news-graphic layout).

    Instead the background model is built from pixels that are much more
    likely to be background:
      * the outer border strip of the frame (bars almost never touch it), and
      * low-saturation pixels (white/gray/light backgrounds, dark panels).
    When the evidence is achromatic (median saturation < 55) the background is
    a saturation threshold; when it is chromatic (saturated panels, e.g. the
    WeChat test clip) the background is a hue band around the median evidence
    hue, with achromatic pixels always treated as background too.
    """
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    H, W = h.shape
    bw = max(3, W // 40)
    bh = max(3, H // 40)
    border = np.zeros((H, W), dtype=bool)
    border[:bh, :] = True
    border[-bh:, :] = True
    border[:, :bw] = True
    border[:, -bw:] = True

    low_sat = s < 40
    evidence = border | low_sat
    if evidence.sum() == 0:
        # Edge-to-edge saturated frame with no achromatic pixels: fall back to
        # the dominant saturated hue (legacy behaviour).
        sel = (s > 40) & (v > 40)
        if sel.sum() == 0:
            return border
        hist, _ = np.histogram(h[sel], bins=180, range=(0, 180))
        bg_hue = int(np.argmax(hist))
        return _hue_near(h, bg_hue, 18) | border

    if float(np.median(s[evidence])) < 55:
        # Achromatic background (white / light gray / dark panel): the bars are
        # whatever saturated pixels remain.
        sat_tol = max(55, int(np.median(s[evidence])) + 35)
        return (s < sat_tol) | border

    # Chromatic background: circular median hue of the border + achromatic
    # evidence (the panel colour), then a hue band around it.
    hist, _ = np.histogram(h[evidence], bins=180, range=(0, 180))
    cum = np.cumsum(hist)
    bg_hue = int(np.searchsorted(cum, max(1, cum[-1] // 2)))
    return _hue_near(h, bg_hue, 18) | (s < 30) | border


def detect_bars(image_path: str | Path) -> list[dict[str, Any]]:
    """Detect colored bar regions in a chart keyframe via color segmentation.

    Short bars (e.g. a 1% bar only a few pixels tall) are kept, so the height
    threshold is low; components that do not sit on the dominant baseline of
    the tall bars (typically value circles or category-label text) are dropped.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    H, W = img.shape[:2]

    bg = _estimate_background_mask(hsv)
    fg = ((s > 50) & (v > 50) & (~bg)).astype(np.uint8) * 255
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
            # Use the most common bottom position (rounded to 10 px) as the
            # baseline instead of the median: a median can be pulled away by a
            # couple of non-bar components (e.g. value circles above bars),
            # which then makes every real bar look off-baseline.
            bottoms = np.array([b["y"] + b["h"] for b in tall])
            rounded = np.round(bottoms / 10.0).astype(np.int64)
            counts = np.bincount(rounded - rounded.min())
            peak = rounded.min() + int(np.argmax(counts))
            baseline = float(peak * 10)
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
    normalized = re.sub(r"[^a-z0-9]+", "", str(text).lower())
    aliases = {
        "eu": "europeanunion",
        "usa": "unitedstates",
        "us": "unitedstates",
        "uk": "unitedkingdom",
    }
    return aliases.get(normalized, normalized)


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


def _text_width_estimate(text: str, font_size: float) -> float:
    return max(10.0, len(str(text)) * 0.55 * font_size)


def _contrast_outline_color(img: np.ndarray, box: list[int]) -> tuple[int, int, int]:
    """Pick white on dark surroundings and black on light surroundings so the
    text outline stays visible on any chart background."""
    x1, y1, x2, y2 = [int(v) for v in box]
    H, W = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return (255, 255, 255)
    pad = 4
    top = img[max(0, y1 - pad) : y1, max(0, x1 - pad) : min(W, x2 + pad)]
    bottom = img[min(H, y2) : min(H, y2 + pad), max(0, x1 - pad) : min(W, x2 + pad)]
    left = img[max(0, y1 - pad) : min(H, y2 + pad), max(0, x1 - pad) : x1]
    right = img[max(0, y1 - pad) : min(H, y2 + pad), x2 : min(W, x2 + pad)]
    parts = [part.reshape(-1, 3) for part in (top, bottom, left, right) if part.size]
    if not parts:
        return (255, 255, 255)
    mean = np.concatenate(parts).mean(axis=0)
    lum = 0.299 * mean[2] + 0.587 * mean[1] + 0.114 * mean[0]
    return (20, 20, 20) if lum > 128 else (255, 255, 255)


def _text_line_boxes(
    img: np.ndarray,
    *,
    detect_threshold: int = 40,
    ratio_threshold: int = 50,
) -> list[dict[str, Any]]:
    """Detect horizontal text-line bounding boxes in the whole frame.

    Text strokes produce dense, small gradient blobs; long flat components
    (bar edges, axes, dividers) are dropped, and the remaining components are
    clustered into text lines by vertical overlap.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    mask = (grad > detect_threshold).astype(np.uint8) * 255
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    comps = []
    for i in range(1, n):
        cx, cy, cw, ch, area = stats[i]
        if ch < 8 or ch > 120 or cw < 5 or area < 20:
            continue
        if cw > 10 * max(ch, 8):
            continue
        comps.append([int(cx), int(cy), int(cw), int(ch)])
    comps.sort(key=lambda c: c[1])
    lines: list[dict[str, int]] = []
    for cx, cy, cw, ch in comps:
        placed = False
        for ln in lines:
            if cy <= ln["y2"] + 2 and cy + ch >= ln["y1"] - 2:
                gap = max(0, cx - ln["x2"], ln["x1"] - (cx + cw))
                if gap <= 50:
                    ln["x1"] = min(ln["x1"], cx)
                    ln["x2"] = max(ln["x2"], cx + cw)
                    ln["y1"] = min(ln["y1"], cy)
                    ln["y2"] = max(ln["y2"], cy + ch)
                    placed = True
                    break
        if not placed:
            lines.append({"x1": cx, "y1": cy, "x2": cx + cw, "y2": cy + ch})
    for ln in lines:
        area = (ln["x2"] - ln["x1"]) * (ln["y2"] - ln["y1"])
        strong = (grad > ratio_threshold).astype(np.uint8) * 255
        ln["ratio"] = (
            float(strong[ln["y1"] : ln["y2"], ln["x1"] : ln["x2"]].sum() / 255) / area
            if area
            else 0.0
        )
    return lines


def _tighten_text_line(mask: np.ndarray, line: dict[str, int]) -> list[int]:
    """Narrow a text line to its dense text pixels (drops circle outlines,
    bar edges and other graphics that merged into the line)."""
    x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
    sub = mask[y1:y2, x1:x2] > 0
    if sub.size == 0:
        return [x1, y1, x2, y2]
    col = sub.sum(axis=0)
    row = sub.sum(axis=1)
    cth = max(1, int(col.max() * 0.25))
    rth = max(1, int(row.max() * 0.25))
    cols = np.where(col >= cth)[0]
    rows = np.where(row >= rth)[0]
    if len(cols) == 0 or len(rows) == 0:
        return [x1, y1, x2, y2]
    return [
        x1 + int(cols.min()),
        y1 + int(rows.min()),
        x1 + int(cols.max()) + 1,
        y1 + int(rows.max()) + 1,
    ]


def locate_text_boxes(
    image_path: str | Path,
    aligned: list[dict[str, Any]],
    cfg: dict[str, Any] | None = None,
) -> dict[str, dict[str, list[int]]]:
    """Locate the original printed value/label text boxes for each (vertical)
    bar using pure CV geometry.

    Text lines are detected from gradient edges across the whole frame; for
    each bar the value box is the text line horizontally aligned with the bar
    whose bottom sits just above/at the bar top, and the label box is the
    line just below the baseline.  Deterministic, no extra model calls, and
    independent of text color or chart theme.
    """
    result: dict[str, dict[str, list[int]]] = {}
    img = cv2.imread(str(image_path))
    if img is None or not aligned:
        return result
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    textmask = (grad > 50).astype(np.uint8) * 255
    lines = _text_line_boxes(img)

    def _span_hits(ln: dict[str, int], x: int, w: int) -> bool:
        center = (ln["x1"] + ln["x2"]) / 2
        return x - 15 <= center <= x + w + 15

    for item in aligned:
        eid = str(item.get("entity_id") or "")
        x, y, w, hh = item.get("x"), item.get("y"), item.get("w"), item.get("h")
        if not eid or None in (x, y, w, hh):
            continue
        x, y, w, hh = int(x), int(y), int(w), int(hh)
        boxes = result.setdefault(eid, {})
        baseline = y + hh
        value_zone_end = y + max(40, min(130, int(hh * 0.6)))
        value_candidates = [
            ln
            for ln in lines
            if y - 120 <= ln["y2"] <= value_zone_end and _span_hits(ln, x, w)
        ]
        label_candidates = [
            ln
            for ln in lines
            if baseline + 2 <= ln["y1"] <= baseline + 100 and _span_hits(ln, x, w)
        ]
        if value_candidates:
            solid = [ln for ln in value_candidates if ln["ratio"] >= 0.08]
            pool = solid or value_candidates
            ln = max(pool, key=lambda ln: ln["y2"])
            box = _tighten_text_line(textmask, ln)
            boxes["value_box"] = [max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])]
        if label_candidates:
            solid = [ln for ln in label_candidates if ln["ratio"] >= 0.08]
            pool = solid or label_candidates
            ln = min(pool, key=lambda ln: (ln["y1"], -ln["ratio"]))
            x1, y1, x2, y2 = ln["x1"], ln["y1"], ln["x2"], ln["y2"]
            # Multi-line labels: extend the box down over the wrapped second
            # line(s) of the same label (e.g. "Sub-Saharan" / "Africa").
            for other in lines:
                if other is ln or other["y1"] < y1 - 4:
                    continue
                if baseline + 2 <= other["y1"] <= y2 + 35:
                    if other["x2"] >= x - 15 and other["x1"] <= x + w + 15:
                        x1 = min(x1, other["x1"])
                        x2 = max(x2, other["x2"])
                        y2 = max(y2, other["y2"])
            box = _tighten_text_line(
                textmask, {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "ratio": 1.0}
            )
            boxes["label_box"] = [max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])]
    return result


def _render_overlay(
    image_path: str | Path,
    aligned: list[dict[str, Any]],
    out: Path,
    text_boxes: dict[str, dict[str, list[int]]] | None = None,
) -> bool:
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(img)
    bar_color = (230, 25, 75)
    for a in aligned:
        d.rectangle([a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]], outline=bar_color, width=3)
        boxes = (text_boxes or {}).get(str(a.get("entity_id") or "")) or {}
        for key in ("value_box", "label_box"):
            box = boxes.get(key)
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            if x2 <= x1 or y2 <= y1 or x2 > img.width or y2 > img.height:
                continue
            outline = _contrast_outline_color(np.asarray(img), (x1, y1, x2, y2))
            d.rectangle([x1, y1, x2, y2], outline=outline, width=3)
    img.save(out)
    return out.exists()


def _render_aligned_svg(
    aligned: list[dict[str, Any]],
    out: Path,
    text_boxes: dict[str, dict[str, list[int]]] | None = None,
) -> bool:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720" data-role="semantic-chart" data-generator="datavideo.cv_align_v1">',
    ]
    for i, a in enumerate(aligned):
        eid = re.sub(r"[^a-z0-9]+", "-", str(a.get("entity_id") or "").lower()).strip("-") or f"bar-{i}"
        value = a.get("value_text") or ""
        boxes = (text_boxes or {}).get(str(a.get("entity_id") or "")) or {}
        value_box_attr = (
            f' data-value-box="{",".join(str(int(v)) for v in boxes["value_box"])}"'
            if boxes.get("value_box")
            else ""
        )
        label_box_attr = (
            f' data-label-box="{",".join(str(int(v)) for v in boxes["label_box"])}"'
            if boxes.get("label_box")
            else ""
        )
        lines.append(f'<g id="entity-{eid}" data-role="entity" data-entity-id="{eid}" data-label="{html.escape(str(a.get("label") or ""))}">')
        lines.append(
            f'<rect id="{eid}-bar" data-role="bar" data-entity-id="{eid}" data-value="{html.escape(value)}" '
            f'x="{a["x"]}" y="{a["y"]}" width="{a["w"]}" height="{a["h"]}" '
            'data-animation-property="height" data-anchor="bottom" fill="#3cb44b"/>'
        )
        if value:
            vw = max(52.0, _text_width_estimate(value, 20) + 18)
            vx = a["x"] + a["w"] / 2 - vw / 2
            vy = max(0, a["y"] - 36)
            lines.append(
                f'<rect id="{eid}-value-box" data-role="value-box" data-entity-id="{eid}" '
                f'x="{vx:.1f}" y="{vy:.1f}" width="{vw:.1f}" height="28" fill="#ffffff" stroke="#333333" stroke-width="2"{value_box_attr}/>'
            )
            lines.append(
                f'<text data-role="value-label" x="{a["x"] + a["w"] / 2}" y="{vy + 21}" '
                f'text-anchor="middle" font-size="20" font-weight="700">{html.escape(value)}</text>'
            )
        label = str(a.get("label") or "")
        if label:
            baseline = a["y"] + a["h"]
            lw = max(float(a["w"]), _text_width_estimate(label, 16) + 18)
            lx = a["x"] + a["w"] / 2 - lw / 2
            ly = baseline + 8
            lines.append(
                f'<rect id="{eid}-label-box" data-role="category-box" data-entity-id="{eid}" '
                f'x="{lx:.1f}" y="{ly:.1f}" width="{lw:.1f}" height="28" fill="#ffffff" stroke="#333333" stroke-width="2"{label_box_attr}/>'
            )
            lines.append(
                f'<text data-role="category-label" x="{a["x"] + a["w"] / 2}" y="{ly + 21}" '
                f'text-anchor="middle" font-size="16">{html.escape(label)}</text>'
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
    text_boxes: dict[str, dict[str, list[int]]] = {}
    try:
        text_boxes = locate_text_boxes(image_path, values, cfg)
    except Exception as exc:
        warnings.append(f"text box localization failed: {exc}")
    for item in values:
        item["value_box"] = (text_boxes.get(str(item.get("entity_id") or "")) or {}).get("value_box")
        item["label_box"] = (text_boxes.get(str(item.get("entity_id") or "")) or {}).get("label_box")
    overlay_ok = _render_overlay(image_path, values, overlay, text_boxes)
    svg_ok = _render_aligned_svg(values, svg, text_boxes)
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
