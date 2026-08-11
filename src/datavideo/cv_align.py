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

    Orientation is inferred from the candidate geometry: vertical bars share a
    bottom baseline (height encodes the value), horizontal bars share a left
    (or right) edge and vary in width.  Each returned bar carries an
    ``orientation`` field so downstream steps can dispatch correctly.
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

    orientation = _classify_bar_orientation(candidates)
    if orientation == "horizontal":
        candidates = _keep_horizontal_bars(candidates)
    elif orientation == "vertical":
        candidates = _keep_vertical_bars(candidates)
    for b in candidates:
        b["orientation"] = orientation
    return candidates


def _classify_bar_orientation(candidates: list[dict[str, Any]]) -> str:
    """Decide whether candidate components look like vertical or horizontal
    bars: vertical bars share a bottom baseline; horizontal bars share a left
    (or right) edge and vary in width."""
    if len(candidates) < 2:
        return "vertical"
    tall = [b for b in candidates if b["h"] >= 25]
    if len(tall) >= 2:
        bottoms = np.array([b["y"] + b["h"] for b in tall])
        if float(np.ptp(bottoms)) <= 30:
            return "vertical"
    wide = [b for b in candidates if b["w"] >= 25 and 8 <= b["h"] <= 150]
    if len(wide) >= 2:
        for key in ("left", "right"):
            values = np.array([b["x"] if key == "left" else b["x"] + b["w"] for b in wide])
            rounded = np.round(values / 10.0).astype(np.int64)
            counts = np.bincount(rounded - rounded.min())
            peak = rounded.min() + int(np.argmax(counts))
            members = [b for b in wide if abs(round((b["x"] if key == "left" else b["x"] + b["w"]) / 10.0) - peak) <= 2]
            if len(members) >= 2:
                widths = np.array([b["w"] for b in members])
                if float(np.ptp(widths)) >= max(20.0, 0.2 * float(widths.max())):
                    return "horizontal"
    return "vertical"


def _keep_vertical_bars(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep components sitting on the dominant bottom baseline (vertical bars),
    dropping value circles / label text / other floating components."""
    if not candidates:
        return []
    tall = [b for b in candidates if b["h"] >= 25]
    if tall:
        # Use the most common bottom position (rounded to 10 px) as the
        # baseline instead of the median: a median can be pulled away by a
        # couple of non-bar components (e.g. value circles above bars).
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


def _keep_horizontal_bars(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep components that form horizontal bars: the majority share a common
    left (or right) edge and have a similar thickness; sort top-to-bottom."""
    if not candidates:
        return []
    wide = [b for b in candidates if b["w"] >= 25 and 8 <= b["h"] <= 150]
    if not wide:
        return []
    kept = []
    for key in ("left", "right"):
        values = np.array([b["x"] if key == "left" else b["x"] + b["w"] for b in wide])
        rounded = np.round(values / 10.0).astype(np.int64)
        counts = np.bincount(rounded - rounded.min())
        peak = rounded.min() + int(np.argmax(counts))
        members = [b for b in wide if abs(round((b["x"] if key == "left" else b["x"] + b["w"]) / 10.0) - peak) <= 2]
        if len(members) > len(kept):
            kept = members
    if len(kept) < 2:
        return []
    # Similar thickness (tolerant: labels may merge with bars, making one bar
    # noticeably thicker), then drop non-bar fragments that are far shorter
    # than the longest bar.
    thickness = np.array([b["h"] for b in kept])
    med = float(np.median(thickness))
    kept = [b for b in kept if med * 0.5 <= b["h"] <= med * 2.5]
    maxw = max((b["w"] for b in kept), default=0)
    if maxw > 0:
        kept = [b for b in kept if b["w"] >= max(25.0, 0.12 * maxw)]
    kept.sort(key=lambda b: b["y"])
    return kept


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


def _labels_match(want: str, label: str) -> bool:
    """Match two raw labels by token sets (case-insensitive).

    Substring matching is too loose: "insuredunitedstates" is a substring of
    "uninsuredunitedstates", so a recovered "Insured United States" swallows
    the "Uninsured United States" bar. Token-set matching keeps such labels
    distinct while still tolerating minor OCR noise (extra/duplicated words).
    Tokens must be extracted from the raw labels (before normalization strips
    the spaces), otherwise "insuredunitedstates" collapses into one token.
    """
    want_tokens = set(re.findall(r"[a-z0-9]+", str(want).lower()))
    label_tokens = set(re.findall(r"[a-z0-9]+", str(label).lower()))
    if not want_tokens or not label_tokens:
        return False
    return want_tokens <= label_tokens or label_tokens <= want_tokens


def _labeled_value_pairs(text: str) -> list[tuple[str, str]]:
    """Extract ``Label: value`` pairs from a vision response."""
    pairs: list[tuple[str, str]] = []
    for match in re.finditer(r"([A-Za-z][A-Za-z0-9 $&'\-\.\,]*?):\s*(-?\d+(?:\.\d+)?\s*%?)", text):
        label = _clean_vision_label(match.group(1))
        value_match = re.search(r"-?\d+(?:\.\d+)?\s*%?", match.group(2))
        if label and value_match:
            pairs.append((label, value_match.group(0).strip()))
    return pairs


def _parse_label_json(text: str) -> list[str]:
    """Extract a JSON array of label strings (or {"label": ...} objects) from
    a vision response, tolerating ```json fences and prose."""
    cleaned = re.sub(r"```(?:json)?", "", text)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except Exception:
        return []
    labels: list[str] = []
    for item in parsed if isinstance(parsed, list) else []:
        if isinstance(item, str):
            labels.append(item)
        elif isinstance(item, dict) and item.get("label"):
            labels.append(str(item["label"]))
    return [str(label).strip() for label in labels if str(label).strip()]


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
                    if label == norm_want or (
                        label and _labels_match(want, str(e.get("label") or ""))
                    ):
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


def read_entity_order(
    image_path: str | Path,
    cfg: dict[str, Any] | None = None,
    orientation: str = "vertical",
) -> list[str]:
    """Ask the vision model for the category labels of the bars, ordered along
    the categorical axis (left-to-right for vertical bars, top-to-bottom for
    horizontal bars)."""
    if orientation == "horizontal":
        prompt = (
            "这是横向条形图的一帧。请从上到下依次列出每根条形左侧的类别名称。"
            '只返回 JSON 数组，格式 [{"label":"类别名"}, ...]，'
            '类别名必须保持完整（例如 "Less than $20,000"，里面的逗号是数字分隔符，绝不能拆分），不要额外解释。'
        )
    else:
        prompt = (
            "这是柱状图的一帧。请从左到右依次列出每根柱子底部的类别标签，"
            "只要名字本身，用逗号分隔，不要额外解释。"
        )
    text = _call_vision(
        image_path,
        prompt,
        cfg,
        temperature=0.0,
    )
    if orientation == "horizontal":
        json_labels = _parse_label_json(text)
        if json_labels:
            return [_clean_vision_label(label) for label in json_labels]
    labels = []
    skip = {"从左到右", "依次为", "第一根", "第二根", "第三根", "第四根", "柱子", "类别", "标签"}
    for part in re.split(r"[,，\n;；、]", text):
        part = _clean_vision_label(part)
        if part and part.lower() not in skip:
            labels.append(part)
    return labels


def read_frame_title(
    image_path: str | Path,
    cfg: dict[str, Any] | None = None,
) -> str:
    """Read the chart title printed in the frame via the vision model.

    Used when the VLM data recovery missed the title (its visible_text has no
    title candidate at all, e.g. bar_29 "Monthly price of Advair asthma
    inhaler" while the recovered title was the video title).
    """
    text = _call_vision(
        image_path,
        "读出这张图表顶部的主标题文字（忽略副标题和数据来源行），只返回标题本身，不要解释。",
        cfg,
        temperature=0.0,
    )
    title = text.strip().strip("\"'`。，,.")
    if len(title) > 120:
        title = title[:120]
    return title


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
    orientation: str = "vertical",
) -> list[dict[str, Any]]:
    """Read each bar's printed value via one full-frame vision call.

    The vision model reads all printed values left-to-right, which is far more
    reliable than per-bar crops (a short bar's crop can include unrelated text).
    Bars with no value from the full-frame call fall back to a focused crop.
    """
    values: list[str | None] = [None] * len(aligned)
    if orientation == "horizontal":
        value_prompt = (
            "这是横向条形图。请找出画面中**实际印刷了数值**的条形，用「标签: 数值」的格式"
            "列出它们的左侧完整名称和右端数值（例如 Less than $20,000: 890）。"
            "没有印刷数值的条形**绝对不能写数值、绝对不能编造**，只列有数值的条形即可；"
            "宁可少报，不要瞎编。不要解释。"
        )
    else:
        value_prompt = (
            "这是柱状图的一帧。请找出画面中**实际印刷了数值**的柱子，用「标签: 数值」"
            "的格式列出它们的底部标签和顶部数值（例如 Sub-Saharan Africa: 36.1%）。"
            "没有印刷数值的柱子**绝对不能写数值、绝对不能编造**，只列有数值的柱子即可；"
            "宁可少报，不要瞎编。不要解释。"
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
        if not assigned and orientation != "horizontal":
            tokens = re.findall(r"-?\d+(?:\.\d+)?\s*%?", text)
            percentages = [token for token in tokens if "%" in token]
            sequence = percentages if len(percentages) >= len(aligned) else tokens
            for idx, item in enumerate(aligned):
                if idx >= len(sequence):
                    break
                per_bar[idx].append(sequence[idx])

    for idx, candidates in enumerate(per_bar):
        if candidates:
            majority = max(set(candidates), key=candidates.count)
            count = candidates.count(majority)
            # Only trust a value when at least two of three attempts agree;
            # a value seen once is likely a hallucination (e.g. an unlabeled
            # bar given a neighbour's number) and should be left for the
            # scale estimation instead.
            if len(attempts) >= 2 and count < 2:
                values[idx] = None
            else:
                values[idx] = majority

    img = cv2.imread(str(image_path))
    # A full-frame read that found no printed value at all is a strong signal
    # that this chart has no printed values (e.g. an unlabeled bar chart with
    # only an axis). Skip the per-bar crop fallback in that case: it would
    # burn one vision call per bar for nothing and the axis-tick estimation
    # should fill the values instead.
    any_full_frame_value = any(values)
    out: list[dict[str, Any]] = []
    for i, item in enumerate(aligned):
        value_text = values[i]
        if value_text is None and img is not None and any_full_frame_value:
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
    value_text = str(item.get("value_text") or "")
    if value < 0:
        return False, f"value {value:g} is negative"
    if "%" in value_text and value > 100:
        return False, f"value {value:g} outside plausible 0-100 range for percentages"
    if item.get("value_read_verified"):
        # Directly printed values are trusted over noisy bar-length geometry
        # (detection can under/over-measure a bar by several percent), so the
        # ratio check is skipped for them.
        return True, "ok"
    lengths = [_bar_length(a) for a in aligned if _bar_length(a) > 0]
    values = [
        float(a["value"])
        for a in aligned
        if isinstance(a.get("value"), (int, float)) and a["value"] > 0
    ]
    if len(lengths) >= 2 and len(values) >= 2:
        max_len = max(lengths)
        max_v = max(values)
        value_ratio = value / max_v
        length_ratio = _bar_length(item) / max_len
        tolerance = 0.15 if value < 5 else 0.12
        if abs(value_ratio - length_ratio) > tolerance:
            return False, (
                f"value ratio {value_ratio:.3f} vs bar length ratio {length_ratio:.3f} "
                "differs beyond tolerance"
            )
    return True, "ok"


def _ratio_consistency(aligned: list[dict[str, Any]]) -> tuple[bool, str]:
    if len(aligned) < 2:
        return True, "too few bars to check"
    lengths = [_bar_length(a) for a in aligned]
    max_len = max(lengths)
    if max_len <= 0:
        return True, "degenerate bar lengths"
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
        expected = _bar_length(a) / max_len
        val_ratio = float(v) / maxv
        if abs(val_ratio - expected) > 0.12:
            errors.append(f"{a['label']}: value ratio {val_ratio:.2f} vs bar length ratio {expected:.2f}")
    if errors:
        return False, "; ".join(errors)
    return True, "ok"


def _bar_length(item: dict[str, Any]) -> float:
    """The dimension that encodes the value: height for vertical bars, width
    for horizontal bars."""
    if str(item.get("orientation")) == "horizontal":
        return float(item.get("w") or 0.0)
    return float(item.get("h") or 0.0)


def estimate_unlabeled_values(
    aligned: list[dict[str, Any]],
    chart_type: str = "bar",
) -> int:
    """Estimate values for marks without printed values from the linear scale
    implied by the marks that do have printed values.

    The general rule: whatever geometric dimension encodes the value (bar
    length/height, line/point position along the value axis, pie arc angle),
    a linear calibration over the labeled marks maps that dimension back to
    values for the unlabeled ones.

    Currently implemented:
      * bar (vertical and horizontal): value ~ bar length (height or width).
    Estimated marks are flagged with ``value_estimated`` / ``value_type`` and
    carry a lower confidence + ``needs_review`` when reconciled.
    """
    if chart_type not in ("bar", "combined"):
        return 0
    labeled: list[tuple[float, float]] = []
    for item in aligned:
        value = item.get("value")
        length = _bar_length(item)
        if length > 0 and isinstance(value, (int, float)) and item.get("value_text"):
            labeled.append((length, float(value)))
    if len(labeled) < 2:
        return 0
    xs = np.array([x for x, _ in labeled], dtype=float)
    ys = np.array([y for _, y in labeled], dtype=float)
    try:
        slope, intercept = np.polyfit(xs, ys, 1)
    except Exception:
        return 0
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        return 0
    count = 0
    for item in aligned:
        if item.get("value") is not None:
            continue
        length = _bar_length(item)
        if length <= 0:
            continue
        est = float(intercept + slope * length)
        if est < 0 or not np.isfinite(est):
            est = 0.0
        item["value"] = est
        item["value_text"] = f"{est:.0f}"
        item["value_estimated"] = True
        item["value_type"] = "estimated"
        item["plausibility_message"] = "estimated from labeled-bar scale"
        count += 1
    return count


def detect_axis_tick_marks(
    image_path: str | Path,
    orientation: str = "vertical",
) -> list[dict[str, Any]]:
    """Detect value-axis tick positions.

    Returns a list of ``{"coord": float}`` sorted along the value axis
    (``coord`` is the tick's y for vertical bars, x for horizontal bars).
    Many real charts draw no axis line at all — they only have dashed grid
    lines plus tick labels (e.g. Vox's drug-spending chart). We therefore
    first look for thin horizontal/vertical lines spanning the plot (grid
    lines and the baseline); when none are found we fall back to detecting
    short tick strokes next to a long axis line.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.int16)
    if float(np.median(gray)) < 128:
        foreground = (gray > 127) & (saturation < 100)
    else:
        foreground = (gray < 215) & (saturation < 100)
    binary = np.where(foreground, 255, 0).astype(np.uint8)
    height, width = binary.shape

    # Pass 1: thin lines spanning the plot (grid lines / baseline). Detection
    # is restricted to the plot area (right 75% for vertical bars, middle 75%
    # height for horizontal bars) so tick labels and the title never count;
    # bands thicker than a few pixels are text, not grid lines.
    if orientation == "horizontal":
        y_start = int(height * 0.12)
        y_end = int(height * 0.88)
        column_spans = []
        for x in range(width):
            indices = np.where(binary[y_start:y_end, x] > 0)[0]
            if indices.size == 0:
                continue
            span = int(indices.max()) - int(indices.min())
            if span >= 0.45 * (y_end - y_start):
                column_spans.append(x)
        bands = _cluster_consecutive(column_spans, gap=4)
        coords = []
        for band in bands:
            if len(band) <= 10:
                coords.append(float(sum(band)) / len(band))
    else:
        x_start = int(width * 0.25)
        row_spans = []
        for y in range(height):
            indices = np.where(binary[y, x_start:] > 0)[0]
            if indices.size == 0:
                continue
            span = int(indices.max()) - int(indices.min())
            if span >= 0.4 * (width - x_start):
                row_spans.append(y)
        bands = _cluster_consecutive(row_spans, gap=4)
        coords = []
        for band in bands:
            if len(band) <= 8:
                coords.append(float(sum(band)) / len(band))
    if coords:
        return [{"coord": coord} for coord in sorted(coords)]

    # Pass 2 (fallback): short tick strokes next to a long axis line.
    if orientation == "horizontal":
        # Axis is the bottom horizontal line; ticks are short vertical strokes.
        long_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        stroke_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 9))
        long_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, long_kernel)
        strokes = cv2.morphologyEx(binary, cv2.MORPH_OPEN, stroke_kernel)
        row_scores = long_lines.sum(axis=1)
        axis_rows = [
            row
            for row in range(height)
            if row_scores[row] > width * 0.12
        ]
        if not axis_rows:
            return []
        axis_y = float(np.median(axis_rows))
        contours, _ = cv2.findContours(strokes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coords: list[float] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h < 6 or h > 34 or w > 12:
                continue
            if abs((y + h / 2) - axis_y) > 22:
                continue
            coords.append(x + w / 2)
    else:
        # Axis is the left vertical line; ticks are short horizontal strokes.
        long_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        stroke_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 2))
        long_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, long_kernel)
        strokes = cv2.morphologyEx(binary, cv2.MORPH_OPEN, stroke_kernel)
        col_scores = long_lines.sum(axis=0)
        axis_cols = [
            col
            for col in range(width)
            if col_scores[col] > height * 0.12
        ]
        if not axis_cols:
            return []
        axis_x = float(np.median(axis_cols))
        contours, _ = cv2.findContours(strokes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coords: list[float] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 6 or w > 34 or h > 12:
                continue
            if abs((x + w / 2) - axis_x) > 22:
                continue
            coords.append(y + h / 2)

    coords = sorted(set(round(coord, 1) for coord in coords))
    merged: list[float] = []
    for coord in coords:
        if not merged or abs(coord - merged[-1]) > 6.0:
            merged.append(coord)
    return [{"coord": coord} for coord in merged]


def _cluster_consecutive(values: list[int], *, gap: int) -> list[list[int]]:
    """Group sorted ints into consecutive bands (gaps <= ``gap`` are merged)."""
    bands: list[list[int]] = []
    for value in sorted(values):
        if bands and value - bands[-1][-1] <= gap:
            bands[-1].append(value)
        else:
            bands.append([value])
    return bands


def _parse_tick_labels(text: str) -> tuple[list[float], str]:
    """Parse a vision JSON array of axis tick labels into numbers + unit."""
    unit = "$" if "$" in text else ("%" if "%" in text else "")
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return [], unit
    try:
        raw = json.loads(match.group(0))
    except Exception:
        return [], unit
    if not isinstance(raw, list):
        return [], unit
    labels: list[float] = []
    for entry in raw:
        text_entry = str(entry).strip()
        digits = re.sub(r"[^0-9.\-]", "", text_entry)
        if not digits:
            return [], unit
        try:
            labels.append(float(digits))
        except ValueError:
            return [], unit
    return labels, unit


def read_tick_labels(
    image_path: str | Path,
    cfg: dict[str, Any] | None = None,
    orientation: str = "vertical",
) -> tuple[list[float], str]:
    """Read the value-axis tick labels via one full-frame vision call.

    Returns ``(labels, unit)`` where labels are ordered along the value axis
    (bottom-to-top for vertical bars, left-to-right for horizontal bars).
    """
    if orientation == "horizontal":
        prompt = (
            "这是横向条形图。请读出底部横轴（数值轴）上的刻度标签，"
            "从左到右依次列出，只返回 JSON 数字数组，例如 [0, 100, 200]。"
            '若标签带 "$" 或 "%" 等符号请保留，例如 ["$0", "$100"]。不要解释。'
        )
    else:
        prompt = (
            "这是柱状图。请读出左侧纵轴（数值轴）上的刻度标签，"
            "从下到上依次列出，只返回 JSON 数字数组，例如 [0, 10, 20]。"
            '若标签带 "$" 或 "%" 等符号请保留，例如 ["0%", "10%"]。不要解释。'
        )
    text = _call_vision(image_path, prompt, cfg, temperature=0.0)
    return _parse_tick_labels(text)


def _pair_ticks_with_labels(
    tick_marks: list[dict[str, Any]],
    labels: list[float],
    orientation: str = "vertical",
) -> list[dict[str, Any]]:
    """Pair detected tick coords with vision-read labels in value order."""
    if not tick_marks or len(labels) != len(tick_marks):
        return []
    reverse = orientation != "horizontal"
    ordered = sorted(tick_marks, key=lambda item: float(item["coord"]), reverse=reverse)
    return [
        {"coord": float(item["coord"]), "value": float(value)}
        for item, value in zip(ordered, labels)
    ]


def _infer_baseline_coord(
    aligned: list[dict[str, Any]],
    orientation: str = "vertical",
) -> float | None:
    """Infer the value-axis baseline (the 0/start anchor) from bar geometry.

    Vertical bars share a bottom edge; horizontal bars share a left edge
    (or right edge for right-aligned charts, handled by orientation of the
    majority). This anchor is usually missing from grid-line detection
    because the bars' own pixels merge with the baseline.
    """
    if orientation == "horizontal":
        xs = [
            float(item["x"])
            for item in aligned
            if item.get("x") is not None and _bar_length(item) > 0
        ]
        return min(xs) if xs else None
    bottoms = [
        float(item["y"]) + float(item["h"])
        for item in aligned
        if item.get("y") is not None and item.get("h") is not None and float(item["h"]) > 0
    ]
    return max(bottoms) if bottoms else None


def bar_layout_regularity(bars: list[dict[str, Any]]) -> float:
    """Score how regularly bars are laid out along the category axis (0..1).

    A clean chart has evenly spaced, same-width bars. A cross-fade frame
    (two charts superimposed, e.g. a Vox transition) shows duplicated and
    misaligned bars: irregular gaps along the category axis and inconsistent
    widths for vertical charts. Such frames must never become keyframes
    because their tick marks and bars come from two different charts.
    """
    if len(bars) < 3:
        return 1.0
    first = bars[0]
    orientation = str(first.get("orientation") or "")
    if not orientation:
        orientation = "horizontal" if float(first.get("w") or 0.0) >= float(first.get("h") or 0.0) else "vertical"
    if orientation == "horizontal":
        positions = sorted(float(bar.get("y") or 0.0) for bar in bars)
        widths = [float(bar.get("w") or 0.0) for bar in bars]
    else:
        positions = sorted(float(bar.get("x") or 0.0) for bar in bars)
        widths = [float(bar.get("w") or 0.0) for bar in bars]
    gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    if not gaps or max(gaps) <= 0:
        return 1.0
    mean_gap = sum(gaps) / len(gaps)
    gap_cv = (sum((gap - mean_gap) ** 2 for gap in gaps) / len(gaps)) ** 0.5 / mean_gap
    mean_width = sum(widths) / len(widths) if widths else 0.0
    width_cv = (
        (sum((width - mean_width) ** 2 for width in widths) / len(widths)) ** 0.5 / mean_width
        if mean_width
        else 1.0
    )
    if orientation == "horizontal":
        # Width encodes the value for horizontal bars; only spacing matters.
        return max(0.0, 1.0 - gap_cv * 3.0)
    return max(0.0, 1.0 - (gap_cv + width_cv) * 2.0)


def estimate_unlabeled_values_from_ticks(
    aligned: list[dict[str, Any]],
    tick_marks: list[dict[str, Any]],
) -> int:
    """Estimate unlabeled bar values from value-axis tick marks.

    ``tick_marks`` are ``{"coord", "value"}`` pairs along the value axis (y
    for vertical bars, x for horizontal bars). Values are linearly
    interpolated between adjacent ticks; bars that extend beyond the drawn
    axis (e.g. the US bar in the drug-price charts) are extrapolated along the
    tick scale instead of being clamped to the last tick value.
    """
    ticks = sorted(
        (
            (float(item["coord"]), float(item["value"]))
            for item in tick_marks
            if item.get("coord") is not None and item.get("value") is not None
        ),
        key=lambda pair: pair[0],
    )
    if len(ticks) < 2:
        return 0
    coords = np.array([pair[0] for pair in ticks], dtype=float)
    values = np.array([pair[1] for pair in ticks], dtype=float)
    if np.ptp(coords) == 0 or np.ptp(values) == 0 or not np.all(np.isfinite(values)):
        return 0
    slope = (values[-1] - values[0]) / (coords[-1] - coords[0])
    count = 0
    for item in aligned:
        if item.get("value") is not None:
            continue
        length = _bar_length(item)
        if length <= 0:
            continue
        if str(item.get("orientation") or "") == "horizontal":
            coord = float(item.get("x") or 0.0) + length
        else:
            coord = float(item.get("y") or 0.0)
        if not np.isfinite(coord):
            continue
        if coord <= coords[0]:
            estimated = values[0] + slope * (coord - coords[0])
        elif coord >= coords[-1]:
            estimated = values[-1] + slope * (coord - coords[-1])
        else:
            estimated = float(np.interp(coord, coords, values))
        estimated = max(0.0, estimated)
        item["value"] = estimated
        item["value_text"] = f"{estimated:.0f}"
        item["value_estimated"] = True
        item["value_type"] = "estimated"
        item["plausibility_message"] = "estimated from axis tick scale"
        item["value_read_verified"] = False
        count += 1
    return count


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
    # Text components (connected strokes) used by both orientation branches:
    # matching components by bar overlap keeps one box per label, unlike the
    # merged text lines that can swallow adjacent category labels.
    n_comp, _, stats, _ = cv2.connectedComponentsWithStats(textmask, 8)
    comps = []
    for i in range(1, n_comp):
        cx, cy, cw, ch, area = [int(value) for value in stats[i]]
        if ch < 6 or ch > 130 or cw < 3 or area < 15:
            continue
        comps.append(
            {
                "x1": cx,
                "y1": cy,
                "x2": cx + cw,
                "y2": cy + ch,
                "ratio": float(area) / float(max(1, cw * ch)),
            }
        )

    def _span_hits(ln: dict[str, int], x: int, w: int) -> bool:
        center = (ln["x1"] + ln["x2"]) / 2
        return x - 15 <= center <= x + w + 15

    def _zone_lines(zone: tuple[int, int, int, int]) -> list[dict[str, int]]:
        zx1, zy1, zx2, zy2 = zone
        out = []
        for ln in lines:
            cx = (ln["x1"] + ln["x2"]) / 2
            cy = (ln["y1"] + ln["y2"]) / 2
            if zx1 - 25 <= cx <= zx2 + 25 and zy1 - 25 <= cy <= zy2 + 25:
                out.append(ln)
        return out

    def _center_dist(ln: dict[str, int], anchor: tuple[int, int]) -> float:
        ax, ay = anchor
        return ((ln["x1"] + ln["x2"]) / 2 - ax) ** 2 + ((ln["y1"] + ln["y2"]) / 2 - ay) ** 2

    for item in aligned:
        eid = str(item.get("entity_id") or "")
        x, y, w, hh = item.get("x"), item.get("y"), item.get("w"), item.get("h")
        if not eid or None in (x, y, w, hh):
            continue
        x, y, w, hh = int(x), int(y), int(w), int(hh)
        orientation = str(item.get("orientation") or ("horizontal" if w > hh else "vertical"))
        boxes = result.setdefault(eid, {})
        baseline = y + hh
        if orientation == "horizontal":
            # Horizontal bars: the value is printed at the right end of the
            # bar and the category label sits above the bar. Text lines keep
            # this branch stable (multi-line labels and long labels like
            # "Less than $20,000" are handled by line selection, not by
            # merging every overlapping component).
            value_candidates = _zone_lines((x + w - 140, y - 12, x + w + 180, y + hh + 12))
            label_candidates = _zone_lines((x - 12, max(0, y - 80), x + w + 12, y - 2))
            if value_candidates:
                solid = [ln for ln in value_candidates if ln["ratio"] >= 0.08]
                pool = solid or value_candidates
                ln = min(pool, key=lambda ln: _center_dist(ln, (x + w, y + hh // 2)))
                box = _tighten_text_line(textmask, ln)
                boxes["value_box"] = [max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])]
            if label_candidates:
                solid = [ln for ln in label_candidates if ln["ratio"] >= 0.08]
                pool = solid or label_candidates
                ln = max(pool, key=lambda ln: ln["y2"])
                box = _tighten_text_line(textmask, ln)
                boxes["label_box"] = [max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])]
        else:
            # Vertical bars: match text components by horizontal overlap with
            # the bar instead of merged text lines. Text lines merge adjacent
            # category labels (e.g. "Germany Switzerland United States" in one
            # box), while component-level matching keeps one box per label.
            # Value candidates must look like solid text (ratio >= 0.10) so
            # arrows, dashed grid lines and decorations never become value
            # boxes.
            def _overlap(a: dict[str, int], bx: int, bw: int) -> int:
                return max(0, min(a["x2"], bx + bw) - max(a["x1"], bx))

            value_zone_end = y + max(40, min(130, int(hh * 0.6)))
            value_comps = [
                comp
                for comp in comps
                if comp["y2"] >= y - 120
                and comp["y1"] <= value_zone_end
                and comp["ratio"] >= 0.10
                and _overlap(comp, x, w) >= max(4, int((comp["x2"] - comp["x1"]) * 0.35))
            ]
            if value_comps:
                box = _tighten_text_line(
                    textmask,
                    {
                        "x1": min(comp["x1"] for comp in value_comps),
                        "y1": min(comp["y1"] for comp in value_comps),
                        "x2": max(comp["x2"] for comp in value_comps),
                        "y2": max(comp["y2"] for comp in value_comps),
                        "ratio": 1.0,
                    },
                )
                boxes["value_box"] = [max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])]
            label_comps = [
                comp
                for comp in comps
                if comp["y1"] >= baseline + 2
                and comp["y1"] <= baseline + 110
                and _overlap(comp, x, w) >= max(4, int((comp["x2"] - comp["x1"]) * 0.35))
            ]
            if label_comps:
                box = _tighten_text_line(
                    textmask,
                    {
                        "x1": min(comp["x1"] for comp in label_comps),
                        "y1": min(comp["y1"] for comp in label_comps),
                        "x2": max(comp["x2"] for comp in label_comps),
                        "y2": max(comp["y2"] for comp in label_comps),
                        "ratio": 1.0,
                    },
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
        orientation = str(a.get("orientation") or ("horizontal" if a["w"] >= a["h"] else "vertical"))
        anim_prop = "width" if orientation == "horizontal" else "height"
        anchor = "left" if orientation == "horizontal" else "bottom"
        lines.append(f'<g id="entity-{eid}" data-role="entity" data-entity-id="{eid}" data-label="{html.escape(str(a.get("label") or ""))}">')
        lines.append(
            f'<rect id="{eid}-bar" data-role="bar" data-entity-id="{eid}" data-value="{html.escape(value)}" '
            f'x="{a["x"]}" y="{a["y"]}" width="{a["w"]}" height="{a["h"]}" '
            f'data-animation-property="{anim_prop}" data-anchor="{anchor}" data-orientation="{orientation}" fill="#3cb44b"/>'
        )
        if value:
            vw = max(52.0, _text_width_estimate(value, 20) + 18)
            if orientation == "horizontal":
                vx = a["x"] + a["w"] + 8
                vy = max(0, a["y"] + a["h"] / 2 - 14)
                text_anchor = "start"
                tx = vx + 9
            else:
                vx = a["x"] + a["w"] / 2 - vw / 2
                vy = max(0, a["y"] - 36)
                text_anchor = "middle"
                tx = a["x"] + a["w"] / 2
            lines.append(
                f'<rect id="{eid}-value-box" data-role="value-box" data-entity-id="{eid}" '
                f'x="{vx:.1f}" y="{vy:.1f}" width="{vw:.1f}" height="28" fill="#ffffff" stroke="#333333" stroke-width="2"{value_box_attr}/>'
            )
            lines.append(
                f'<text data-role="value-label" x="{tx:.1f}" y="{vy + 21}" '
                f'text-anchor="{text_anchor}" font-size="20" font-weight="700">{html.escape(value)}</text>'
            )
        label = str(a.get("label") or "")
        if label:
            lw = max(float(a["w"]), _text_width_estimate(label, 16) + 18)
            if orientation == "horizontal":
                lx = a["x"] - 4
                ly = max(0, a["y"] - 34)
                text_anchor = "start"
                tx = lx + 9
            else:
                baseline = a["y"] + a["h"]
                lx = a["x"] + a["w"] / 2 - lw / 2
                ly = baseline + 8
                text_anchor = "middle"
                tx = a["x"] + a["w"] / 2
            lines.append(
                f'<rect id="{eid}-label-box" data-role="category-box" data-entity-id="{eid}" '
                f'x="{lx:.1f}" y="{ly:.1f}" width="{lw:.1f}" height="28" fill="#ffffff" stroke="#333333" stroke-width="2"{label_box_attr}/>'
            )
            lines.append(
                f'<text data-role="category-label" x="{tx:.1f}" y="{ly + 21}" '
                f'text-anchor="{text_anchor}" font-size="16">{html.escape(label)}</text>'
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
    orientation = boxes[0].get("orientation") if boxes else "vertical"
    try:
        vision_order = read_entity_order(image_path, cfg, orientation)
    except Exception:
        vision_order = None
    aligned, warnings = match_entities(boxes, entities, vision_order)
    values = read_bar_values(image_path, aligned, cfg, orientation)
    for item in values:
        if item.get("value_text"):
            try:
                item["value"] = float(re.sub(r"[^0-9.]", "", item["value_text"]))
            except ValueError:
                item["value"] = None
        else:
            item["value"] = None
    # A zero read on a visible bar is almost always a misread of an unlabeled
    # bar (or an axis tick), not a genuine zero; let the scale estimation
    # fill it in instead.
    for item in values:
        if item.get("value") == 0 and _bar_length(item) > 5 and str(item.get("value_text") or "") not in ("0%", "0 %"):
            item["value"] = None
            item["value_text"] = None
    # Values that survived the majority vote are treated as directly printed
    # (trusted); the scale estimation fills in whatever is still unlabeled.
    for item in values:
        if item.get("value_text"):
            item["value_read_verified"] = True
    estimated_count = estimate_unlabeled_values(values)
    tick_marks: list[dict[str, Any]] = []
    tick_unit = ""
    tick_estimated_count = 0
    verified_count = sum(1 for item in values if item.get("value_read_verified"))
    if verified_count < 2:
        # Not enough printed values for the labeled-bar scale: try calibrating
        # from the value-axis tick marks instead (e.g. unlabeled bar charts
        # that still draw a "$0/$100/$200" axis).
        try:
            tick_marks = detect_axis_tick_marks(image_path, orientation)
        except Exception:
            tick_marks = []
        if tick_marks:
            try:
                tick_labels, tick_unit = read_tick_labels(image_path, cfg, orientation)
            except Exception:
                tick_labels = []
            if tick_labels:
                paired = _pair_ticks_with_labels(tick_marks, tick_labels, orientation)
                if not paired and len(tick_labels) == len(tick_marks) + 1 and values:
                    # The 0/start anchor is usually hidden under the bars;
                    # recover it from the shared bar baseline.
                    baseline = _infer_baseline_coord(values, orientation)
                    if baseline is not None:
                        paired = _pair_ticks_with_labels(
                            [*tick_marks, {"coord": baseline}],
                            tick_labels,
                            orientation,
                        )
                tick_estimated_count = estimate_unlabeled_values_from_ticks(values, paired)
    estimated_count += tick_estimated_count
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
        "estimated_value_count": estimated_count,
        "tick_estimated_value_count": tick_estimated_count,
        "tick_mark_count": len(tick_marks),
        "tick_unit": tick_unit,
        "orientation": (
            (values[0].get("orientation") if values else None)
            or (boxes[0].get("orientation") if boxes else None)
        ),
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
