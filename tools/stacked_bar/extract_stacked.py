"""Stacked-bar extraction v7: annotation removal + hybrid (CV heights +
vision composition).

Preprocessing removes red hand-drawn annotations (circle/arrow) via a
saturation-based mask + inpainting -- a general step, not video-specific.
Then CV measures actual bar heights inside the vision boxes, and vision
provides the per-bar color composition (fractions).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from datavideo.cv_align import _call_vision  # noqa: E402
from stacked_bar_template import build_intent, build_stacked_svg  # noqa: E402


import os as _os

CFG = {
    "cv_align": {
        "node_path": _os.environ.get("DATAVIDEO_VISION_NODE", "node"),
        "script": _os.environ.get("DATAVIDEO_VISION_SCRIPT", "vision.js"),
    }
}

STACKED_PROMPT = (
    "这是图表关键帧。请像逐行读表一样逐根柱独立测量，只返回一个 JSON 对象（不要解释）："
    '{"stacked": true/false, '
    '"bar_count": 柱子的总数量（整数，务必准确）, '
    '"bar_labels": 横轴标签数组（按从左到右顺序；没有印刷标签就 null）, '
    '"legend": [{"name": "类别名", "color": "#rrggbb"}, ...]（图例/色段类别，按图例从上到下顺序）, '
    '"bars": [{"x": 左缘比例0~1, "y": 顶缘比例0~1, "w": 宽比例, "h": 高比例, '
    '"segments": [{"name": "色段类别名（必须与 legend 一致）", "fraction": 该色段占该柱的比例（0~1）}, ...]'
    '（按从下到上顺序，所有 fraction 加起来必须等于 1）}, ...]}。'
    "测量要求：1) 逐根柱独立测量，不要假设柱高存在趋势或平滑变化，照实读图；"
    "2) 每根柱的每个颜色段都要独立给出占比（薄段也要列出，如 0.02）；"
    "3) legend 顺序=图例从上到下；4) bar_count 必须准确；看不清标签填 null 不要编造。"
)


def remove_red_annotations(img: np.ndarray) -> np.ndarray:
    """Inpaint saturated red pixels (hand-drawn circles/arrows/checks)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((h <= 12) | (h >= 168)) & (s > 90) & (v > 70)
    mask = mask.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    if int(mask.sum()) == 0:
        return img
    return cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)


def _cv_bar_heights(img: np.ndarray, boxes: list[tuple[int, int, int, int]], legend: list[dict]) -> list[float]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    nb = (hsv[..., 1].astype(np.int32) > 45) | (hsv[..., 2].astype(np.int32) < 190)
    heights = []
    for x0, y0, x1, y1 in boxes:
        y0, y1 = max(0, y0), min(img.shape[0] - 1, y1)
        x0, x1 = max(0, x0), min(img.shape[1] - 1, x1)
        if x1 <= x0 or y1 <= y0:
            heights.append(0.0)
            continue
        rows = np.where(nb[y0 : y1 + 1, x0 : x1 + 1].any(axis=1))[0]
        heights.append(float(rows.max() - rows.min()) if len(rows) else 0.0)
    return heights


def main(image_path: str, out_dir: str) -> None:
    raw = cv2.imread(str(image_path))
    if raw is None:
        print("CANNOT READ", image_path)
        return
    img = remove_red_annotations(raw)
    try:
        text = _call_vision(str(image_path), STACKED_PROMPT, CFG, temperature=0.0)
        analysis = json.loads(text[text.find("{") : text.rfind("}") + 1])
    except Exception as exc:
        print("VISION FAIL", exc)
        return
    legend = analysis.get("legend") or []
    bars = analysis.get("bars") or []
    bar_count = int(analysis.get("bar_count") or 0)
    print("vision:", {"stacked": analysis.get("stacked"), "bar_count": bar_count, "bars": len(bars)})
    if not legend or not bars:
        print("NO LEGEND/BARS")
        return

    img_h, img_w = img.shape[:2]
    boxes = []
    for b in bars[:bar_count]:
        try:
            x, y, w, h = (float(b.get(k)) for k in ("x", "y", "w", "h"))
        except (TypeError, ValueError):
            continue
        if max(abs(x), abs(y), abs(w), abs(h)) > 1.5:
            x, w = x / img_w, w / img_w
            y, h = y / img_h, h / img_h
        boxes.append((int(x * img_w), int(y * img_h), int((x + w) * img_w), int((y + h) * img_h)))
    heights = _cv_bar_heights(img, boxes, legend) if len(boxes) >= len(bars) else [0.0] * len(bars)
    print("cv heights samples:", [int(h) for h in heights[::5]])
    max_h = max(heights) or 1.0

    series_order = {s["name"]: i for i, s in enumerate(legend)}
    rows, orders, values = [], {}, {}
    bar_labels = analysis.get("bar_labels") or []
    real_labels = [x for x in bar_labels if x not in (None, "")]
    if len(real_labels) >= 2 and len(bars) > 2:
        try:
            first, last = int(real_labels[0]), int(real_labels[-1])
            if last > first and last - first + 1 >= len(bars) - 1:
                bar_labels = [str(first + i) for i in range(len(bars))]
        except (TypeError, ValueError):
            pass
    for i, b in enumerate(bars):
        label = str(b.get("label") or (bar_labels or [None] * len(bars))[i] or f"bar-{i + 1}")
        if label in (None, "", "None"):
            label = f"bar-{i + 1}"
        bar_rel = (heights[i] / max_h) if i < len(heights) else 0.0
        segs = b.get("segments") or []
        total = sum(float(s.get("fraction") or 0.0) for s in segs) or 1.0
        orders[label] = {}
        values.setdefault(label, {})
        for pos, s in enumerate(segs):
            name = str(s.get("name") or "")
            frac = (float(s.get("fraction") or 0.0)) / total
            v = round(bar_rel * frac, 4)
            rows.append((label, name, v, pos))
            orders[label][name] = pos
            values[label][name] = v

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "data_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "bar", "series", "value", "unit", "series_order", "stack_pos", "value_type"])
        for label, name, v, pos in rows:
            writer.writerow(["", label, name, f"{v:g}", "", series_order.get(name, 0), pos, "relative"])
    title = str(analysis.get("title") or "")
    svg = build_stacked_svg(title, list(values.keys()), legend, values, "", orders=orders)
    (out / "semantic.svg").write_text(svg, encoding="utf-8")
    intent = build_intent(title or "Stacked Bar Chart", list(values.keys()), legend, values)
    (out / "intent.json").write_text(json.dumps(intent, ensure_ascii=False, indent=2), encoding="utf-8")
    print("rows:", len(rows), "DONE", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
