"""Line-chart CV/vision alignment (bar-style overall detect).

One vision pass over the keyframe is the authoritative read: title, unit,
x-axis labels, y ticks and per-series data points (start/end/turning points,
10-15 evenly spaced).  CV tracing (``cv_align.detect_lines``) provides dense
geometry; vision points calibrate those dense CV points so the final table
has many accurate points instead of a sparse turning-point read.  When no
absolute tick scale exists the series fall back to relative values (highest
point = 1.0), mirroring the no-value bar path so a semantic SVG always
renders.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from datavideo.cv_align import (  # noqa: E402
    _assign_x_labels,
    _call_vision,
    _detect_tick_label_blocks,
    _detect_x_axis_tick_positions,
    _estimate_coord_value,
    _extract_json_object,
    _pair_ticks_with_labels,
    _rebuild_even_ticks,
    _tick_scale,
    detect_axis_tick_marks,
    detect_lines,
    ensure_dir,
    read_series_labels,
    read_tick_labels,
    read_x_axis_labels,
    write_json,
)


def read_line_analysis(
    image_path: str | Path,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Single vision pass over a line-chart keyframe (bar-style overall detect).

    Returns ``{"title", "unit", "x_labels", "ticks", "series"}`` where each
    series carries ``name``, ``color`` (hex) and ``points`` =
    ``[[x_label, value], ...]`` including start, end and turning points.
    """
    prompt = (
        "这是一张折线图的关键帧（可能有多条线，也可能没有 y 轴刻度）。"
        "请一次性读出整张图的结构，只返回一个 JSON 对象，不要解释：\n"
        '{"title": 图中标题原文，没有则空字符串, '
        '"unit": y 轴刻度使用的单位符号（如 $、%、k、M、million；图中没有任何单位则空字符串，禁止臆测）, '
        '"x_labels": [横轴从左到右的刻度标签数组, 如 ["2000","2005","2010"]], '
        '"ticks": [{"value": y 轴刻度数值, "label": 刻度原文}, ...]（按刻度在轴上的顺序排列；没有刻度则空数组）, '
        '"series": [{"name": 系列名（图例/标题原文，没有则空字符串）, '
        '"color": 该折线的主色 hex 如 "#E45756", '
        '"points": [[横轴标签, 数值], ...]}, ...]}\n'
        "points 要求：每条线读 10~15 个点，从横轴起点到最右端均匀覆盖，"
        "必须包含起点、终点和所有明显峰/谷转折点，后半段不能缺失；"
        "数值按折线在 y 轴刻度处的位置读数，不要计算。"
        "x_labels 和 points 里的横轴标签要能对应。"
    )
    # Vision is stochastic: accept the first dense response, otherwise retry
    # once and keep the denser of the two.  "Dense" means at least 5 x-axis
    # labels and 5 points per series (a sparse turning-point-only reading is
    # the "only three points" problem we are trying to fix).
    attempts: list[dict[str, Any]] = []
    for _ in range(2):
        try:
            text = _call_vision(image_path, prompt, cfg, temperature=0.0)
        except Exception:
            continue
        candidate = _extract_json_object(text)
        if not isinstance(candidate, dict):
            continue
        attempts.append(candidate)
        n_x = len([str(x) for x in (candidate.get("x_labels") or []) if str(x).strip()])
        total_points = sum(
            len(s.get("points") or [])
            for s in (candidate.get("series") or [])
            if isinstance(s, dict)
        )
        if n_x >= 5 and total_points >= 5:
            break
    if not attempts:
        return {}
    obj = max(
        attempts,
        key=lambda c: (
            sum(
                len(s.get("points") or [])
                for s in (c.get("series") or [])
                if isinstance(s, dict)
            ),
            len([str(x) for x in (c.get("x_labels") or []) if str(x).strip()]),
        ),
    )
    out: dict[str, Any] = {"title": "", "unit": "", "x_labels": [], "ticks": [], "series": []}
    out["title"] = str(obj.get("title") or "").strip()
    out["unit"] = str(obj.get("unit") or "").strip()
    out["x_labels"] = [str(x).strip() for x in (obj.get("x_labels") or []) if str(x).strip()]
    ticks: list[dict[str, Any]] = []
    for item in obj.get("ticks") or []:
        if not isinstance(item, dict):
            continue
        try:
            value = float(item.get("value"))
        except (TypeError, ValueError):
            continue
        ticks.append({"value": value, "label": str(item.get("label") or item.get("value") or value)})
    out["ticks"] = ticks
    series: list[dict[str, Any]] = []
    for item in obj.get("series") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        color = str(item.get("color") or "").strip()
        points: list[tuple[str, float]] = []
        for point in item.get("points") or []:
            try:
                label = str(point[0]).strip()
                value = float(point[1])
            except (TypeError, ValueError, IndexError):
                continue
            if label:
                points.append((label, value))
        if points:
            series.append({"name": name, "color": color, "points": points})
    out["series"] = series
    return out


def _x_numeric(label: Any) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", str(label or ""))
    return float(match.group(0)) if match else None


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int] | None:
    h = hex_color.strip().lstrip("#")
    if len(h) != 6:
        return None
    try:
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return None


def _calibrate_cv_points(
    cv_points: list[dict[str, Any]],
    vision_points: list[tuple[str, float]],
) -> list[dict[str, Any]] | None:
    """Map dense CV-traced points onto the vision (x_label, value) pairs.

    Linear interpolation in the shared numeric x space; CV points beyond the
    last vision point are dropped so the polyline never extends past the data
    (the "extra segment" / over-extension problem).
    """
    pairs: list[tuple[float, float]] = []
    for label, value in vision_points:
        xv = _x_numeric(label)
        if xv is not None:
            pairs.append((xv, float(value)))
    pairs.sort()
    if len(pairs) < 2:
        return None
    xs = [p[0] for p in pairs]
    vs = [p[1] for p in pairs]
    out: list[dict[str, Any]] = []
    for pt in cv_points:
        xl = pt.get("x_label")
        xv = _x_numeric(xl) if xl else None
        if xv is None:
            continue
        if xv < xs[0]:
            value = vs[0]
        elif xv > xs[-1]:
            continue
        else:
            value = float(np.interp(xv, xs, vs))
        out.append({**pt, "x_label": xl, "value": round(max(0.0, value), 4)})
    return out


def run_cv_align_line(
    clip_id: str,
    image_path: str | Path,
    out_dir: str | Path,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Line-chart alignment: vision overall detect + CV dense calibration.

    Value-read order: vision ticks (values) aligned onto CV tick coordinates
    -> CV tick scale -> relative normalisation (highest point = 1.0).
    Series values: vision-calibrated CV dense points -> vision points ->
    CV estimated values -> relative.
    """
    out_dir = ensure_dir(out_dir)
    lines = detect_lines(image_path)
    frame = cv2.imread(str(image_path))
    height = frame.shape[0] if frame is not None else 720

    analysis = read_line_analysis(image_path, cfg)
    a_title = str(analysis.get("title") or "")
    a_unit = str(analysis.get("unit") or "")
    a_x_labels = [str(x) for x in analysis.get("x_labels") or []]
    a_ticks = analysis.get("ticks") or []
    a_series = analysis.get("series") or []

    # X axis: the vision labels are authoritative; only fall back to the
    # separate x-axis read when the single pass returned none.
    x_labels = list(a_x_labels)
    if not x_labels:
        try:
            x_labels = read_x_axis_labels(image_path, cfg)
        except Exception:
            x_labels = []
    x_ticks: list[float] = []
    try:
        if frame is not None:
            x_ticks = _detect_x_axis_tick_positions(frame)
    except Exception:
        pass
    if x_labels:
        all_xs = [float(px) for line in lines for px, _ in line.get("points", [])]
        if len(x_ticks) >= 2 and len(x_ticks) == len(x_labels):
            pass
        elif all_xs:
            # Rebuild evenly across the traced data extent so the year
            # interpolation never extrapolates past the data (the 2240-41
            # over-extension problem).
            lo, hi = min(all_xs), max(all_xs)
            n = max(1, len(x_labels) - 1)
            x_ticks = [lo + (hi - lo) * i / n for i in range(len(x_labels))]

    tick_marks: list[dict[str, Any]] = []
    cv_tick_unit = ""
    if not a_ticks:
        # Only when the single pass reported no y ticks, fall back to the CV
        # tick strokes + separate label read.
        try:
            tick_marks = detect_axis_tick_marks(image_path, "vertical")
        except Exception:
            tick_marks = []
        if tick_marks:
            try:
                tick_labels, cv_tick_unit = read_tick_labels(image_path, cfg, "vertical")
            except Exception:
                tick_labels = []
            paired = _pair_ticks_with_labels(tick_marks, tick_labels, "vertical") if tick_labels else []
            if not paired and tick_labels and len(tick_labels) != len(tick_marks):
                frame2 = cv2.imread(str(image_path))
                label_blocks = _detect_tick_label_blocks(frame2, "vertical") if frame2 is not None else []
                cv_coords = [float(item["coord"]) for item in tick_marks]
                base = (
                    cv_coords
                    if (cv_coords and (max(cv_coords) - min(cv_coords)) >= 0.2 * height)
                    else (label_blocks if len(label_blocks) >= 2 else cv_coords)
                )
                rebuilt = _rebuild_even_ticks(base, len(tick_labels))
                paired = _pair_ticks_with_labels([{"coord": c} for c in rebuilt], tick_labels, "vertical")
            tick_marks = paired

    scale = None
    value_read_method = "none"
    tick_unit = a_unit or cv_tick_unit
    if a_ticks and len(a_ticks) >= 2:
        cv_coords = sorted(float(m["coord"]) for m in tick_marks if m.get("coord") is not None)
        coords_synthesized = len(cv_coords) < 2
        vv = sorted((float(t["value"]), str(t.get("label") or t["value"])) for t in a_ticks if t.get("value") is not None)
        if len(vv) >= 2 and coords_synthesized:
            # Vision reported the tick values but CV found no tick strokes;
            # distribute the tick positions evenly across the traced data's
            # y extent (higher y pixel = lower value) and flag for review.
            ys = [float(py) for line in lines for _, py in line.get("points", [])]
            if len(ys) >= 2:
                ybot, ytop = max(ys), min(ys)
                if ybot != ytop:
                    n = max(1, len(vv) - 1)
                    cv_coords = [ybot + (ytop - ybot) * i / n for i in range(len(vv))]
        if len(cv_coords) >= 2 and len(vv) >= 2:
            if len(cv_coords) != len(vv):
                cv_coords = _rebuild_even_ticks(cv_coords, len(vv))
            if len(cv_coords) == len(vv):
                paired_vision = [{"coord": c, "value": v, "label": lab} for c, (v, lab) in zip(cv_coords, vv)]
                scale = _tick_scale(paired_vision)
                if scale is not None:
                    tick_marks = paired_vision
                    value_read_method = "vision_ticks_even" if coords_synthesized else "vision_ticks"
                    for line in lines:
                        if coords_synthesized:
                            line.setdefault("value_review", True)
    if scale is None and tick_marks:
        scale = _tick_scale(tick_marks)
        if scale is not None:
            value_read_method = "cv_tick_scale"
    relative = scale is None
    if relative:
        value_read_method = "relative"

    series_labels: list[str] = [str(s.get("name") or "").strip() for s in a_series if s.get("name")]
    if not series_labels:
        try:
            series_labels = read_series_labels(image_path, cfg)
        except Exception:
            series_labels = []

    vision_used: set[int] = set()

    def _vision_match(line: dict[str, Any]) -> dict[str, Any] | None:
        label = str(line.get("label") or "")
        for idx, series in enumerate(a_series):
            if idx in vision_used:
                continue
            if series.get("name") and label and series["name"] == label:
                vision_used.add(idx)
                return series
        line_rgb = tuple(reversed([int(v) for v in line.get("color", [0, 0, 0])]))
        best_idx, best_d = None, 1e18
        for idx, series in enumerate(a_series):
            if idx in vision_used:
                continue
            rgb = _hex_to_rgb(str(series.get("color") or ""))
            if rgb is None:
                continue
            distance = sum((a - b) ** 2 for a, b in zip(line_rgb, rgb))
            if distance < best_d:
                best_idx, best_d = idx, distance
        if best_idx is not None and best_d < 12000:
            vision_used.add(best_idx)
            return a_series[best_idx]
        # Order-based fallback: a single-series chart often has an empty
        # series name/color in the vision read; assign the first unmatched
        # vision series to this line.
        for idx, series in enumerate(a_series):
            if idx in vision_used:
                continue
            vision_used.add(idx)
            return series
        return None

    def _apply_relative(line: dict[str, Any]) -> None:
        ys = [float(pt["y"]) for pt in line.get("points", []) if pt.get("y") is not None]
        if not ys:
            return
        ymax, ymin = max(ys), min(ys)
        for pt in line.get("points", []):
            if pt.get("y") is None:
                continue
            pt["value"] = round((ymax - float(pt["y"])) / (ymax - ymin), 4) if ymax != ymin else 1.0
        line["value_source"] = "relative"
        line["value_review"] = True

    for line in lines:
        bbox = [float(v) for v in line.get("bbox", [0, 0, 1, 1])]
        left, right = bbox[0], bbox[2]
        enriched = []
        for px, py in line.get("points", []):
            value = _estimate_coord_value(float(py), scale) if scale is not None else None
            enriched.append({"x": int(px), "y": int(py), "value": value})
        _assign_x_labels(enriched, x_ticks, x_labels, left, right)
        line["points"] = enriched

    for index, line in enumerate(lines):
        if series_labels:
            line["label"] = series_labels[index] if index < len(series_labels) else f"series {index + 1}"

    point_count = 0
    for line in lines:
        match = _vision_match(line)
        if match is None:
            if relative:
                _apply_relative(line)
            else:
                line["value_source"] = "cv_tick_scale"
            for pt in line.get("points", []):
                if pt.get("value") is not None:
                    point_count += 1
            continue
        line["label"] = match.get("name") or line.get("label") or "series"
        vpts = match.get("points") or []
        calibrated = _calibrate_cv_points(line.get("points", []), vpts)
        if calibrated and len(calibrated) >= 2:
            line["points"] = calibrated
            line["value_source"] = "vision_calibrated"
        elif len(vpts) >= 2:
            line["points"] = [{"x_label": str(lab), "value": float(val)} for lab, val in vpts]
            line["value_source"] = "vision_tick_read"
        elif relative:
            _apply_relative(line)
        else:
            line["value_source"] = "cv_tick_scale"
        for pt in line.get("points", []):
            if pt.get("value") is not None:
                point_count += 1

    report = {
        "clip_id": clip_id,
        "line_count": len(lines),
        "point_count": point_count,
        "tick_mark_count": len(tick_marks),
        "tick_unit": tick_unit,
        "x_axis_tick_count": len(x_ticks),
        "x_axis_labels": x_labels,
        "title": a_title,
        "vision_analysis": analysis,
        "value_read_method": value_read_method,
        "relative_mode": relative,
        "lines": lines,
        "success": bool(lines),
    }
    write_json(out_dir / "aligned_lines_report.json", report)
    return report
