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


def _parse_marks(raw: Any) -> list[list[float]]:
    """[[x_value, y], ...] with numeric values only."""
    out: list[list[float]] = []
    for m in raw or []:
        if not isinstance(m, (list, tuple)) or len(m) < 2:
            continue
        try:
            out.append([float(m[0]), float(m[1])])
        except (TypeError, ValueError):
            continue
    return out


def _parse_point_labels(raw: Any) -> list[dict[str, Any]]:
    """[{"x": ..., "label": ...}, ...]"""
    out: list[dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        try:
            xv = float(item.get("x"))
        except (TypeError, ValueError):
            continue
        label = str(item.get("label") or "").strip()
        if label:
            out.append({"x": xv, "label": label})
    return out


def _parse_segment_labels(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        try:
            xs = float(item.get("x_start")); xe = float(item.get("x_end"))
        except (TypeError, ValueError):
            continue
        label = str(item.get("label") or "").strip()
        if label:
            out.append({"x_start": xs, "x_end": xe, "label": label})
    return out


def _parse_reference_lines(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        try:
            value = float(item.get("value"))
        except (TypeError, ValueError):
            continue
        out.append({"value": value, "label": str(item.get("label") or "")})
    return out


def _parse_line_plot(obj: dict[str, Any]) -> dict[str, Any]:
    """Parse one line-chart plot (top-level or one entry of ``plots``)."""
    out: dict[str, Any] = {"title": "", "unit": "", "x_labels": [], "ticks": [], "series": []}
    out["title"] = str(obj.get("title") or "").strip()
    out["x_axis_name"] = str(obj.get("x_axis_name") or "").strip()
    out["y_axis_name"] = str(obj.get("y_axis_name") or "").strip()
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
    out["relative"] = bool(obj.get("relative")) or (len(ticks) == 0)
    out["x_indexed"] = bool(obj.get("x_indexed")) or (len(out["x_labels"]) == 0)
    out["show_values"] = bool(obj.get("show_values"))
    out["segment_labels"] = _parse_segment_labels(obj.get("segment_labels"))
    out["reference_lines"] = _parse_reference_lines(obj.get("reference_lines"))
    raw_style = obj.get("style") if isinstance(obj.get("style"), dict) else {}
    out["style"] = {
        "background": str(raw_style.get("background") or "#ffffff"),
        "gridlines": bool(raw_style.get("gridlines", False)),
        "axis_color": str(raw_style.get("axis_color") or "#666666"),
        "tick_color": str(raw_style.get("tick_color") or "#444444"),
        "label_color": str(raw_style.get("label_color") or "#444444"),
        "title_color": str(raw_style.get("title_color") or "#222222"),
        "legend": str(raw_style.get("legend") or "none"),
    }
    series: list[dict[str, Any]] = []
    for item in obj.get("series") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        color = str(item.get("color") or "").strip()
        curve_type = str(item.get("curve_type") or "").strip().lower()
        if curve_type not in {"smooth", "polyline"}:
            curve_type = "polyline"
        line_style = str(item.get("line_style") or "").strip().lower()
        if line_style not in {"solid", "dashed"}:
            line_style = "solid"
        try:
            line_width = float(item.get("line_width") or 5)
        except (TypeError, ValueError):
            line_width = 5.0
        points: list[tuple[float, float | None, float | None, float]] = []
        for point in item.get("points") or []:
            try:
                x_px = float(point[0])
            except (TypeError, ValueError, IndexError):
                continue
            quad = len(point) >= 4
            x_value = None
            if quad and point[1] is not None:
                try:
                    x_value = float(point[1])
                except (TypeError, ValueError):
                    x_value = None
            y_value = None
            y_idx = 2 if quad else 1
            if len(point) > y_idx and point[y_idx] is not None:
                try:
                    y_value = float(point[y_idx])
                except (TypeError, ValueError):
                    y_value = None
            y_rel = None
            r_idx = 3 if quad else 2
            if len(point) > r_idx and point[r_idx] is not None:
                try:
                    y_rel = float(point[r_idx])
                except (TypeError, ValueError):
                    y_rel = None
            if y_rel is None:
                y_rel = y_value
            if y_rel is None:
                continue
            points.append((x_px, x_value, y_value, y_rel))
        if points:
            series.append(
                {
                    "name": name,
                    "color": color,
                    "curve_type": curve_type,
                    "line_style": line_style,
                    "line_width": line_width,
                    "points": points,
                    "marks": _parse_marks(item.get("marks")),
                    "point_labels": _parse_point_labels(item.get("point_labels")),
                }
            )
    out["series"] = series
    if out.get("relative"):
        for _s in out["series"]:
            _s["points"] = [
                [p[0], p[1], None, p[3]] if len(p) > 3 else [p[0], p[1], None, p[2]]
                for p in (_s.get("points") or [])
            ]
    # Collapse consecutive points sharing the same x_value (keep the last).
    for _item in out["series"]:
        _pts = _item.get("points") or []
        if len(_pts) < 2:
            continue
        _collapsed: list = []
        for _p in _pts:
            if (
                _collapsed
                and _p[1] is not None
                and _collapsed[-1][1] is not None
                and abs(_p[1] - _collapsed[-1][1]) < 1e-9
            ):
                _collapsed[-1] = _p
            else:
                _collapsed.append(_p)
        _item["points"] = _collapsed
    return out


def read_line_analysis(
    image_path: str | Path,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Single vision pass over a line-chart keyframe (bar-style overall detect).

    Returns ``{"title", "unit", "x_labels", "ticks", "relative", "x_indexed",
    "series"}`` where each series carries ``name``, ``color`` (hex),
    ``curve_type`` ("smooth" | "polyline"), ``line_style`` and ``points`` =
    ``[[x, y], ...]``.  Smooth curves are sampled evenly (40 pts); polylines
    keep turning points and are filled to 20-40 pts.  Without any y ticks /
    printed values ``relative`` is true and y is a 0..1 relative height.
    """
    _emphasis = (cfg or {}).get("line_read_emphasis") if isinstance(cfg, dict) else None
    prompt = (
        "这是一张折线图的关键帧（可能有多条线，也可能没有 y 轴刻度）。"
        "如果这张图是多个相互独立、标题/刻度各不相同的折线图拼在一起（如并排分栏对比），"
        "先数清子图数量（最多 4 个），返回 JSON 顶层带 \"plots\": [子图对象数组]，"
        "每个子图对象含 title/unit/x_labels/ticks/relative/x_indexed/show_values/style/series（结构同单图顶层）；"
        "只有单个折线图时不要用 plots，直接返回顶层字段。\n"
        "请一次性读出整张图的结构，只返回一个 JSON 对象，不要解释：\n"
        '{"title": 图中标题原文，没有则空字符串, '
        '"x_axis_name": 横轴标题原文（如 "Year"/"年份"，没有则空字符串）, '
        '"y_axis_name": 纵轴标题原文（如 "Price ($)"，没有则空字符串）, '
        '"unit": y 轴刻度使用的单位符号（如 $、%、k、M、million；图中没有任何单位则空字符串，禁止臆测）, '
        '"x_labels": [横轴从左到右的刻度标签数组, 如 ["2000","2005","2010"]], '
        '"ticks": [{"value": y 轴刻度数值, "label": 刻度原文}, ...]（按刻度在轴上的顺序排列；没有刻度则空数组）, '
        '"relative": true/false（仅当图中没有任何 y 轴刻度或印刷数值时为 true）, '
        '"x_indexed": true/false（仅当横轴没有可见标签时为 true）, '
        '"show_values": true/false（原图的数据点上是否印刷了数值，没有则 false）, '
        '"reference_lines": [{"value": 参考线位置的相对高度 0~1 或数值, "label": 线旁文字（没有则空字符串）}]（没有则空数组）, '
        '"style": {"background": 背景色 hex 如 "#ffffff", "gridlines": true/false 是否有网格线, '
        '"axis_color": 坐标轴颜色 hex, "tick_color": 刻度线/刻度文字颜色 hex, '
        '"label_color": 横轴标签颜色 hex, "title_color": 标题颜色 hex（无标题可省略）, '
        '"legend": "none"（无图例）或 "top"（顶部）或 "right"（右侧），图中没有图例时必须用 "none" }, '
        '"series": [{"name": 图例名（图例/画面上的系列标识原文；若图例是旗帜/图形等非文字，用 emoji 或简短描述；图中没有图例则空字符串，严禁编造 "series"/"series 1"）, '
        '"color": 该线主色 hex 如 "#E45756", '
        '"curve_type": "smooth"（光滑曲线）或 "polyline"（有明显折角的折线）, '
        '"line_style": "solid" 或 "dashed", '
        '"line_width": 线条粗细像素（如 3 或 5，按原图）, '
        '"points": [[x_px, x_value, y_value, y_relative], ...], '
        '"marks": [[x_value, y_value_or_relative], ...]（图中该线上画了圆点/标记的点，没有则空数组）, '
        '"point_labels": [{"x": x_value, "label": 点旁标注文字原文如 "47%"}]（没有则空数组）}, ...]}\n'
        "采样规则：\n"
        "- 每条线的点按实际形状给出。smooth 曲线：从左到右等距采 40 个点，"
        "x_px 必须从左到右等距递增；polyline 折线：必须给出所有可见的方向转折点（每个峰/谷/折角都算，不能为了简化而漏掉），"
        "如果转折很多（可能 60~200 个），允许总点数超过 40，最多 200 个；如果转折确实很少（不足 20），再在相邻转折点之间等距补点到 20~40。\n"
        "- 每个点都是四元组 [x_px, x_value, y_value, y_relative]："
        "x_px 是该点在整张图里的像素 x 坐标（0~图片宽度，供数据表 ptx 列）；"
        "x_value 是该点在 x 轴上的数值（有横轴刻度时用刻度值/刻度之间的位置，如 1990；x_indexed 时用序号 0,1,2,...）；"
        "y_value 是绝对数值（relative=false 时按该点在 y 轴刻度处的位置读数，不要计算；relative=true 时填 null）；"
        "y_relative 是该点的相对高度 0~1（0=绘图区底部，1=顶部），任何情况下都必须给出。\n"
        "- x_value 必须落在横轴刻度范围（x_labels 的最小~最大值）内，严禁把曲线/折线延伸到刻度范围之外；每条线的点必须按 x 从左到右顺序排列，不允许回折。\n"
"- 多线图：自行判断如何区分每条线（颜色、图例、位置等），"
        "每条线的点只能属于该线；交点处两条线分别给各自的点，不要混点。"
    )
    if _emphasis:
        prompt = "特别注意：" + str(_emphasis) + "\n" + prompt
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
    out = _parse_line_plot(obj)
    raw_plots = obj.get("plots")
    if isinstance(raw_plots, list):
        plots = [_parse_line_plot(p) for p in raw_plots if isinstance(p, dict)]
        plots = [p for p in plots if p.get("series")]
        if plots:
            out["plots"] = plots
            out.pop("series", None)
    return out


def build_line_data_table(
    spec: dict[str, Any],
    clip_id: str,
    image_width: int | float | None = None,
) -> list[dict[str, Any]]:
    """Rows for a line clip's data_table.csv.

    Smooth curves carry no real data -> empty list (the caller writes a
    header-only placeholder).  Polylines keep turning points with the x pixel
    position, the absolute value (when ticks exist) and the 0..1 relative
    height.
    """
    series = spec.get("series") or []
    if not series:
        return []
    if all(
        str(s.get("curve_type") or "polyline") == "smooth"
        for s in series
        if isinstance(s, dict)
    ):
        return []
    unit = str(spec.get("unit") or "")
    relative = bool(spec.get("relative", False))
    rows: list[dict[str, Any]] = []
    for s in series:
        if not isinstance(s, dict):
            continue
        name = str(s.get("name") or "")
        eid = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        for point in s.get("points") or []:
            try:
                x_px = float(point[0])
            except (TypeError, ValueError, IndexError):
                continue
            y_value = None
            y_idx = 2 if len(point) >= 4 else 1
            if len(point) > y_idx and point[y_idx] is not None:
                try:
                    y_value = float(point[y_idx])
                except (TypeError, ValueError):
                    y_value = None
            y_rel = None
            r_idx = 3 if len(point) >= 4 else 2
            if len(point) > r_idx and point[r_idx] is not None:
                try:
                    y_rel = float(point[r_idx])
                except (TypeError, ValueError):
                    y_rel = None
            if y_rel is None:
                y_rel = y_value
            if y_rel is None:
                continue
            if image_width and float(image_width) > 0 and x_px > 1.0:
                frac = max(0.0, min(1.0, x_px / float(image_width)))
            else:
                frac = max(0.0, min(1.0, x_px))
            rows.append(
                {
                    "clip_id": clip_id,
                    "entity": name,
                    "entity_id": eid,
                    "x_px": int(round(frac * float(image_width))) if image_width else None,
                    "value": "" if (relative or y_value is None) else round(y_value, 4),
                    "relative_value": round(y_rel, 4),
                    "unit": unit,
                    "source_type": "vision_line",
                }
            )
    return rows


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
    if analysis.get("relative"):
        # Vision says there is no y scale; trust it over any CV tick marks.
        relative = True
        scale = None
        tick_unit = ""
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
        line["curve_type"] = match.get("curve_type") or "polyline"
        line["line_style"] = match.get("line_style") or "solid"
        vpts = match.get("points") or []
        if relative and len(vpts) >= 2:
            # No axis: vision already returned 0..1 relative heights.
            line["points"] = [
                {"x_label": str(lab), "value": float(val), "x_index": idx}
                for idx, (lab, val) in enumerate(vpts)
            ]
            line["value_source"] = "vision_relative"
            line["value_review"] = True
            for pt in line["points"]:
                if pt.get("value") is not None:
                    point_count += 1
            continue
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
