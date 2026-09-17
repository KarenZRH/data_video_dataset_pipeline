"""Data-driven semantic SVG renderer.

Renders chart semantic.svg / semantic_components.json from recovered data
instead of VLM-predicted bounding boxes.  Bar geometry is computed
deterministically from the values by default; when CV-detected bar boxes
(``geometry``) and a vision style spec (``style``) are provided, the renderer
places bars at their real pixel positions and mimics the original chart's
visual style while keeping values from the data table.
"""

from __future__ import annotations

import html
import re
import shutil
from pathlib import Path
from typing import Any

from .cv_align import _call_vision, _extract_json_object, _normalize_label
from .schemas import ensure_dir, write_json


W, H = 1280, 720
LEFT, RIGHT, TOP, BOTTOM = 120, 1220, 140, 600


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return slug or "item"


def _entity_id(entity: dict[str, Any]) -> str:
    return _slug(str(entity.get("entity_id") or entity["label"]))


def _mark_id(entity: dict[str, Any]) -> str:
    metric = str(entity.get("metric") or "").strip()
    base = str(entity.get("entity_id") or entity["label"])
    return _slug(f"{base}-{metric}" if metric else base)


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
    try:
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().replace(",", "").replace("$", "").replace("%", "")
        return float(text) if text else None
    except (TypeError, ValueError):
        match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?", str(value or ""))
        if not match:
            return None
        try:
            return float(match.group(0).replace(",", ""))
        except ValueError:
            return None


def entities_from_metadata(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Prefer semantic render metadata series; fall back to entities.

    Category labels are kept faithful to the original chart: a metric suffix
    is appended ONLY to disambiguate grouped bars (the same category name
    appears with several metrics).  Spurious metrics echoed by the VLM
    (e.g. "Australia" as metric) never show up in the rendered label.
    """
    raw: list[dict[str, Any]] = []
    series = metadata.get("series") if isinstance(metadata.get("series"), list) else []
    if series:
        for item in series:
            if not isinstance(item, dict):
                continue
            label = str(item.get("name") or "").strip()
            if not label and not item.get("entity_id"):
                continue
            metric = str(item.get("metric") or "").strip()
            values = item.get("values") if isinstance(item.get("values"), list) else []
            value = _to_float(values[0]) if values else None
            if value is None:
                continue
            raw.append(
                {
                    "label": label,
                    "metric": metric,
                    "value": value,
                    "value_type": item.get("value_type"),
                    "value_read_verified": item.get("value_read_verified"),
                    "side": item.get("side"),
                    "entity_id": str(item["entity_id"]) if item.get("entity_id") else None,
                }
            )
    if not raw:
        for item in metadata.get("entities") if isinstance(metadata.get("entities"), list) else []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            if (not label and not item.get("entity_id")) or label.startswith("entity_"):
                continue
            value = _to_float(item.get("value"))
            if value is None:
                continue
            metric = str(item.get("metric") or "").strip()
            raw.append(
                {
                    "label": label,
                    "metric": metric,
                    "value": value,
                    "value_type": item.get("value_type"),
                    "value_read_verified": item.get("value_read_verified"),
                    "side": item.get("side"),
                    "entity_id": str(item["entity_id"]) if item.get("entity_id") else None,
                }
            )

    metric_counts: dict[str, set[str]] = {}
    for row in raw:
        metric_counts.setdefault(row["label"].lower(), set()).add(row["metric"].lower())
    entities: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in raw:
        key = (
            str(row.get("entity_id") or row["label"]).lower(),
            row["metric"].lower(),
            str(row.get("side") or "").lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        label = row["label"]
        if label and len(metric_counts.get(row["label"].lower(), set())) > 1 and row["metric"]:
            label = f"{label} - {row['metric']}"
        entity = {"label": label, "value": row["value"]}
        if row.get("entity_id"):
            entity["entity_id"] = row["entity_id"]
        if row.get("metric"):
            entity["metric"] = row["metric"]
        if row.get("value_type"):
            entity["value_type"] = row["value_type"]
        if row.get("value_read_verified") is not None:
            entity["value_read_verified"] = row["value_read_verified"]
        if row.get("side"):
            entity["side"] = row["side"]
        entities.append(entity)
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


def _nice_step(span: float, min_count: int = 3, max_count: int = 6) -> float:
    """A 1/2/5x10^k step that yields ``min_count..max_count`` ticks in span."""
    import math

    if span <= 0:
        return 1.0
    exp = math.floor(math.log10(span)) - 1
    best_step: float | None = None
    best_score: int | None = None
    for e in (exp, exp + 1):
        raw = 10 ** e
        for mult in (1, 2, 5, 10):
            candidate = mult * raw
            count = int(span / candidate) + 1
            if min_count <= count <= max_count:
                score = 0
            elif count < min_count:
                score = min_count - count
            else:
                score = count - max_count
            if best_score is None or score < best_score or (
                score == best_score and (best_step is None or candidate < best_step)
            ):
                best_score = score
                best_step = candidate
    return best_step if best_step is not None else 10 ** exp


def _value_ticks(baseline: float, maxv: float) -> list[float]:
    """Ticks on ``[baseline, maxv]`` with a nice step (mirrors _nice_ticks)."""
    baseline = max(0.0, float(baseline))
    span = max(0.0, float(maxv) - baseline)
    if span <= 0:
        return [baseline]
    step = _nice_step(span)
    ticks = []
    v = baseline
    while v <= maxv * 1.02 and len(ticks) < 10:
        ticks.append(round(v, 6))
        v += step
    return ticks


def _nice_baseline(baseline: float, span: float) -> float:
    """Round a fitted axis baseline to a multiple of the tick step.

    The bars' widths imply the real axis origin (e.g. SAT scores start at
    400, not 0).  Rounding to a multiple of the tick step keeps tick labels
    clean (400/600/800/...) while staying close to the fitted origin.  A
    baseline close to zero is treated as 0.
    """
    baseline = float(baseline)
    span = float(span)
    if baseline <= 0 or span <= 0:
        return 0.0
    if baseline < 0.05 * (baseline + span):
        return 0.0
    step = _nice_step(span)
    return max(0.0, round(baseline / step) * step)


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


def match_chart_style(
    image_path: str | Path,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ask the vision model for a visual style spec for the keyframe.

    The returned spec is applied on top of the data-driven layout, so the
    generated SVG resembles the original chart (bar colors, background,
    gridlines, rounded corners, value placement, legend) while the numbers
    still come from the data table.  Best effort: any failure returns {}.
    """
    try:
        text = _call_vision(
            image_path,
            "这是柱状图/条形图的一帧。请一次性读出整张图，只输出一个 JSON 对象（不要解释）："
            '{"colors": {"类别名": "#rrggbb", ...}, "background": "#rrggbb", '
            '"gridlines": true/false, "rounded_corners": 圆角像素数, '
            '"show_values": true/false, "value_position": "above"|"inside"|"right", '
            '"legend": "none"|"top"|"right", "title": "标题原文或空字符串", '
            '"orientation": "vertical"|"horizontal", '
            '"value_labels": true/false（柱上或柱端是否有印刷数值）, '
            '"ticks": [坐标轴刻度标签，按从下到上/从左到右顺序，没有则[]], '
            '"series": [{"label": 柱子标签原文, "value": 数值（印刷值转数字；无印刷值则按柱长相对最长柱0-1估算）, "color": 柱子颜色hex}]}。'
            "颜色必须按画面中每根条形/柱子的实际颜色给出，不能编造；"
            "类别名必须与画面中的名称一致。",
            cfg,
            temperature=0.0,
        )
        obj = _extract_json_object(text)
        if not isinstance(obj, dict):
            return {}
        series = []
        for s in obj.get("series") or []:
            if not isinstance(s, dict):
                continue
            item = {"label": str(s.get("label") or "").strip(), "color": str(s.get("color") or "").strip()}
            try:
                item["value"] = float(s["value"])
            except (KeyError, TypeError, ValueError):
                item["value"] = None
            if item["label"]:
                series.append(item)
        if series:
            obj["series"] = series
        obj.setdefault("value_labels", bool(obj.get("show_values", True)))
        return obj
    except Exception:
        return {}


def _layout_from_geometry(
    entities: list[dict[str, Any]],
    geometry: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Place entities at their real (CV-detected) pixel boxes."""
    by_id: dict[str, dict[str, Any]] = {}
    by_label: dict[str, dict[str, Any]] = {}
    for g in geometry:
        eid = _slug(str(g.get("entity_id") or g.get("label") or ""))
        if eid:
            by_id.setdefault(eid, g)
        label = _normalize_label(g.get("label"))
        if label:
            by_label.setdefault(label, g)
    out: list[dict[str, Any]] = []
    for e in entities:
        eid = _slug(str(e.get("entity_id") or e["label"]))
        g = by_id.get(eid) or by_label.get(_normalize_label(e["label"]))
        if not g:
            continue
        out.append(
            {
                **e,
                "x": float(g["x"]),
                "y": float(g["y"]),
                "w": float(g["w"]),
                "h": float(g["h"]),
            }
        )
    return out


def _geometry_value_scale(
    layout: list[dict[str, Any]],
    horizontal: bool,
) -> dict[str, float] | None:
    """Value-axis calibration derived from the bar geometry itself.

    Vertical bars encode the value with their height (baseline = bar bottom),
    horizontal bars with their width (start = shared left edge).  When bars
    sit at real pixel positions, ticks must be calibrated from the geometry,
    otherwise the axis floats independently and contradicts the bars.  The
    axis origin is *not* assumed to be 0: a regression over (value, length)
    pairs recovers the real baseline (e.g. a SAT-score axis that starts at
    400), rounded to a multiple of the tick step.
    Returns {"scale", "anchor", "baseline"} or None.
    """
    pairs: list[tuple[float, float]] = []
    anchors: list[float] = []
    for e in layout:
        value = float(e["value"])
        length = float(e["w"]) if horizontal else float(e["h"])
        if value <= 0 or length <= 0:
            continue
        pairs.append((value, length))
        anchors.append(float(e["x"]) if horizontal else (float(e["y"]) + float(e["h"])))
    if not pairs:
        return None
    anchors.sort()
    anchor = anchors[len(anchors) // 2]
    # Exact (directly printed) values are the trustworthy calibration points;
    # estimated values may come from a different scale (vision length
    # calibration) and would skew the fitted axis origin.
    def _is_exact(e: dict[str, Any]) -> bool:
        vt = str(e.get("value_type") or "")
        if vt == "exact":
            return True
        if vt in ("", "None") and e.get("value_read_verified"):
            return True
        return False

    exact_pairs = []
    for e in layout:
        if not _is_exact(e):
            continue
        value = float(e["value"])
        length = float(e["w"]) if horizontal else float(e["h"])
        if value > 0 and length > 0:
            exact_pairs.append((value, length))
    fit_pairs = exact_pairs if len(exact_pairs) >= 2 else pairs
    if len(fit_pairs) >= 2:
        vs = [p[0] for p in fit_pairs]
        ls = [p[1] for p in fit_pairs]
        mean_v = sum(vs) / len(vs)
        mean_l = sum(ls) / len(ls)
        denom = sum((v - mean_v) ** 2 for v in vs)
        if denom > 0:
            scale = sum((v - mean_v) * (l - mean_l) for v, l in fit_pairs) / denom
            intercept = mean_l - scale * mean_v
            if scale > 0:
                baseline = -intercept / scale
                span = max(vs) - baseline
                nice = _nice_baseline(baseline, span) if span > 0 else 0.0
                return {"scale": scale, "anchor": anchor, "baseline": nice}
    scales = sorted(l / v for v, l in fit_pairs if v > 0) or sorted(l / v for v, l in pairs if v > 0)
    return {
        "scale": scales[len(scales) // 2],
        "anchor": anchor,
        "baseline": 0.0,
    }


def _enforce_value_geometry(
    layout: list[dict[str, Any]],
    horizontal: bool,
    metadata: dict[str, Any],
) -> None:
    """Make each bar's value-encoding length match the fitted axis scale.

    The detected geometry (CV/vision boxes) can disagree with the value axis
    fitted from the same bars, so a bar's end does not line up with the tick
    for its value (e.g. an SAT score of 1006 that ends before the 1000 tick).
    When a numeric value axis exists, the value is authoritative: the bar's
    length is recomputed from the fitted scale and its start is pinned to the
    shared axis anchor, so bars, data table and ticks are always consistent.
    Relative / no-axis charts and diverging (pyramid) layouts keep their
    detected geometry.
    """
    if str(metadata.get("value_axis") or "") == "none":
        return
    if any(e.get("side") for e in layout):
        return
    geo_scale = _geometry_value_scale(layout, horizontal)
    if not geo_scale:
        return
    scale = geo_scale["scale"]
    baseline = geo_scale["baseline"]
    anchor = geo_scale["anchor"]
    for e in layout:
        try:
            value = float(e.get("value") or 0.0)
        except (TypeError, ValueError):
            continue
        length = max(0.0, (value - baseline) * scale)
        if horizontal:
            e["x"] = anchor
            e["w"] = length
        else:
            e["y"] = anchor - length
            e["h"] = length


def _bar_slot(layout: list[dict[str, Any]], index: int) -> float:
    """Available horizontal space around one bar (nearest neighbour centres)."""
    centers = [float(e["x"]) + float(e["w"]) / 2 for e in layout]
    gaps = []
    if index > 0:
        gaps.append(centers[index] - centers[index - 1])
    if index < len(layout) - 1:
        gaps.append(centers[index + 1] - centers[index])
    return min(gaps) if gaps else float(layout[index]["w"])


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
        eid = _entity_id(e)
        mid = _mark_id(e)
        x, y, w, hgt = e["x"], e["y"], e["w"], e["h"]
        objects.append(
            {
                "id": f"{mid}-bar",
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
                "id": f"{mid}-value-label",
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
                "id": f"{mid}-label",
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
                "component_ids": [f"{mid}-bar", f"{mid}-value-label", f"{mid}-label"],
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
        "uses_render_metadata": True,
        "metadata_title": title,
        "metadata_chart_type": "bar",
        "chart_type": "vertical_bar",
        "needs_review": False,
        "objects": objects,
        "entity_groups": groups,
        "reconciliation_actions": [],
        "warnings": [],
    }


def _parse_marks_sr(raw: Any) -> list[list[float]]:
    out: list[list[float]] = []
    for m in raw or []:
        if not isinstance(m, (list, tuple)) or len(m) < 2:
            continue
        try:
            out.append([float(m[0]), float(m[1])])
        except (TypeError, ValueError):
            continue
    return out


def _parse_point_labels_sr(raw: Any) -> list[dict[str, Any]]:
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


def _parse_line_series_points(
    series_raw: list[Any],
    x_indexed: bool = False,
    relative: bool = False,
) -> list[dict[str, Any]]:
    series_points: list[dict[str, Any]] = []
    for item in series_raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        color = str(item.get("color") or "")
        curve_type = str(item.get("curve_type") or "polyline")
        line_style = str(item.get("line_style") or "solid")
        try:
            parsed_lw = float(item.get("line_width") or 5)
        except (TypeError, ValueError):
            parsed_lw = 5.0
        raw_points = item.get("points") if isinstance(item.get("points"), list) else None
        if raw_points:
            values: list[float] = []
            x_values: list[float | None] = []
            x_px: list[float | None] = []
            for point in raw_points:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                quad = len(point) >= 4
                parsed_abs = _to_float(point[2]) if quad else _to_float(point[1])
                parsed_rel = _to_float(point[3]) if quad else (_to_float(point[2]) if len(point) > 2 else None)
                if parsed_abs is None and parsed_rel is None:
                    continue
                values.append(parsed_rel if relative else (parsed_abs if parsed_abs is not None else parsed_rel))
                try:
                    px_value = float(point[0])
                except (TypeError, ValueError):
                    px_value = None
                x_px.append(px_value if px_value is not None and px_value >= 0 else None)
                x_values.append(_to_float(point[1]) if quad else None)
            if values:
                series_points.append(
                    {
                        "name": name,
                        "color": color,
                        "curve_type": curve_type,
                        "line_style": line_style,
                        "line_width": parsed_lw,
                        "values": values,
                        "x_values": x_values,
                        "x_px": x_px,
                        "marks": _parse_marks_sr(item.get("marks")),
                        "point_labels": _parse_point_labels_sr(item.get("point_labels")),
                    }
                )
            continue
        raw_values = item.get("values") or []
        raw_x_labels = item.get("x_labels") or []
        values: list[float] = []
        x_values: list[float | None] = []
        for index, value in enumerate(raw_values):
            parsed = _to_float(value)
            if parsed is None:
                continue
            values.append(parsed)
            label = raw_x_labels[index] if index < len(raw_x_labels) else ""
            x_values.append(_parse_x_value(label))
        if values:
            series_points.append(
                {
                    "name": name,
                    "color": color,
                    "curve_type": curve_type,
                    "line_style": line_style,
                    "line_width": parsed_lw,
                    "values": values,
                    "x_values": x_values,
                    "marks": _parse_marks_sr(item.get("marks")),
                    "point_labels": _parse_point_labels_sr(item.get("point_labels")),
                }
            )
    return series_points


def _build_multi_line_svg(plots: list[dict[str, Any]]) -> str:
    """Lay out several independent line charts into one 1280x720 SVG."""
    n = len(plots)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    cell_w = W / cols
    cell_h = H / rows
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" data-role="semantic-chart" data-generator="datavideo.semantic_render_v1">',
        f'<rect id="scene-background-fill" data-role="background-fill" x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>',
    ]
    for i, plot in enumerate(plots):
        col, row = i % cols, i // cols
        x, y = col * cell_w, row * cell_h
        full = _build_line_svg(
            plot["series_points"],
            plot.get("title") or "",
            plot.get("unit") or "",
            plot.get("x_labels") or [],
            "vertical",
            None,
            relative=plot.get("relative", False),
            custom_ticks=plot.get("custom_ticks"),
            image_width=plot.get("image_width"),
            show_values=plot.get("show_values", False),
            style=plot.get("style") or {},
            x_axis_name=plot.get("x_axis_name") or "",
            y_axis_name=plot.get("y_axis_name") or "",
        )
        # Strip the xml + <svg> wrapper so the chart can be embedded.
        inner = full.split("\n", 2)[2].rsplit("</svg>", 1)[0]
        lines.append(
            f'<svg x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" viewBox="0 0 {W} {H}" overflow="visible" data-role="subplot">'
        )
        lines.append(inner)
        lines.append("</svg>")
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _render_line_multi_plot(
    clip_id: str,
    metadata: dict[str, Any],
    plots_meta: list[Any],
    out_dir: str | Path,
) -> dict[str, Any]:
    """Render several independent line charts into one SVG (multiple axes)."""
    out_dir = ensure_dir(out_dir)
    image_width = metadata.get("image_width")
    plots: list[dict[str, Any]] = []
    for plot in plots_meta:
        if not isinstance(plot, dict):
            continue
        series_points = _parse_line_series_points(
            plot.get("series") if isinstance(plot.get("series"), list) else [],
            bool(plot.get("x_indexed", False)),
            bool(plot.get("relative", False)),
        )
        if not series_points:
            continue
        plots.append(
            {
                "series_points": series_points,
                "title": str(plot.get("title") or "").replace("\r", " ").split("\n", 1)[0].strip(),
                "unit": _sanitize_unit(plot.get("unit"), plot),
                "x_labels": plot.get("x_labels") if isinstance(plot.get("x_labels"), list) else [],
                "relative": bool(plot.get("relative", False)),
                "custom_ticks": plot.get("ticks") if isinstance(plot.get("ticks"), list) else None,
                "show_values": bool(plot.get("show_values", False)),
                "style": plot.get("style") if isinstance(plot.get("style"), dict) else {},
                "image_width": image_width,
                "x_axis_name": str(plot.get("x_axis_name") or ""),
                "y_axis_name": str(plot.get("y_axis_name") or ""),
            }
        )
    if not plots:
        return {
            "tool": "semantic_render",
            "generator": "datavideo.semantic_render_v1",
            "success": False,
            "failure_reason": "no_plot_series_values",
            "entity_count": 0,
        }
    svg_text = _build_multi_line_svg(plots)
    svg_path = out_dir / "semantic.svg"
    components_svg_path = out_dir / "semantic_components.svg"
    svg_path.write_text(svg_text, encoding="utf-8")
    components_svg_path.write_text(svg_text, encoding="utf-8")
    all_series: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []
    for plot in plots:
        for series in plot["series_points"]:
            all_series.append(
                {
                    "name": series["name"],
                    "values": series["values"],
                    "type": "polyline",
                    "unit": plot["unit"],
                }
            )
            entities.append(
                {
                    "entity_id": _slug(series["name"]) or "series",
                    "label": series["name"] or "series",
                    "type": "polyline",
                    "values": series["values"],
                    "unit": plot["unit"],
                }
            )
    comp_path = out_dir / "semantic_components.json"
    scene_path = out_dir / "semantic_scene.json"
    write_json(
        comp_path,
        {
            "clip_id": clip_id,
            "source_keyframe": "",
            "annotation_method": "data_driven_line_render_v1",
            "automation_level": "deterministic",
            "chart_type": "line",
            "needs_review": True,
            "contains_data_values": True,
            "series": all_series,
        },
    )
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
            "entities": entities,
            "non_entity_components": [],
        },
    )
    return {
        "tool": "semantic_render",
        "generator": "datavideo.semantic_render_v1",
        "input": "",
        "annotation": str(comp_path),
        "semantic_svg": str(svg_path),
        "semantic_components_svg": str(components_svg_path),
        "semantic_scene": str(scene_path),
        "success": True,
        "failure_reason": None,
        "entity_count": len(plots),
        "point_count": sum(len(s["values"]) for s in all_series),
    }


def render_data_driven_line(
    clip_id: str,
    metadata: dict[str, Any],
    out_dir: str | Path,
) -> dict[str, Any]:
    """Line-chart semantic render: one polyline per series + data points.

    ``metadata["series"]`` is a list of ``{"name", "values": [...],
    "x_labels": [...]}``; when the per-point x labels are numeric (years,
    months), points are positioned by their true x value instead of evenly.
    The bottom axis labels are positioned by the same x range.
    """
    out_dir = ensure_dir(out_dir)
    plots_meta = metadata.get("plots")
    if isinstance(plots_meta, list) and plots_meta:
        return _render_line_multi_plot(clip_id, metadata, plots_meta, out_dir)
    series_raw = metadata.get("series") if isinstance(metadata.get("series"), list) else []
    relative = bool(metadata.get("relative"))
    x_indexed = bool(metadata.get("x_indexed"))
    custom_ticks = metadata.get("ticks") if isinstance(metadata.get("ticks"), list) else None
    series_points = _parse_line_series_points(series_raw, x_indexed, relative)
    if not series_points:
        return {
            "tool": "semantic_render",
            "generator": "datavideo.semantic_render_v1",
            "success": False,
            "failure_reason": "no_series_values",
            "entity_count": 0,
        }
    title = str(metadata.get("title") or "").replace("\r", " ").split("\n", 1)[0].strip()
    unit = _sanitize_unit(metadata.get("unit"), metadata)
    x_axis_name = str(metadata.get("x_axis_name") or "")
    y_axis_name = str(metadata.get("y_axis_name") or "")
    segment_labels = metadata.get("segment_labels") if isinstance(metadata.get("segment_labels"), list) else []
    reference_lines = metadata.get("reference_lines") if isinstance(metadata.get("reference_lines"), list) else []
    orientation = str(metadata.get("orientation") or "vertical")
    x_labels = metadata.get("x_labels") if isinstance(metadata.get("x_labels"), list) else []
    x_range = _line_x_range(series_points, x_labels)
    image_width = metadata.get("image_width")
    show_values = bool(metadata.get("show_values"))
    style = metadata.get("style") if isinstance(metadata.get("style"), dict) else {}
    svg_path = out_dir / "semantic.svg"
    components_svg_path = out_dir / "semantic_components.svg"
    comp_path = out_dir / "semantic_components.json"
    scene_path = out_dir / "semantic_scene.json"
    preview_path = out_dir / "semantic_preview.png"
    components_preview_path = out_dir / "semantic_components_preview.png"
    svg_path.write_text(
        _build_line_svg(
            series_points,
            title,
            unit,
            x_labels,
            orientation,
            x_range,
            relative=relative,
            custom_ticks=custom_ticks,
            image_width=image_width,
            show_values=show_values,
            style=style,
            x_axis_name=x_axis_name,
            y_axis_name=y_axis_name,
            segment_labels=segment_labels,
            reference_lines=reference_lines,
        ),
        encoding="utf-8",
    )
    components_svg_path.write_text(
        _build_line_components_svg(
            series_points,
            title,
            unit,
            x_labels,
            orientation,
            x_range,
            relative=relative,
            custom_ticks=custom_ticks,
            image_width=image_width,
            show_values=show_values,
            style=style,
            x_axis_name=x_axis_name,
            y_axis_name=y_axis_name,
            segment_labels=segment_labels,
            reference_lines=reference_lines,
        ),
        encoding="utf-8",
    )
    point_count = sum(len(series["values"]) for series in series_points)
    write_json(
        comp_path,
        {
            "clip_id": clip_id,
            "source_keyframe": "",
            "annotation_method": "data_driven_line_render_v1",
            "automation_level": "deterministic",
            "chart_type": "line",
            "needs_review": True,
            "series": [
                {
                    "name": series["name"],
                    "values": series["values"],
                    "type": "polyline",
                    "unit": unit,
                }
                for series in series_points
            ],
        },
    )
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
                    "entity_id": _slug(series["name"]),
                    "label": series["name"],
                    "type": "polyline",
                    "values": series["values"],
                    "unit": unit,
                }
                for series in series_points
            ],
            "non_entity_components": [],
        },
    )
    preview_success = _render_line_preview(series_points, title, unit, x_labels, preview_path, x_range, relative=relative, image_width=image_width, show_values=show_values, style=style, custom_ticks=custom_ticks)
    components_preview_success = _render_line_preview(series_points, title, unit, x_labels, components_preview_path, x_range, relative=relative, image_width=image_width, show_values=show_values, style=style, custom_ticks=custom_ticks)
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
        "success": bool(series_points) and svg_path.exists(),
        "failure_reason": None,
        "preview_success": preview_success,
        "preview_failure_reason": None,
        "components_preview_success": components_preview_success,
        "entity_count": len(series_points),
        "point_count": point_count,
    }


def _parse_x_value(label: Any) -> float | None:
    """Extract a leading numeric value from an x-axis label (2000-01 -> 2000)."""
    match = re.search(r"-?\d+(?:\.\d+)?", str(label or ""))
    return float(match.group(0)) if match else None


def _line_x_range(
    series_points: list[dict[str, Any]],
    x_labels: list[str],
) -> tuple[float, float] | None:
    """Numeric x range shared by data points and bottom labels, or None."""
    nums: list[float] = []
    for series in series_points:
        nums.extend(value for value in series.get("x_values", []) if value is not None)
    if not nums:
        for label in x_labels:
            value = _parse_x_value(label)
            if value is not None:
                nums.append(value)
    if len(nums) < 2 or max(nums) <= min(nums):
        return None
    return (min(nums), max(nums))


def _x_label_x(label: Any, x_range: tuple[float, float] | None) -> float | None:
    if x_range is None:
        return None
    value = _parse_x_value(label)
    if value is None:
        return None
    xmin, xmax = x_range
    return LEFT + (value - xmin) / (xmax - xmin) * (RIGHT - LEFT)


def _line_axis_scale(
    series_points: list[dict[str, Any]],
    relative: bool,
    custom_ticks: list[dict[str, Any]] | None,
) -> tuple[float, float]:
    """Return (y_min, y_max) for the y axis.

    Uses the tick range when ticks exist and cover the data so the chart
    keeps the original's proportions; otherwise scales from 0 with 5%
    headroom.  Relative charts always use 0..1.
    """
    if relative:
        return 0.0, 1.0
    all_values = [v for s in series_points for v in s.get("values", [])]
    if not all_values:
        return 0.0, 1.0
    data_min = min(all_values)
    data_max = max(all_values)
    y_min, y_max = 0.0, data_max * 1.05
    if custom_ticks:
        vals = []
        for tick in custom_ticks:
            try:
                vals.append(float(tick["value"]))
            except (TypeError, ValueError, KeyError):
                continue
        if len(vals) >= 2:
            tmin, tmax = min(vals), max(vals)
            bottom = tmin if data_min >= tmin * 0.9 else 0.0
            top = max(tmax, data_max)
            if top > bottom:
                return float(bottom), float(top)
    if y_max <= y_min:
        y_max = y_min + 1.0
    return y_min, y_max


def _line_point_xy(
    values: list[float],
    index: int,
    maxv: float,
    x_range: tuple[float, float] | None = None,
    x_values: list[float | None] | None = None,
    x_px: list[float | None] | None = None,
    image_width: float | int | None = None,
    y_min: float = 0.0,
) -> tuple[float, float]:
    count = len(values)
    span = (maxv - y_min) or 1.0

    def _y(value: float) -> float:
        return BOTTOM - (value - y_min) / span * (BOTTOM - TOP)

    if x_range is not None and x_values is not None and index < len(x_values):
        xv = x_values[index]
        if xv is not None:
            xmin, xmax = x_range
            x = LEFT + (xv - xmin) / (xmax - xmin) * (RIGHT - LEFT)
            return x, _y(float(values[index]))
    if count > 1:
        x = LEFT + index / (count - 1) * (RIGHT - LEFT)
    else:
        x = LEFT + (RIGHT - LEFT) / 2
    return x, _y(float(values[index]))


def _catmull_rom_path(points: list[tuple[float, float]]) -> str:
    """Smooth SVG path through the sampled points (Catmull-Rom -> cubic Bezier)."""
    d = f"M {points[0][0]:.1f} {points[0][1]:.1f}"
    for i in range(len(points) - 1):
        p0 = points[i - 1] if i > 0 else points[0]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[i + 2] if i + 2 < len(points) else points[-1]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0)
        d += f" C {c1[0]:.1f} {c1[1]:.1f}, {c2[0]:.1f} {c2[1]:.1f}, {p2[0]:.1f} {p2[1]:.1f}"
    return d


def _catmull_rom_points(points: list[tuple[float, float]], samples: int = 8) -> list[tuple[float, float]]:
    """Sample a Catmull-Rom spline through points for bitmap drawing (PIL has no bezier)."""
    if len(points) < 2:
        return list(points)
    out: list[tuple[float, float]] = []
    for i in range(len(points) - 1):
        p0 = points[i - 1] if i > 0 else points[0]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[i + 2] if i + 2 < len(points) else points[-1]
        for s in range(samples):
            t = s / samples
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (
                (2 * p1[0])
                + (-p0[0] + p2[0]) * t
                + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2 * p1[1])
                + (-p0[1] + p2[1]) * t
                + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
            )
            out.append((x, y))
    out.append(points[-1])
    return out


def _build_line_svg(
    series_points: list[dict[str, Any]],
    title: str,
    unit: str,
    x_labels: list[str],
    orientation: str = "vertical",
    x_range: tuple[float, float] | None = None,
    relative: bool = False,
    custom_ticks: list[dict[str, Any]] | None = None,
    image_width: float | int | None = None,
    show_values: bool = False,
    style: dict[str, Any] | None = None,
    x_axis_name: str = "",
    y_axis_name: str = "",
    segment_labels: list[dict[str, Any]] | None = None,
    reference_lines: list[dict[str, Any]] | None = None,
) -> str:
    style = style or {}
    background = str(style.get("background") or "#ffffff")
    axis_color = str(style.get("axis_color") or "#666666")
    tick_color = str(style.get("tick_color") or "#444444")
    label_color = str(style.get("label_color") or "#444444")
    title_color = str(style.get("title_color") or "#222222")
    gridlines = bool(style.get("gridlines", False))
    legend_mode = str(style.get("legend") or "none").lower()
    try:
        _h = background.lstrip("#")
        _r, _g, _b = (int(_h[i : i + 2], 16) for i in (0, 2, 4))
        _dark_bg = 0.299 * _r + 0.587 * _g + 0.114 * _b < 128
    except Exception:
        _dark_bg = False
    if _dark_bg:
        if tick_color == "#444444":
            tick_color = "#cccccc"
        if label_color == "#444444":
            label_color = "#e5e5e5"
        if axis_color == "#666666":
            axis_color = "#cccccc"
        if title_color == "#222222":
            title_color = "#f2f2f2"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" data-role="semantic-chart" data-generator="datavideo.semantic_render_v1">',
        f'<rect id="scene-background-fill" data-role="background-fill" x="0" y="0" width="{W}" height="{H}" fill="{background}"/>',
        '<g id="chart-plot" data-role="plot">',
        f'<line data-role="axis" x1="{LEFT}" y1="{BOTTOM}" x2="{RIGHT}" y2="{BOTTOM}" stroke="{axis_color}" stroke-width="3"/>',
        f'<line data-role="axis" x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{BOTTOM}" stroke="{axis_color}" stroke-width="3"/>',
    ]
    if title:
        lines.insert(3, f'<text id="chart-title" data-role="title" x="{W / 2}" y="70" text-anchor="middle" font-family="Arial, sans-serif" font-size="36" font-weight="700" fill="{title_color}">{html.escape(title)}</text>')
    y_min, maxv = _line_axis_scale(series_points, relative, custom_ticks)
    tick_specs: list[tuple[float, str]] = []
    if not relative:
        if custom_ticks:
            for tick in custom_ticks:
                try:
                    tick_specs.append((float(tick["value"]), str(tick.get("label") or tick["value"])))
                except (KeyError, TypeError, ValueError):
                    continue
        else:
            tick_specs = [(tv, _format_value(tv, unit)) for tv in _nice_ticks(maxv)]
    if gridlines:
        for tv, _ in tick_specs:
            ty = BOTTOM - (tv - y_min) / (maxv - y_min) * (BOTTOM - TOP)
            lines.append(f'<line data-role="gridline" x1="{LEFT}" y1="{ty:.1f}" x2="{RIGHT}" y2="{ty:.1f}" stroke="{tick_color}" stroke-width="1" stroke-opacity="0.35"/>')
    for tv, label in tick_specs:
        ty = BOTTOM - (tv - y_min) / (maxv - y_min) * (BOTTOM - TOP)
        lines.append(f'<line data-role="tick" x1="{LEFT - 8}" y1="{ty:.1f}" x2="{LEFT}" y2="{ty:.1f}" stroke="{tick_color}" stroke-width="2"/>')
        lines.append(f'<text data-role="tick-label" x="{LEFT - 16}" y="{ty + 6:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="22" fill="{tick_color}">{html.escape(str(label))}</text>')
    for series_index, series in enumerate(series_points):
        color = str(series.get("color") or "") or _color(series_index)
        curve_type = str(series.get("curve_type") or "polyline")
        line_style = str(series.get("line_style") or "solid")
        try:
            line_width = float(series.get("line_width") or 5)
        except (TypeError, ValueError):
            line_width = 5.0
        eid = _slug(series["name"]) or f"series-{series_index}"
        values = series["values"]
        points = [
            _line_point_xy(values, index, maxv, x_range, series.get("x_values"), series.get("x_px"), image_width, y_min)
            for index in range(len(values))
        ]
        if relative:
            points = [(x, min(BOTTOM, max(TOP, y))) for x, y in points]
        series["points_px"] = points
        dash = ' stroke-dasharray="8,6"' if line_style == "dashed" else ""
        lines.append(
            f'<g id="entity-{eid}" data-role="entity" data-entity-id="{eid}" data-label="{html.escape(series["name"])}" data-series="true">'
        )
        if curve_type == "smooth" and len(points) >= 3:
            lines.append(
                f'<path id="{eid}-path" data-role="line" data-entity-id="{eid}" d="{_catmull_rom_path(points)}" '
                f'fill="none" stroke="{color}" stroke-width="{line_width:g}"{dash}/>'
            )
        else:
            polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            lines.append(
                f'<polyline id="{eid}-polyline" data-role="polyline" data-entity-id="{eid}" points="{polyline}" '
                f'fill="none" stroke="{color}" stroke-width="{line_width:g}"{dash} data-animation-property="points"/>'
            )
        if not relative and show_values:
            for index, (x, y) in enumerate(points):
                lines.append(
                    f'<text data-role="value-label" data-entity-id="{eid}" data-index="{index}" x="{x:.1f}" y="{y - 14:.1f}" '
                    f'text-anchor="middle" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#222222">{html.escape(_format_value(values[index], unit))}</text>'
                )
        # marks + point labels anchored by x value
        anchor = {}
        for index, (x, y) in enumerate(points):
            xv = (series.get("x_values") or [None] * len(points))[index]
            if xv is not None:
                anchor[round(float(xv), 3)] = (x, y)
        for mx, _my in series.get("marks") or []:
            key = min(anchor, key=lambda k: abs(k - round(float(mx), 3))) if anchor else None
            if key is None:
                continue
            ax, ay = anchor[key]
            lines.append(
                f'<circle data-role="point-mark" data-entity-id="{eid}" cx="{ax:.1f}" cy="{ay:.1f}" r="6" fill="{color}" stroke="#ffffff" stroke-width="2"/>'
            )
        for pl in series.get("point_labels") or []:
            key = min(anchor, key=lambda k: abs(k - round(float(pl.get("x")), 3))) if anchor else None
            if key is None:
                continue
            ax, ay = anchor[key]
            label_lines = [ln for ln in str(pl.get("label") or "").splitlines() if ln.strip()]
            if not label_lines:
                continue
            lstart = ay - 18 - 18 * (len(label_lines) - 1)
            ts = "".join(
                f'<tspan x="{ax:.1f}" dy="{18 if i else 0}">{html.escape(ln)}</tspan>'
                for i, ln in enumerate(label_lines)
            )
            lines.append(
                f'<text data-role="point-label" x="{ax:.1f}" y="{lstart:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="{label_color}">{ts}</text>'
            )
        lines.append("</g>")
    if len(x_labels) >= 2:
        for index, label in enumerate(x_labels):
            lx = LEFT + index / (len(x_labels) - 1) * (RIGHT - LEFT)
            positioned = _x_label_x(label, x_range)
            if positioned is not None and x_range is not None:
                xmin_r, xmax_r = x_range
                parsed = _parse_x_value(label)
                if parsed is not None and (parsed < xmin_r or parsed > xmax_r):
                    positioned = None
            if positioned is not None:
                lx = positioned
            lines.append(
                f'<text data-role="x-axis-label" x="{lx:.1f}" y="{BOTTOM + 30}" text-anchor="middle" '
                f'font-family="Arial, sans-serif" font-size="22" fill="{label_color}">{html.escape(str(label))}</text>'
            )
    legend_entries = [s for s in series_points if str(s.get("name") or "").strip()]
    if legend_mode == "none" and legend_entries:
        legend_mode = "top"
    for ref in reference_lines or []:
        try:
            rv = float(ref.get("value"))
        except (TypeError, ValueError):
            continue
        ry = BOTTOM - (rv - y_min) / (maxv - y_min) * (BOTTOM - TOP)
        ry = min(BOTTOM, max(TOP, ry))
        lines.append(f'<line data-role="reference-line" x1="{LEFT}" y1="{ry:.1f}" x2="{RIGHT}" y2="{ry:.1f}" stroke="#b0b0b0" stroke-width="2" stroke-dasharray="6,5"/>')
        if ref.get("label"):
            lines.append(f'<text data-role="reference-label" x="{RIGHT - 8}" y="{ry - 8:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="18" fill="#888888">{html.escape(str(ref.get("label")))}</text>')
    for seg in segment_labels or []:
        try:
            sxs = float(seg.get("x_start")); sxe = float(seg.get("x_end"))
        except (TypeError, ValueError):
            continue
        if x_range is not None:
            xmin_r, xmax_r = x_range
            if xmax_r > xmin_r:
                sx0 = LEFT + (sxs - xmin_r) / (xmax_r - xmin_r) * (RIGHT - LEFT)
                sx1 = LEFT + (sxe - xmin_r) / (xmax_r - xmin_r) * (RIGHT - LEFT)
            else:
                continue
        else:
            continue
        lines.append(f'<text data-role="segment-label" x="{(sx0 + sx1) / 2:.1f}" y="{TOP - 12}" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="{label_color}">{html.escape(str(seg.get("label")))}</text>')
    if x_axis_name:
        lines.append(f'<text data-role="x-axis-name" x="{W / 2}" y="{BOTTOM + 64}" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" fill="{label_color}">{html.escape(x_axis_name)}</text>')
    if y_axis_name:
        lines.append(f'<text data-role="y-axis-name" transform="translate(34, {(TOP + BOTTOM) / 2}) rotate(-90)" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" fill="{label_color}">{html.escape(y_axis_name)}</text>')
    if legend_mode == "end" and legend_entries:
        for _s in legend_entries:
            _end = _s.get("points_px")
            if not _end:
                continue
            _ex, _ey = _end[-1]
            _ec = str(_s.get("color") or "") or _color(legend_entries.index(_s))
            lines.append(
                f'<text data-role="end-label" x="{_ex + 10:.1f}" y="{_ey + 7:.1f}" text-anchor="start" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="{_ec}">{html.escape(str(_s.get("name")))}</text>'
            )
    if legend_mode != "none" and legend_mode != "end" and legend_entries:
        if legend_mode == "right":
            legend_x0 = RIGHT - 300
            legend_text_x = RIGHT - 200
        else:
            legend_x0 = W / 2 - 120
            legend_text_x = W / 2 - 30
        legend_y = 92
        for series_index, series in enumerate(legend_entries):
            color = str(series.get("color") or "") or _color(series_index)
            try:
                legend_w = float(series.get("line_width") or 5)
            except (TypeError, ValueError):
                legend_w = 5.0
            lines.append(
                f'<line x1="{legend_x0}" y1="{legend_y}" x2="{legend_x0 + 80}" y2="{legend_y}" stroke="{color}" stroke-width="{legend_w:g}"'
                f'{" stroke-dasharray=\"8,6\"" if str(series.get("line_style") or "solid") == "dashed" else ""}/>'
            )
            lines.append(
                f'<text x="{legend_text_x}" y="{legend_y + 8}" font-family="Arial, sans-serif" font-size="22" fill="{label_color}">{html.escape(series["name"])}</text>'
            )
            legend_y += 32
    lines.append("</g>")
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _build_line_components_svg(
    series_points: list[dict[str, Any]],
    title: str,
    unit: str,
    x_labels: list[str],
    orientation: str = "vertical",
    x_range: tuple[float, float] | None = None,
    relative: bool = False,
    custom_ticks: list[dict[str, Any]] | None = None,
    image_width: float | int | None = None,
    show_values: bool = False,
    style: dict[str, Any] | None = None,
    x_axis_name: str = "",
    y_axis_name: str = "",
    segment_labels: list[dict[str, Any]] | None = None,
    reference_lines: list[dict[str, Any]] | None = None,
) -> str:
    style = style or {}
    background = str(style.get("background") or "#ffffff")
    axis_color = str(style.get("axis_color") or "#666666")
    tick_color = str(style.get("tick_color") or "#444444")
    label_color = str(style.get("label_color") or "#444444")
    title_color = str(style.get("title_color") or "#222222")
    gridlines = bool(style.get("gridlines", False))
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" data-role="semantic-chart" data-generator="datavideo.semantic_render_v1">',
        f'<rect id="scene-background-fill" data-role="background-fill" x="0" y="0" width="{W}" height="{H}" fill="{background}"/>',
        '<g id="chart-plot" data-role="plot">',
    ]
    if title:
        lines.insert(3, f'<text id="chart-title" data-role="title" x="{W / 2}" y="67" text-anchor="middle" font-family="Arial, sans-serif" font-size="34" font-weight="700" fill="{title_color}">{html.escape(title)}</text>')
        lines.insert(4, f'<rect id="chart-title-box" data-role="title-box" x="{W / 2 - 300}" y="32" width="600" height="54" fill="{background}" stroke="{axis_color}" stroke-width="2"/>')
    y_min, maxv = _line_axis_scale(series_points, relative, custom_ticks)
    tick_specs: list[tuple[float, str]] = []
    if not relative:
        if custom_ticks:
            for tick in custom_ticks:
                try:
                    tick_specs.append((float(tick["value"]), str(tick.get("label") or tick["value"])))
                except (KeyError, TypeError, ValueError):
                    continue
        else:
            tick_specs = [(tv, _format_value(tv, unit)) for tv in _nice_ticks(maxv)]
    if gridlines:
        for tv, _ in tick_specs:
            ty = BOTTOM - (tv - y_min) / (maxv - y_min) * (BOTTOM - TOP)
            lines.append(f'<line data-role="gridline" x1="{LEFT}" y1="{ty:.1f}" x2="{RIGHT}" y2="{ty:.1f}" stroke="{tick_color}" stroke-width="1" stroke-opacity="0.35"/>')
    for tv, label in tick_specs:
        ty = BOTTOM - (tv - y_min) / (maxv - y_min) * (BOTTOM - TOP)
        lines.append(f'<line data-role="tick" x1="{LEFT - 8}" y1="{ty:.1f}" x2="{LEFT}" y2="{ty:.1f}" stroke="{tick_color}" stroke-width="2"/>')
        lines.append(f'<text data-role="tick-label" x="{LEFT - 16}" y="{ty + 6:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="22" fill="{tick_color}">{html.escape(str(label))}</text>')
    for series_index, series in enumerate(series_points):
        color = str(series.get("color") or "") or _color(series_index)
        try:
            line_width = float(series.get("line_width") or 5)
        except (TypeError, ValueError):
            line_width = 5.0
        eid = _slug(series["name"]) or f"series-{series_index}"
        values = series["values"]
        points = [
            _line_point_xy(values, index, maxv, x_range, series.get("x_values"), series.get("x_px"), image_width, y_min)
            for index in range(len(values))
        ]
        if relative:
            points = [(x, min(BOTTOM, max(TOP, y))) for x, y in points]
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        lines.append(f'<g id="entity-{eid}" data-role="entity" data-entity-id="{eid}" data-label="{html.escape(series["name"])}">')
        lines.append(f'<polyline data-role="polyline" data-entity-id="{eid}" points="{polyline}" fill="none" stroke="{color}" stroke-width="{line_width:g}"/>')
        for index, (x, y) in enumerate(points):
            if not relative and show_values:
                lines.append(
                    f'<rect data-role="value-box" data-entity-id="{eid}" data-index="{index}" x="{x - 28:.1f}" y="{y - 32:.1f}" width="56" height="26" fill="#fafafa" stroke="#333333" stroke-width="2"/>'
                )
            if not relative and show_values:
                lines.append(
                    f'<text data-role="value-label" data-entity-id="{eid}" data-index="{index}" x="{x:.1f}" y="{y - 14:.1f}" '
                    f'text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#222222">{html.escape(_format_value(values[index], unit))}</text>'
                )
        lines.append("</g>")
    if x_axis_name:
        lines.append(f'<text data-role="x-axis-name" x="{W / 2}" y="{BOTTOM + 64}" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" fill="{label_color}">{html.escape(x_axis_name)}</text>')
    if y_axis_name:
        lines.append(f'<text data-role="y-axis-name" transform="translate(34, {(TOP + BOTTOM) / 2}) rotate(-90)" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" fill="{label_color}">{html.escape(y_axis_name)}</text>')
    if len(x_labels) >= 2:
        for index, label in enumerate(x_labels):
            lx = LEFT + index / (len(x_labels) - 1) * (RIGHT - LEFT)
            positioned = _x_label_x(label, x_range)
            if positioned is not None:
                lx = positioned
            lines.append(
                f'<text data-role="x-axis-label" x="{lx:.1f}" y="{BOTTOM + 30}" text-anchor="middle" '
                f'font-family="Arial, sans-serif" font-size="20" fill="{label_color}">{html.escape(str(label))}</text>'
            )
    lines.append("</g>")
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _render_line_preview(
    series_points: list[dict[str, Any]],
    title: str,
    unit: str,
    x_labels: list[str],
    out: Path,
    x_range: tuple[float, float] | None = None,
    relative: bool = False,
    image_width: float | int | None = None,
    show_values: bool = False,
    style: dict[str, Any] | None = None,
    custom_ticks: list[dict[str, Any]] | None = None,
) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return False
    style = style or {}

    def _rgb(hex_color: str) -> tuple[int, int, int]:
        try:
            h = hex_color.lstrip("#")
            return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
        except Exception:
            return (255, 255, 255)

    bg = _rgb(str(style.get("background") or "#ffffff"))
    axis_c = _rgb(str(style.get("axis_color") or "#666666"))
    tick_c = _rgb(str(style.get("tick_color") or "#444444"))
    label_c = _rgb(str(style.get("label_color") or "#444444"))
    title_c = _rgb(str(style.get("title_color") or "#222222"))
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)
    try:
        font_t = ImageFont.truetype("arial.ttf", 34)
        font_v = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font_t = font_v = ImageFont.load_default()
    if title:
        draw.text((W / 2 - draw.textlength(title, font=font_t) / 2, 30), title, fill=title_c, font=font_t)
    draw.line([(LEFT, BOTTOM), (RIGHT, BOTTOM)], fill=axis_c, width=3)
    draw.line([(LEFT, TOP), (LEFT, BOTTOM)], fill=axis_c, width=3)
    y_min, maxv = _line_axis_scale(series_points, relative, custom_ticks)
    if not relative:
        tick_vals: list[tuple[float, str]] = []
        if custom_ticks:
            for tick in custom_ticks:
                try:
                    tick_vals.append((float(tick["value"]), str(tick.get("label") or tick["value"])))
                except (KeyError, TypeError, ValueError):
                    continue
        if not tick_vals:
            tick_vals = [(tv, _format_value(tv, unit)) for tv in _nice_ticks(maxv)]
        for tv, label in tick_vals:
            ty = BOTTOM - (tv - y_min) / (maxv - y_min) * (BOTTOM - TOP)
            draw.line([(LEFT - 8, ty), (LEFT, ty)], fill=tick_c, width=2)
            draw.text((LEFT - 70, ty - 12), label, fill=tick_c, font=font_v)

    def _dash_segments(pts, dash_on=10, dash_off=6):
        segs = []
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if length <= 0:
                continue
            dx, dy = (x2 - x1) / length, (y2 - y1) / length
            pos = 0.0
            on = True
            while pos < length:
                run = min((dash_on if on else dash_off), length - pos)
                if on:
                    segs.append(((x1 + dx * pos, y1 + dy * pos), (x1 + dx * (pos + run), y1 + dy * (pos + run))))
                pos += run
                on = not on
        return segs

    for series_index, series in enumerate(series_points):
        color_hex = str(series.get("color") or "") or _color(series_index)
        color = tuple(int(color_hex[index : index + 2], 16) for index in (1, 3, 5))
        values = series["values"]
        points = [
            _line_point_xy(values, index, maxv, x_range, series.get("x_values"), series.get("x_px"), image_width, y_min)
            for index in range(len(values))
        ]
        if relative:
            points = [(x, min(BOTTOM, max(TOP, y))) for x, y in points]
        try:
            line_width = float(series.get("line_width") or 5)
        except (TypeError, ValueError):
            line_width = 5.0
        if str(series.get("curve_type") or "polyline") == "smooth" and len(points) >= 3:
            draw_points = _catmull_rom_points(points)
        else:
            draw_points = points
        draw.line(draw_points, fill=color, width=int(round(line_width)))
        if str(series.get("line_style") or "solid") == "dashed":
            for seg in _dash_segments(draw_points):
                draw.line(seg, fill=color, width=int(round(line_width)))
        if not relative and show_values:
            for index, (x, y) in enumerate(points):
                label = _format_value(values[index], unit)
                tw = draw.textlength(label, font=font_v)
                draw.text((x - tw / 2, y - 26), label, fill=(20, 20, 20), font=font_v)
    if len(x_labels) >= 2:
        for index, label in enumerate(x_labels):
            lx = LEFT + index / (len(x_labels) - 1) * (RIGHT - LEFT)
            positioned = _x_label_x(label, x_range)
            if positioned is not None:
                lx = positioned
            draw.text((lx - draw.textlength(str(label), font=font_v) / 2, BOTTOM + 8), str(label), fill=label_c, font=font_v)
    img.save(out)
    return out.exists()


def _build_svg(
    layout: list[dict[str, Any]],
    title: str,
    unit: str,
    orientation: str = "vertical",
    style: dict[str, Any] | None = None,
) -> str:
    style = style or {}
    horizontal = orientation == "horizontal"
    background = str(style.get("background") or "#ffffff")

    def _lum(hex_color: str) -> float:
        try:
            h = hex_color.lstrip("#")
            r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
        except Exception:
            return 255.0
        return 0.299 * r + 0.587 * g + 0.114 * b

    # Text color follows the background so dark-mode charts stay readable.
    dark_bg = _lum(background) < 128
    text_color = "#f2f2f2" if dark_bg else "#222222"
    tick_color = "#c9c9c9" if dark_bg else "#444444"
    label_color = "#e5e5e5" if dark_bg else "#333333"
    title = str(style.get("title") or title)
    colors = style.get("colors") if isinstance(style.get("colors"), dict) else {}
    gridlines = bool(style.get("gridlines", False))
    rounded = float(style.get("rounded_corners") or 0)
    show_values = bool(style.get("show_values", True))
    value_position = str(style.get("value_position") or ("right" if horizontal else "above"))
    legend = str(style.get("legend") or "none")
    maxv = max((e["value"] for e in layout), default=1.0) or 1.0
    geo_scale = _geometry_value_scale(layout, horizontal)
    custom_ticks = style.get("custom_ticks")
    if not isinstance(custom_ticks, list):
        custom_ticks = None
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" data-role="semantic-chart" data-generator="datavideo.semantic_render_v1">',
        f'<rect id="scene-background-fill" data-role="background-fill" x="0" y="0" width="{W}" height="{H}" fill="{html.escape(background)}"/>',
        f'<text id="chart-title" data-role="title" x="{W / 2}" y="70" text-anchor="middle" font-family="Arial, sans-serif" font-size="36" font-weight="700" fill="{text_color}">{html.escape(title)}</text>',
        '<g id="chart-plot" data-role="plot">',
    ]
    if geo_scale is not None:
        # Geometry mode: the bars carry the real positions, so the value axis
        # (and its baseline tick) sits exactly at the bars' common start line
        # instead of the canvas edge.
        if horizontal:
            anchor_x = geo_scale["anchor"]
            lines.append(f'<line data-role="axis" x1="{anchor_x:.1f}" y1="{TOP}" x2="{anchor_x:.1f}" y2="{BOTTOM}" stroke="#666666" stroke-width="3"/>')
            lines.append(f'<line data-role="axis" x1="{anchor_x:.1f}" y1="{BOTTOM}" x2="{RIGHT}" y2="{BOTTOM}" stroke="#666666" stroke-width="3"/>')
        else:
            anchor_y = geo_scale["anchor"]
            lines.append(f'<line data-role="axis" x1="{LEFT}" y1="{anchor_y:.1f}" x2="{RIGHT}" y2="{anchor_y:.1f}" stroke="#666666" stroke-width="3"/>')
            lines.append(f'<line data-role="axis" x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{anchor_y:.1f}" stroke="#666666" stroke-width="3"/>')
    else:
        lines.append(f'<line data-role="axis" x1="{LEFT}" y1="{BOTTOM}" x2="{RIGHT}" y2="{BOTTOM}" stroke="#666666" stroke-width="3"/>')
        lines.append(f'<line data-role="axis" x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{BOTTOM}" stroke="#666666" stroke-width="3"/>')
    baseline = float(geo_scale.get("baseline") or 0.0) if geo_scale else 0.0
    suppress_axis = str(style.get("value_axis") or "") == "none"
    tick_items = []
    if custom_ticks:
        for t in custom_ticks:
            if not isinstance(t, dict):
                continue
            try:
                tick_items.append((float(t["value"]), str(t.get("label") or t.get("value") or "")))
            except (TypeError, ValueError, KeyError):
                continue
    else:
        tick_items = [
            (tv, _format_value(tv, unit))
            for tv in ([] if suppress_axis else _value_ticks(baseline, maxv * 1.08))
        ]
    for tv, tick_label in tick_items:
        if horizontal:
            tx = (geo_scale["anchor"] + (tv - baseline) * geo_scale["scale"]) if geo_scale else LEFT + tv / maxv * (RIGHT - LEFT)
            if gridlines and tv > baseline:
                lines.append(f'<line data-role="gridline" x1="{tx:.1f}" y1="{TOP}" x2="{tx:.1f}" y2="{BOTTOM}" stroke="#cccccc" stroke-width="1" stroke-dasharray="4,4"/>')
            lines.append(f'<line data-role="tick" x1="{tx:.1f}" y1="{BOTTOM}" x2="{tx:.1f}" y2="{BOTTOM + 8}" stroke="#666666" stroke-width="2"/>')
            lines.append(f'<text data-role="tick-label" x="{tx:.1f}" y="{BOTTOM + 30:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" fill="{tick_color}">{html.escape(tick_label)}</text>')
        else:
            ty = (geo_scale["anchor"] - (tv - baseline) * geo_scale["scale"]) if geo_scale else BOTTOM - tv / maxv * (BOTTOM - TOP)
            if gridlines and tv > baseline:
                lines.append(f'<line data-role="gridline" x1="{LEFT}" y1="{ty:.1f}" x2="{RIGHT}" y2="{ty:.1f}" stroke="#cccccc" stroke-width="1" stroke-dasharray="4,4"/>')
            lines.append(f'<line data-role="tick" x1="{LEFT - 8}" y1="{ty:.1f}" x2="{LEFT}" y2="{ty:.1f}" stroke="#666666" stroke-width="2"/>')
            lines.append(f'<text data-role="tick-label" x="{LEFT - 16}" y="{ty + 6:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="22" fill="{tick_color}">{html.escape(tick_label)}</text>')
    for i, e in enumerate(layout):
        eid = _entity_id(e)
        mid = _mark_id(e)
        color = str(colors.get(e["label"]) or colors.get(eid) or _color(i))
        lines.append(f'<g id="entity-{mid}" data-role="entity" data-entity-id="{eid}" data-label="{html.escape(e["label"])}">')
        radius_attr = f' rx="{rounded:.1f}" ry="{rounded:.1f}"' if rounded > 0 else ""
        lines.append(
            f'<rect id="{mid}-bar" data-role="bar" data-entity-id="{eid}" data-value="{e["value"]:g}" '
            f'x="{e["x"]:.1f}" y="{e["y"]:.1f}" width="{e["w"]:.1f}" height="{e["h"]:.1f}" fill="{html.escape(color)}"{radius_attr} '
            f'data-animation-property="{"width" if horizontal else "height"}" data-anchor="{"left" if horizontal else "bottom"}" data-animation-axis="{"x" if horizontal else "y"}" data-orientation="{orientation}"/>'
        )
        if value_position == "inside":
            value_label_x = (e["x"] + e["w"] / 2) if horizontal else (e["x"] + e["w"] / 2)
            value_label_y = (e["y"] + e["h"] / 2 + 8) if horizontal else (e["y"] + e["h"] / 2 + 8)
            value_anchor = "middle"
        elif horizontal:
            value_label_x = e["x"] + e["w"] + 14
            value_label_y = e["y"] + e["h"] / 2 + 8
            value_anchor = "start"
        else:
            value_label_x = e["x"] + e["w"] / 2
            value_label_y = e["y"] - 14
            value_anchor = "middle"
        if horizontal:
            # Horizontal-bar charts label each bar to the LEFT of the value
            # axis, vertically centred on the bar -- not above the bar.
            category_label_x = e["x"] - 14
            category_label_y = e["y"] + e["h"] / 2 - 8
            category_anchor = "end"
        else:
            category_label_x = e["x"] + e["w"] / 2
            category_label_y = BOTTOM + 30
            category_anchor = "middle"
        # Keep the label font constant; wrap to multiple lines when the label
        # would overflow its slot instead of shrinking the text.
        side = str(e.get("side") or "").strip().lower()
        if horizontal and side == "left":
            # Diverging / pyramid charts: the mirrored left+right bars share
            # one category label, drawn once by the right-side bar, centered
            # in the gap between the pair.
            label_lines = []
        elif horizontal and side == "right":
            left_bar = next(
                (
                    x
                    for x in layout
                    if x.get("label") == e.get("label")
                    and str(x.get("side") or "").strip().lower() == "left"
                ),
                None,
            )
            if left_bar is not None:
                # Population pyramids label each row on the far left, with the
                # mirrored bars extending from the center axis.
                category_label_x = 150.0
                category_anchor = "end"
                label_lines = _wrap_text(e["label"], 130.0)
            else:
                label_lines = _wrap_text(e["label"], max(40.0, category_label_x - 12))
        elif horizontal:
            label_lines = _wrap_text(e["label"], max(40.0, category_label_x - 12))
        else:
            label_lines = _wrap_text(e["label"], max(40.0, _bar_slot(layout, i) * 0.95))
        label_font_size = 20
        if show_values:
            lines.append(
                f'<text id="{eid}-value-label" data-role="value-label" data-entity-id="{eid}" '
                f'x="{value_label_x:.1f}" y="{value_label_y:.1f}" text-anchor="{value_anchor}" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="{text_color}">{html.escape(_format_value(e["value"], unit))}</text>'
            )
        label_parts = [html.escape(part) for part in label_lines[:3]]
        if len(label_lines) > 3:
            label_parts[2] = label_parts[2][: max(1, len(label_parts[2]) - 1)] + "&#8230;"
        if label_parts and len(label_parts) == 1:
            lines.append(
                f'<text id="{mid}-label" data-role="category-label" data-entity-id="{eid}" '
                f'x="{category_label_x:.1f}" y="{category_label_y:.1f}" text-anchor="{category_anchor}" font-family="Arial, sans-serif" font-size="{label_font_size}" fill="{label_color}">{label_parts[0]}</text>'
            )
        elif label_parts:
            tspans = "".join(
                f'<tspan x="{category_label_x:.1f}" dy="{20 if idx else 0:.1f}">{part}</tspan>'
                for idx, part in enumerate(label_parts)
            )
            lines.append(
                f'<text id="{mid}-label" data-role="category-label" data-entity-id="{eid}" '
                f'x="{category_label_x:.1f}" y="{category_label_y:.1f}" text-anchor="{category_anchor}" font-family="Arial, sans-serif" font-size="{label_font_size}" fill="{label_color}">{tspans}</text>'
            )
        lines.append("</g>")
    if legend in {"top", "right"} and layout:
        legend_y = 92
        legend_x = W - 420 if legend == "right" else W / 2 + 40
        for i, e in enumerate(layout):
            color = str(colors.get(e["label"]) or colors.get(_entity_id(e)) or _color(i))
            lines.append(
                f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 36}" y2="{legend_y}" stroke="{html.escape(color)}" stroke-width="5"/>'
            )
            lines.append(
                f'<text x="{legend_x + 44}" y="{legend_y + 8}" font-family="Arial, sans-serif" font-size="20" fill="{label_color}">{html.escape(e["label"])}</text>'
            )
            legend_y += 30
    extra_legend = style.get("extra_legend")
    if isinstance(extra_legend, list):
        legend_y = 92
        if legend in {"top", "right"} and layout:
            legend_y = 92 + 30 * len(layout)
        legend_x = W - 420
        for item in extra_legend:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "")
            color = str(item.get("color") or "#ffffff")
            lines.append(
                f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 36}" y2="{legend_y}" stroke="{html.escape(color)}" stroke-width="5"/>'
            )
            lines.append(
                f'<text x="{legend_x + 44}" y="{legend_y + 8}" font-family="Arial, sans-serif" font-size="20" fill="{label_color}">{html.escape(label)}</text>'
            )
            legend_y += 30
    lines.append("</g>")
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _estimate_text_width(text: str, font_size: float) -> float:
    """Rough text width estimate (average glyph ~0.55em) for box sizing."""
    return max(10.0, len(str(text)) * 0.55 * font_size)


def _wrap_text(text: str, max_width: float, font_size: float = 20) -> list[str]:
    """Wrap a label into lines that each fit within ``max_width`` px.

    Word boundaries are preferred; a long unbroken token (e.g. "$140,000") is
    kept whole rather than chopped mid-token.
    """
    text = str(text or "").strip()
    if not text:
        return [""]
    words = text.split()
    if len(words) == 1:
        return [words[0]]
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if not current or _estimate_text_width(trial, font_size) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


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
        eid = _entity_id(e)
        mid = _mark_id(e)
        color = _color(i)
        lines.append(f'<g id="entity-{mid}" data-role="entity" data-entity-id="{eid}" data-label="{html.escape(e["label"])}">')
        lines.append(
            f'<rect id="{mid}-bar" data-role="bar" data-entity-id="{eid}" data-value="{e["value"]:g}" '
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
            f'<rect id="{mid}-value-box" data-role="value-box" data-entity-id="{eid}" '
            f'x="{value_x:.1f}" y="{value_y:.1f}" width="{value_w:.1f}" height="{value_h:.1f}" '
            'fill="#fafafa" stroke="#333333" stroke-width="2"/>'
        )
        lines.append(
            f'<text id="{mid}-value-label" data-role="value-label" data-entity-id="{eid}" '
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
            f'<rect id="{mid}-label-box" data-role="category-box" data-entity-id="{eid}" '
            f'x="{label_x:.1f}" y="{label_y:.1f}" width="{label_w:.1f}" height="{label_h:.1f}" '
            'fill="#fafafa" stroke="#333333" stroke-width="2"/>'
        )
        lines.append(
            f'<text id="{mid}-label" data-role="category-label" data-entity-id="{eid}" '
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
    geo_scale = _geometry_value_scale(layout, horizontal)
    for tv in _nice_ticks(maxv * 1.08):
        if horizontal:
            tx = (geo_scale["anchor"] + tv * geo_scale["scale"]) if geo_scale else LEFT + tv / maxv * (RIGHT - LEFT)
            d.line([(tx, BOTTOM), (tx, BOTTOM + 8)], fill=(100, 100, 100), width=2)
            d.text((tx - 20, BOTTOM + 12), _format_value(tv, unit), fill=(80, 80, 80), font=font_a)
        else:
            ty = (geo_scale["anchor"] - tv * geo_scale["scale"]) if geo_scale else BOTTOM - tv / maxv * (BOTTOM - TOP)
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
    suppress_axis: bool = False,
    colors: dict[str, str] | None = None,
    style: dict[str, Any] | None = None,
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
    geo_scale = _geometry_value_scale(layout, horizontal)
    baseline = float(geo_scale.get("baseline") or 0.0) if geo_scale else 0.0
    style = style or {}
    custom_ticks = style.get("custom_ticks")
    show_values = bool(style.get("show_values", True))
    if isinstance(custom_ticks, list):
        tick_items = [
            (float(t["value"]), str(t.get("label") or ""))
            for t in custom_ticks
            if isinstance(t, dict) and t.get("value") is not None
        ]
    else:
        tick_items = [
            (tv, _format_value(tv, unit))
            for tv in ([] if suppress_axis else _value_ticks(baseline, maxv))
        ]
    for tv, tick_label in tick_items:
        if horizontal:
            tx = (geo_scale["anchor"] + (tv - baseline) * geo_scale["scale"]) if geo_scale else LEFT + tv / maxv * (RIGHT - LEFT)
            d.line([(tx, BOTTOM), (tx, BOTTOM + 8)], fill=(100, 100, 100), width=2)
            d.text((tx - 20, BOTTOM + 12), tick_label, fill=(80, 80, 80), font=font_a)
        else:
            ty = (geo_scale["anchor"] - (tv - baseline) * geo_scale["scale"]) if geo_scale else BOTTOM - tv / maxv * (BOTTOM - TOP)
            d.line([(LEFT - 8, ty), (LEFT, ty)], fill=(100, 100, 100), width=2)
            d.text((LEFT - 70, ty - 12), tick_label, fill=(80, 80, 80), font=font_a)
    extra_legend = style.get("extra_legend")
    if isinstance(extra_legend, list):
        legend_y = 92
        legend_x = W - 420
        for item in extra_legend:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "")
            color = str(item.get("color") or "#ffffff")
            d.line([(legend_x, legend_y), (legend_x + 36, legend_y)], fill=color, width=5)
            d.text((legend_x + 44, legend_y - 8), label, fill=(230, 230, 230), font=font_l)
            legend_y += 30
    for i, e in enumerate(layout):
        fill = (colors or {}).get(str(e.get("label") or "")) or (colors or {}).get(_entity_id(e)) or _color(i)
        d.rectangle([e["x"], e["y"], e["x"] + e["w"], e["y"] + e["h"]], fill=fill, outline=(0, 0, 0))
        text = _format_value(e["value"], unit)
        if horizontal:
            if show_values:
                d.text((e["x"] + e["w"] + 10, e["y"] + e["h"] / 2 - 16), text, fill=(30, 30, 30), font=font_v)
            label_lines = _wrap_text(e["label"], max(40.0, e["x"] - 26))
            line_count = min(3, len(label_lines))
            for li, ln in enumerate(label_lines[:line_count]):
                d.text(
                    (e["x"] - 14, e["y"] + e["h"] / 2 - 8 - (line_count - 1) * 9 + li * 18),
                    ln,
                    fill=(50, 50, 50),
                    font=font_l,
                    anchor="rs",
                )
        else:
            if show_values:
                d.text((e["x"] + e["w"] / 2 - d.textlength(text, font=font_v) / 2, e["y"] - 28), text, fill=(30, 30, 30), font=font_v)
            label_lines = _wrap_text(e["label"], max(40.0, _bar_slot(layout, i) * 0.95))
            line_count = min(3, len(label_lines))
            for li, ln in enumerate(label_lines[:line_count]):
                d.text(
                    (e["x"] + e["w"] / 2 - d.textlength(ln, font=font_l) / 2, BOTTOM + 8 + li * 18),
                    ln,
                    fill=(50, 50, 50),
                    font=font_l,
                )
    img.save(out)
    return out.exists()


def render_data_driven(
    clip_id: str,
    metadata: dict[str, Any],
    out_dir: str | Path,
    geometry: list[dict[str, Any]] | None = None,
    style: dict[str, Any] | None = None,
    geometry_scale: tuple[float, float] | None = None,
    geometry_offset: tuple[float, float] | None = None,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    # The single type-aware vision read (bar spec carries ``series``) is the
    # primary source for title/values/labels/colors/ticks when available.
    if style and isinstance(style.get("series"), list):
        metadata, style = _apply_vision_spec(metadata, style, style)
    entities = entities_from_metadata(metadata)
    orientation = str(metadata.get("orientation") or "vertical")
    layout = _layout_from_geometry(entities, geometry) if geometry else _layout(entities, orientation)
    if not layout:
        layout = _layout(entities, orientation)
    if geometry_scale and layout:
        sx, sy = geometry_scale
        ox, oy = geometry_offset or (0.0, 0.0)
        for e in layout:
            e["x"] = ox + float(e["x"]) * sx
            e["y"] = oy + float(e["y"]) * sy
            e["w"] = float(e["w"]) * sx
            e["h"] = float(e["h"]) * sy
    _enforce_value_geometry(layout, orientation == "horizontal", metadata)
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
    svg_path.write_text(_build_svg(layout, title, unit, orientation, style=style), encoding="utf-8")
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
                    "entity_id": _entity_id(e),
                    "label": e["label"],
                    "value": e["value"],
                    "bbox_px": [round(e["x"]), round(e["y"]), round(e["x"] + e["w"]), round(e["y"] + e["h"])],
                }
                for e in layout
            ],
            "non_entity_components": [],
        },
    )
    preview_success = _render_preview(
        layout,
        title,
        unit,
        preview_path,
        orientation,
        suppress_axis=str((style or {}).get("value_axis") or "") == "none",
        colors=style.get("colors") if isinstance((style or {}).get("colors"), dict) else None,
        style=style,
    )
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


def _norm_label(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _apply_vision_spec(
    metadata: dict[str, Any],
    style: dict[str, Any] | None,
    spec: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prefer the single vision read as the source for title / values / labels /
    colors / background / ticks; CV geometry still controls positions."""
    meta = dict(metadata)
    st = dict(style) if isinstance(style, dict) else {}
    title = str(spec.get("title") or "").strip()
    if title:
        meta["title"] = title
    orient = str(spec.get("orientation") or "").strip().lower()
    if orient in {"vertical", "horizontal"}:
        meta["orientation"] = orient
    series = [s for s in spec.get("series") or [] if isinstance(s, dict) and s.get("label")]
    if series:
        ents = [dict(e) for e in (meta.get("entities") or [])]
        colors = dict(st.get("colors") or {})
        for ent in ents:
            label = str(ent.get("label") or ent.get("name") or "")
            hit = next(
                (
                    s
                    for s in series
                    if _norm_label(str(s.get("label") or "")) == _norm_label(label)
                ),
                None,
            )
            if hit:
                if hit.get("value") is not None:
                    ent["value"] = hit["value"]
                if hit.get("color"):
                    colors[label] = hit["color"]
        if colors:
            st["colors"] = colors
        meta["entities"] = ents
    if spec.get("background"):
        st["background"] = spec["background"]
    if spec.get("value_labels") is False:
        st["show_values"] = False
    ticks = spec.get("ticks")
    if ticks:
        st["custom_ticks"] = list(ticks)
    return meta, st


def render_dynamic_states(
    clip_id: str,
    dynamic: dict[str, Any],
    out_dir: str | Path,
    visible_text: Any = None,
) -> list[dict[str, Any]]:
    """Render one data-driven SVG per recovered state.

    Only rows with an explicit printed state label (state_key/state_label)
    form states; keyless rows are animation frames or duplicates of one
    static chart and are skipped.
    """
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
        key = str(row.get("state_key") or row.get("state_label") or "")
        if not key:
            continue
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
        # Keyless rows (animation frames / duplicates of one static chart)
        # collapse into a single "state" bucket instead of one group per
        # auto state_id.
        key = str(row.get("state_key") or row.get("state_label") or "state")
        groups.setdefault(key, []).append(
            {
                "entity_id": eid,
                "name": str(row.get("entity") or eid),
                "values": [value],
                "metric": str(row.get("metric") or ""),
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

    # Collapse exact duplicate entity+metric marks in the chosen state,
    # preferring the CV-aligned (visual_frame_align) observation.
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        norm = _slug(row["name"])
        if not norm or norm == "unknown":
            continue
        metric_key = _slug(str(row.get("metric") or ""))
        cur = best.get((norm, metric_key))
        if cur is None:
            best[(norm, metric_key)] = row
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
            best[(norm, metric_key)] = row
    series = [
        {
            "name": r["name"],
            "entity_id": r.get("entity_id"),
            "metric": r.get("metric") or "",
            "values": r["values"],
        }
        for r in best.values()
    ]
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
            {
                "label": e["name"],
                "entity_id": e.get("entity_id"),
                "metric": e.get("metric") or "",
                "value": e["values"][0],
                "unit": unit,
            }
            for e in series
        ],
        "visible_text": [],
        "needs_manual_data": False,
        "model_status": "cv_aligned",
        "failure_reason": None,
        "skip_reason": None,
    }
