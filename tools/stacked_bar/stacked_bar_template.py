"""Stacked-bar chart template: input JSON -> flat data table + semantic SVG
+ intent.  Designed to match the pipeline's dataset schema (flat segment rows,
scheme A) and to be reusable as the pipeline's stacked renderer later."""

from __future__ import annotations

import csv
import html
import json
import sys
from pathlib import Path


W, H = 1280, 720
LEFT, RIGHT, TOP, BOTTOM = 120.0, 1220.0, 150.0, 600.0


def _format_value(v: float) -> str:
    return f"{v:g}"


def build_stacked_svg(
    title: str,
    bar_labels: list[str],
    series: list[dict],
    values: dict[str, dict[str, float]],
    unit: str = "",
    orders: dict[str, dict[str, int]] | None = None,
    orientation: str = "vertical",
) -> str:
    """values[bar_label][series_name] = segment value.

    ``orders`` (optional) is bar_label -> {series_name: stack position} so the
    segments are stacked in their real bottom-to-top order; otherwise the
    global series ``order`` is used.
    """
    n = len(bar_labels)
    slot = (RIGHT - LEFT) / n
    plot_h = BOTTOM - TOP
    totals = {b: sum(values.get(b, {}).values()) for b in bar_labels}
    max_total = max(totals.values()) or 1.0
    horizontal = orientation == "horizontal"

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" data-role="semantic-chart" data-generator="stacked_bar_template_v1">',
        f'<rect id="scene-background-fill" data-role="background-fill" x="0" y="0" width="{W}" height="{H}" fill="#f6f6f4"/>',
        f'<text id="chart-title" data-role="title" x="{W / 2}" y="70" text-anchor="middle" font-family="Arial, sans-serif" font-size="36" font-weight="700" fill="#222222">{html.escape(title)}</text>',
        '<g id="chart-plot" data-role="plot">',
        f'<line data-role="axis" x1="{LEFT}" y1="{BOTTOM}" x2="{RIGHT}" y2="{BOTTOM}" stroke="#666666" stroke-width="3"/>',
    ]
    # legend (top-right, compact)
    legend_x = W - 420
    legend_y = 92
    for s in series:
        color = s.get("color") or "#888888"
        lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 36}" y2="{legend_y}" stroke="{html.escape(color)}" stroke-width="6"/>')
        lines.append(f'<text x="{legend_x + 44}" y="{legend_y + 8}" font-family="Arial, sans-serif" font-size="20" fill="#333333">{html.escape(str(s["name"]))}</text>')
        legend_y += 30
    # stacked bars
    for idx, b in enumerate(bar_labels):
        bar_order = (orders or {}).get(b, {})
        ordered_series = sorted(
            series,
            key=lambda s: (bar_order.get(s["name"], 10**9), int(s.get("order", 0))),
        )
        if horizontal:
            row_slot = (BOTTOM - TOP) / n
            bar_h = row_slot * 0.72
            cy = TOP + row_slot * (idx + 0.5)
            x_cursor = LEFT
            for s in ordered_series:
                v = float(values.get(b, {}).get(s["name"], 0.0) or 0.0)
                w_px = v / max_total * (RIGHT - LEFT)
                if w_px <= 0:
                    continue
                color = s.get("color") or "#888888"
                lines.append(
                    f'<rect data-role="stack-segment" data-bar="{html.escape(str(b))}" data-series="{html.escape(str(s["name"]))}" '
                    f'x="{x_cursor:.1f}" y="{cy - bar_h / 2:.1f}" width="{w_px:.1f}" height="{bar_h:.1f}" fill="{html.escape(color)}"/>'
                )
                x_cursor += w_px
            lines.append(
                f'<text x="{LEFT - 14}" y="{cy + 7:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="22" fill="#444444">{html.escape(str(b))}</text>'
            )
        else:
            bar_w = slot * 0.72
            cx = LEFT + slot * (idx + 0.5)
            x = cx - bar_w / 2
            y_cursor = BOTTOM
            for s in ordered_series:
                v = float(values.get(b, {}).get(s["name"], 0.0) or 0.0)
                h_px = v / max_total * plot_h
                if h_px <= 0:
                    continue
                y_top = y_cursor - h_px
                color = s.get("color") or "#888888"
                lines.append(
                    f'<rect data-role="stack-segment" data-bar="{html.escape(str(b))}" data-series="{html.escape(str(s["name"]))}" '
                    f'x="{x:.1f}" y="{y_top:.1f}" width="{bar_w:.1f}" height="{h_px:.1f}" fill="{html.escape(color)}"/>'
                )
                y_cursor = y_top
    if not horizontal:
        for idx in (0, n - 1):
            cx = LEFT + slot * (idx + 0.5)
            lines.append(f'<text x="{cx:.1f}" y="{BOTTOM + 32}" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" fill="#444444">{html.escape(str(bar_labels[idx]))}</text>')
    lines.append("</g>")
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def build_intent(title: str, bar_labels: list[str], series: list[dict], values: dict[str, dict[str, float]]) -> dict:
    """Auto-generate an intent from the data: dominant series at start vs end."""
    def dominant(b: str) -> str:
        seg = values.get(b, {})
        if not seg:
            return ""
        return max(seg, key=lambda k: seg.get(k, 0.0) or 0.0)

    start_b, end_b = bar_labels[0], bar_labels[-1]
    if len(bar_labels) == 1:
        seg = values.get(start_b, {})
        ordered = sorted(seg.items(), key=lambda kv: kv[1] or 0.0, reverse=True)
        if ordered:
            total = sum(seg.values()) or 1.0
            top_name, top_val = ordered[0]
            summary = f"{start_b}，{top_name}占主导（约{top_val / total * 100:.0f}%）"
            if len(ordered) > 1:
                summary += f"，其余主要为{ordered[1][0]}"
            summary += "，各色段比例构成整体分布"
        else:
            summary = f"{start_b}，各色段占比分布"
        return {
            "clip_id": "",
            "chart_type": "bar_stacked",
            "animation": "stacked_grow",
            "intent": summary,
            "summary": summary,
        }
    d_start, d_end = dominant(start_b), dominant(end_b)
    if d_start and d_end and d_start != d_end:
        summary = f"从{start_b}到{end_b}，主导介质由{d_start}转变为{d_end}，柱内各色段占比随之变化"
    else:
        summary = f"从{start_b}到{end_b}，各介质占比持续变化"
    return {
        "clip_id": "",
        "chart_type": "bar_stacked",
        "animation": "stacked_grow",
        "intent": summary,
        "summary": summary,
    }


def main(input_path: str, out_dir: str) -> None:
    cfg = json.loads(Path(input_path).read_text(encoding="utf-8"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    title = str(cfg.get("title") or "Stacked Bar Chart")
    series = cfg.get("series") or []
    unit = str(cfg.get("unit") or "")
    clip_id = str(cfg.get("clip_id") or "")
    orientation = str(cfg.get("orientation") or "vertical")
    series_order = {s["name"]: int(s.get("order", i)) for i, s in enumerate(series)}

    # Two accepted input formats:
    #  A) values[bar][series] = segment value (bar height = sum of segments)
    #  B) bars[{label, rel_len, segments:[{name, fraction}]}] (explicit height
    #     + composition, matching the auto-extraction output)
    values: dict[str, dict[str, float]] = {}
    orders: dict[str, dict[str, int]] = {}
    if cfg.get("bars"):
        bars = cfg["bars"]
        max_rel = max((float(b.get("rel_len") or 0.0) for b in bars), default=1.0) or 1.0
        for b in bars:
            label = str(b.get("label") or f"bar-{len(values) + 1}")
            rel = float(b.get("rel_len") or 0.0) / max_rel
            segs = b.get("segments") or []
            total = sum(float(s.get("fraction") or 0.0) for s in segs) or 1.0
            values[label] = {}
            orders[label] = {}
            for pos, s in enumerate(segs):
                name = str(s.get("name") or "")
                values[label][name] = round(rel * (float(s.get("fraction") or 0.0) / total), 4)
                orders[label][name] = pos
        bar_labels = list(values.keys())
    else:
        bar_labels = [str(x) for x in cfg.get("bar_labels") or []]
        values = {str(k): {str(k2): float(v) for k2, v in (v2 or {}).items()} for k, v2 in (cfg.get("values") or {}).items()}
        orders = None

    # data_table.csv (scheme A: flat segment rows)
    table_path = out / "data_table.csv"
    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "bar", "series", "value", "unit", "series_order", "stack_pos", "value_type"])
        for b in bar_labels:
            for s in sorted(series, key=lambda s: int(s.get("order", 0))):
                v = float(values.get(b, {}).get(s["name"], 0.0) or 0.0)
                pos = (orders or {}).get(b, {}).get(s["name"], s.get("order", 0))
                writer.writerow([clip_id, b, s["name"], f"{v:g}", unit, series_order.get(s["name"], 0), pos, "exact"])

    # semantic.svg
    svg = build_stacked_svg(title, bar_labels, series, values, unit, orders=orders, orientation=orientation)
    (out / "semantic.svg").write_text(svg, encoding="utf-8")

    # intent.json
    intent = build_intent(title, bar_labels, series, values)
    intent["clip_id"] = clip_id
    (out / "intent.json").write_text(json.dumps(intent, ensure_ascii=False, indent=2), encoding="utf-8")

    print("OK", out)
    print("bars:", len(bar_labels), "series:", len(series), "rows:", len(bar_labels) * len(series))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
