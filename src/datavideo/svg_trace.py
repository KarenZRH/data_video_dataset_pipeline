from __future__ import annotations

from pathlib import Path
from typing import Any

from .schemas import ensure_dir, write_json


def trace_svg(image_path: str | Path, out_dir: str | Path, cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    svg_path = out_dir / "trace.svg"
    preview_path = out_dir / "trace_preview.png"
    report = {
        "tool": "vtracer",
        "params": cfg["vtracer"],
        "input": str(image_path),
        "trace_svg": str(svg_path),
        "trace_preview": str(preview_path),
        "success": False,
        "failure_reason": None,
    }
    try:
        if force or not svg_path.exists():
            import vtracer

            vtracer.convert_image_to_svg_py(
                str(image_path),
                str(svg_path),
                colormode="color",
                hierarchical=cfg["vtracer"].get("hierarchical", "stacked"),
                mode=cfg["vtracer"].get("mode", "spline"),
                filter_speckle=int(cfg["vtracer"].get("filter_speckle", 4)),
                color_precision=int(cfg["vtracer"].get("color_precision", 6)),
                layer_difference=16,
                corner_threshold=60,
                length_threshold=4.0,
                max_iterations=10,
                splice_threshold=45,
                path_precision=int(cfg["vtracer"].get("path_precision", 8)),
            )
        if force or not preview_path.exists():
            import cairosvg

            cairosvg.svg2png(url=str(svg_path), write_to=str(preview_path))
        report["success"] = svg_path.exists() and preview_path.exists()
    except Exception as exc:
        report["failure_reason"] = str(exc)
    write_json(out_dir / "svg_report.json", report)
    return report
