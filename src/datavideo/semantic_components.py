from __future__ import annotations

import base64
import html
import json
import re
import struct
import subprocess
from pathlib import Path
from typing import Any

from .model_client import make_model_client
from .schemas import ensure_dir, read_json, write_json


COMPONENTS_JSON = "semantic_components.json"
COMPONENTS_REPORT = "semantic_components_report.json"
COMPONENTS_SVG = "semantic_components.svg"
COMPONENTS_PNG = "semantic_components.png"
RAW_MODEL_JSON = "semantic_components_model_raw.json"


def _png_size(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG image: {path}")
    return struct.unpack(">II", header[16:24])


def _image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "component"


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return default


def _clip_id_for_out_dir(out_dir: Path) -> str:
    try:
        index = out_dir.parts.index("semantic_states")
    except ValueError:
        return out_dir.name
    if index >= 1:
        return out_dir.parts[index - 1]
    return out_dir.name


def _normalize_components(
    raw_result: dict[str, Any],
    image_path: Path,
    out_dir: Path,
    metadata: dict[str, Any],
    *,
    model_status: str,
) -> dict[str, Any]:
    width, height = _png_size(image_path)
    allowed_types = {
        "bar",
        "icon",
        "category_label",
        "value_label",
        "series",
        "axis",
        "annotation",
        "title",
    }
    warnings = [str(warning) for warning in raw_result.get("warnings", [])]
    objects: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    def clean_bbox(value: Any) -> list[int] | None:
        if not isinstance(value, list) or len(value) != 4:
            return None
        try:
            x1, y1, x2, y2 = [int(round(float(item))) for item in value]
        except Exception:
            return None
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    raw_objects = raw_result.get("objects")
    if not isinstance(raw_objects, list):
        raw_objects = raw_result.get("components")
    if not isinstance(raw_objects, list):
        raw_objects = []

    for index, item in enumerate(raw_objects, start=1):
        if not isinstance(item, dict):
            continue
        component_type = str(item.get("type") or item.get("role") or "").strip()
        if component_type not in allowed_types:
            warnings.append(f"dropped object with unsupported type: {item.get('type')}")
            continue
        bbox = clean_bbox(item.get("bbox_px"))
        if bbox is None:
            warnings.append(f"dropped object with invalid bbox: {item.get('id') or component_type}")
            continue

        object_id = _slug(str(item.get("id") or f"{component_type}-{index}"))
        if object_id in used_ids:
            base = object_id
            suffix = 2
            while object_id in used_ids:
                object_id = f"{base}-{suffix}"
                suffix += 1
        used_ids.add(object_id)

        entity_id = item.get("entity_id")
        text = item.get("text")
        animation_axis = item.get("animation_axis")
        anchor = item.get("anchor")
        if component_type == "bar":
            if not animation_axis:
                box_width = bbox[2] - bbox[0]
                box_height = bbox[3] - bbox[1]
                animation_axis = "y" if box_height > box_width * 1.15 else "x"
            if not anchor:
                anchor = "bottom" if animation_axis == "y" else "left"
        elif component_type == "series":
            animation_axis = animation_axis or "path"
            anchor = anchor or "path-start"

        objects.append(
            {
                "id": object_id,
                "entity_id": _slug(str(entity_id)) if entity_id else None,
                "type": component_type,
                "label": str(item.get("label") or item.get("text") or object_id),
                "text": None if text is None else str(text),
                "text_status": str(item.get("text_status") or ("readable" if text else "not_applicable")),
                "bbox_px": bbox,
                "dominant_color": item.get("dominant_color"),
                "confidence": _clamp01(item.get("confidence", 0.0)),
                "reason": str(item.get("reason", "") or ""),
                "animation_axis": animation_axis,
                "anchor": anchor,
            }
        )

    if not objects:
        raise ValueError(f"{model_status} returned no valid semantic components")

    object_ids = {obj["id"] for obj in objects}
    groups: list[dict[str, Any]] = []
    for index, item in enumerate(raw_result.get("entity_groups", []), start=1):
        if not isinstance(item, dict):
            continue
        component_ids = [str(value) for value in item.get("component_ids", []) if str(value) in object_ids]
        if not component_ids:
            continue
        entity_id = _slug(str(item.get("entity_id") or item.get("label") or f"entity-{index}"))
        groups.append(
            {
                "entity_id": entity_id,
                "label": str(item.get("label") or entity_id),
                "component_ids": component_ids,
                "confidence": _clamp01(item.get("confidence", 0.0)),
            }
        )

    grouped_ids = {component_id for group in groups for component_id in group["component_ids"]}
    entity_map: dict[str, list[str]] = {}
    entity_labels: dict[str, str] = {}
    for obj in objects:
        entity_id = obj.get("entity_id")
        if not entity_id or obj["id"] in grouped_ids:
            continue
        entity_map.setdefault(entity_id, []).append(obj["id"])
        entity_labels.setdefault(entity_id, obj.get("label") or entity_id)
    for entity_id, component_ids in entity_map.items():
        groups.append(
            {
                "entity_id": entity_id,
                "label": entity_labels.get(entity_id, entity_id),
                "component_ids": component_ids,
                "confidence": max((obj["confidence"] for obj in objects if obj["id"] in component_ids), default=0.0),
            }
        )

    dangling_ids = []
    for group in raw_result.get("entity_groups", []):
        if not isinstance(group, dict):
            continue
        for component_id in group.get("component_ids", []):
            if str(component_id) not in object_ids:
                dangling_ids.append(str(component_id))
    if dangling_ids:
        warnings.append(f"dropped dangling group component ids: {', '.join(sorted(set(dangling_ids)))}")

    return {
        "clip_id": _clip_id_for_out_dir(out_dir),
        "source_keyframe": str(image_path),
        "image_width": width,
        "image_height": height,
        "annotation_method": f"{model_status}_direct_semantic_components_v1",
        "automation_level": "single_model_direct_semantic_component_generation",
        "uses_chart_metadata": bool(metadata),
        "metadata_title": metadata.get("title"),
        "metadata_chart_type": metadata.get("chart_type"),
        "chart_type": {
            "bar": "horizontal_bar",
        }.get(str(raw_result.get("chart_type") or "unknown"), raw_result.get("chart_type", "unknown")),
        "needs_review": bool(raw_result.get("needs_review", False)),
        "objects": objects,
        "entity_groups": groups,
        "reconciliation_actions": [],
        "warnings": warnings,
    }


def _render_box_svg(
    image_path: Path,
    boxes: list[dict[str, Any]],
    out_path: Path,
    *,
    group_id: str,
    opacity: float = 0.72,
) -> Path:
    width, height = _png_size(image_path)
    href = _image_data_uri(image_path)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<image href="{href}" x="0" y="0" width="{width}" height="{height}" opacity="{opacity}"/>',
        f'<g id="{group_id}" font-family="Arial, sans-serif" font-size="15" font-weight="700">',
    ]
    for box in boxes:
        x1, y1, x2, y2 = box["bbox_px"]
        label = box.get("label") or box.get("text") or box.get("type") or box["id"]
        color = "#22c55e" if box.get("type") == "bar" else "#2563eb"
        if box.get("type") in {"category_label", "value_label", "title"}:
            color = "#a855f7"
        elif box.get("type") == "axis":
            color = "#6b7280"
        lines.append(f'<g id="{html.escape(str(box["id"]), quote=True)}">')
        lines.append(f'<title>{html.escape(str(box.get("type") or ""))}</title>')
        lines.append(
            f'<rect x="{x1}" y="{y1}" width="{x2 - x1}" height="{y2 - y1}" fill="none" stroke="{color}" stroke-width="4"/>'
        )
        lines.append(
            f'<rect x="{x1}" y="{max(0, y1 - 22)}" width="{max(82, len(str(label)) * 8)}" height="20" fill="{color}" fill-opacity="0.92"/>'
        )
        lines.append(f'<text x="{x1 + 4}" y="{max(16, y1 - 6)}" fill="white">{html.escape(str(label))}</text>')
        lines.append("</g>")
    lines.extend(["</g>", "</svg>"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _render_png(svg_path: Path, png_path: Path) -> bool:
    try:
        import cairosvg

        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))
        return png_path.exists()
    except Exception:
        pass
    try:
        subprocess.run(["rsvg-convert", "-o", str(png_path), str(svg_path)], check=True)
        return png_path.exists()
    except Exception:
        return False


def build_semantic_components(
    image_path: str | Path,
    out_dir: str | Path,
    cfg: dict[str, Any] | None = None,
    *,
    client: Any | None = None,
    force: bool = False,
) -> dict[str, Any]:
    image_path = Path(image_path)
    out_dir = ensure_dir(out_dir)
    cfg = cfg or {}
    json_path = out_dir / COMPONENTS_JSON
    report_path = out_dir / COMPONENTS_REPORT
    components_svg = out_dir / COMPONENTS_SVG
    components_png = out_dir / COMPONENTS_PNG
    raw_model_path = out_dir / RAW_MODEL_JSON
    metadata_path = out_dir / "chart_metadata.json"

    report = {
        "tool": "semantic_component_generator",
        "generator": "direct_semantic_components_model_v1",
        "input": str(image_path),
        "chart_metadata": str(metadata_path) if metadata_path.exists() else None,
        "semantic_components": str(json_path),
        "raw_model_response": str(raw_model_path),
        "inspection_svg": str(components_svg),
        "inspection_png": str(components_png),
        "success": False,
        "failure_reason": None,
        "warnings": [],
        "reconciliation_actions": [],
    }

    try:
        if json_path.exists() and not force:
            annotation = read_json(json_path)
        else:
            metadata = read_json(metadata_path) if metadata_path.exists() else {}
            if client is None and "model" not in cfg:
                raise RuntimeError("Model config is required to identify semantic components")
            scorer = client or make_model_client(cfg)
            if not hasattr(scorer, "identify_semantic_components"):
                raise RuntimeError("Configured model client does not implement identify_semantic_components")
            response = scorer.identify_semantic_components(str(image_path), metadata)
            write_json(raw_model_path, {"response": response})
            if response.get("model_status") in {None, "gemini_unavailable", "unavailable"}:
                raise RuntimeError(response.get("failure_reason") or "semantic component identification unavailable")
            result = response.get("result")
            if not isinstance(result, dict) or ("objects" not in result and "components" not in result):
                raise RuntimeError("semantic model must return direct objects/entity_groups output")
            annotation = _normalize_components(
                result,
                image_path,
                out_dir,
                metadata,
                model_status=str(response.get("model_status") or "model"),
            )
            write_json(json_path, annotation)
            report["warnings"].extend(annotation.get("warnings", []))

        _render_box_svg(image_path, annotation["objects"], components_svg, group_id="semantic-components-inspection")
        report["preview_success"] = _render_png(components_svg, components_png)
        report["success"] = json_path.exists() and bool(annotation.get("objects"))
    except Exception as exc:
        report["failure_reason"] = str(exc)

    write_json(report_path, report)
    return report


def main() -> None:
    import argparse

    from .manifest import load_config

    parser = argparse.ArgumentParser(description="Generate semantic component annotations for a keyframe.")
    parser.add_argument("image_path")
    parser.add_argument("out_dir")
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else {}
    print(json.dumps(build_semantic_components(args.image_path, args.out_dir, cfg, force=args.force), indent=2))


if __name__ == "__main__":
    main()
