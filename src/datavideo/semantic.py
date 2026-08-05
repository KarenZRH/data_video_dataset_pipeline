from __future__ import annotations

import base64
import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import ensure_dir, read_json, write_json


SEMANTIC_SVG_FILENAME = "semantic.svg"
SEMANTIC_PREVIEW_FILENAME = "semantic_preview.png"
SEMANTIC_SCENE_FILENAME = "semantic_scene.json"
SEMANTIC_COMPONENTS_FILENAME = "semantic_components.json"


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    width: float
    height: float

    @classmethod
    def from_bbox(cls, bbox: list[int | float]) -> "Box":
        x1, y1, x2, y2 = bbox
        return cls(float(x1), float(y1), float(x2) - float(x1), float(y2) - float(y1))

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    def union(self, other: "Box") -> "Box":
        x1 = min(self.x, other.x)
        y1 = min(self.y, other.y)
        x2 = max(self.x2, other.x2)
        y2 = max(self.y2, other.y2)
        return Box(x1, y1, x2 - x1, y2 - y1)

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.x2, self.y2]


def _slug(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return text or "item"


def _clean_component_id(component_id: str) -> str:
    return component_id.removesuffix("_icon").removesuffix("_label").removesuffix("_bar").removesuffix("_value")


def _label_text(label: str) -> str:
    if ":" in label:
        return label.split(":", 1)[1].strip()
    return label.strip()


def _dominant_fill(component: dict[str, Any], default: str = "#999999") -> str:
    candidate_color = component.get("dominant_color")
    if isinstance(candidate_color, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", candidate_color):
        return candidate_color
    palette = {
        "car_bar": "#ff8c1a",
        "plane_bar": "#79c7e8",
        "spaceship_bar": "#f04b23",
        "sub_saharan_bar": "#ffd52f",
        "latin_bar": "#48df83",
        "east_bar": "#3fd0c3",
        "europe_bar": "#4ed9f2",
        "demand_curve": "#62a7ee",
        "supply_curve": "#303030",
        "chart_axes": "#bfbfbf",
        "x_axis": "#c9c9cf",
        "beam_baseline": "#a65f34",
    }
    return palette.get(component["id"], default)


def _role(component_type: str) -> str:
    return {
        "category_label": "label",
        "value_label": "value-label",
        "series": "line-series",
        "axis": "axis",
        "annotation": "annotation",
        "title": "title",
        "icon": "icon",
        "bar": "bar",
    }.get(component_type, component_type)


def _bar_animation(component: dict[str, Any]) -> dict[str, str]:
    box = Box.from_bbox(component["bbox_px"])
    if component.get("clip_id") == "bar_1":
        return {
            "data-animation-property": "width",
            "data-anchor": "left",
            "data-animation-axis": "x",
        }
    if component.get("clip_id") == "bar_2":
        return {
            "data-animation-property": "height",
            "data-anchor": "bottom",
            "data-animation-axis": "y",
        }
    if box.height > box.width * 1.25:
        return {
            "data-animation-property": "height",
            "data-anchor": "bottom",
            "data-animation-axis": "y",
        }
    return {
        "data-animation-property": "width",
        "data-anchor": "left",
        "data-animation-axis": "x",
    }


def _line_animation(component: dict[str, Any]) -> dict[str, str]:
    return {
        "data-animation-property": "stroke-dashoffset",
        "data-anchor": "path-start",
        "data-animation-axis": "path",
    }


def _attrs(attrs: dict[str, Any]) -> str:
    pairs = []
    for key, value in attrs.items():
        if value is None:
            continue
        pairs.append(f'{key}="{html.escape(str(value), quote=True)}"')
    return " ".join(pairs)


def _image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _component_map(components: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {component["id"]: component for component in components}


def _entity_specs(
    clip_id: str,
    components: list[dict[str, Any]],
    qwen_entity_groups: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    by_id = _component_map(components)
    qwen_specs = []
    for group in qwen_entity_groups or []:
        if not isinstance(group, dict):
            continue
        component_ids = group.get("component_ids") if isinstance(group.get("component_ids"), list) else []
        parts = [str(component_id) for component_id in component_ids if str(component_id) in by_id]
        if parts:
            raw_entity_id = str(group.get("entity_id") or group.get("label") or parts[0])
            if raw_entity_id.lower().replace("_", "-").startswith("entity-group-"):
                raw_entity_id = str(group.get("label") or raw_entity_id)
            qwen_specs.append(
                {
                    "entity_id": _slug(raw_entity_id),
                    "label": str(group.get("label") or group.get("entity_id") or parts[0]),
                    "parts": parts,
                }
            )
    if qwen_specs:
        return qwen_specs

    entity_map: dict[str, list[str]] = {}
    entity_labels: dict[str, str] = {}
    for component in components:
        entity_id = str(component.get("entity_id") or "").strip()
        if not entity_id:
            continue
        entity_map.setdefault(entity_id, []).append(component["id"])
        entity_labels.setdefault(entity_id, str(component.get("label") or entity_id))
    if entity_map:
        return [
            {
                "entity_id": entity_id,
                "label": entity_labels.get(entity_id, entity_id),
                "parts": parts,
            }
            for entity_id, parts in entity_map.items()
        ]

    if clip_id == "bar_1":
        return [
            {"entity_id": "car", "label": "CAR", "parts": ["car_icon", "car_label", "car_bar"]},
            {"entity_id": "plane", "label": "BOING 747", "parts": ["plane_icon", "plane_label", "plane_bar"]},
            {"entity_id": "spaceship", "label": "SPACESHIP", "parts": ["spaceship_icon", "spaceship_label", "spaceship_bar"]},
        ]
    if clip_id == "bar_2":
        bars = sorted((component for component in components if component.get("type") == "bar"), key=lambda item: item["bbox_px"][0])
        if bars:
            ordered_entities = [
                ("sub-saharan-africa", "Sub-Saharan Africa"),
                ("latin-america-caribbean", "Latin America & Caribbean"),
                ("east-asia-pacific", "East Asia & Pacific"),
                ("european-union", "European Union"),
            ]
            return [
                {
                    "entity_id": entity_id,
                    "label": label,
                    "parts": [bar["id"]],
                }
                for (entity_id, label), bar in zip(ordered_entities, bars, strict=False)
            ]
        return [
            {
                "entity_id": "sub-saharan-africa",
                "label": "Sub-Saharan Africa",
                "parts": ["sub_saharan_value", "sub_saharan_bar", "sub_saharan_label"],
            },
            {
                "entity_id": "latin-america-caribbean",
                "label": "Latin America & Caribbean",
                "parts": ["latin_value", "latin_bar", "latin_label"],
            },
            {
                "entity_id": "east-asia-pacific",
                "label": "East Asia & Pacific",
                "parts": ["east_value", "east_bar", "east_label"],
            },
            {
                "entity_id": "european-union",
                "label": "European Union",
                "parts": ["europe_value", "europe_bar", "europe_label"],
            },
        ]
    if clip_id == "line_2":
        return [
            {"entity_id": "supply", "label": "SUPPLY", "parts": ["supply_curve", "supply_label"]},
            {"entity_id": "demand", "label": "DEMAND", "parts": ["demand_curve", "demand_label"]},
        ]
    if clip_id == "line_1" and "price_series" in by_id:
        return [{"entity_id": "ikea-chair-price", "label": "IKEA chair price", "parts": ["price_series"]}]

    entities: list[dict[str, Any]] = []
    bars = [c for c in components if c.get("type") == "bar"]
    for bar in bars:
        stem = _clean_component_id(bar["id"])
        parts = [cid for cid in [f"{stem}_icon", f"{stem}_label", bar["id"], f"{stem}_value"] if cid in by_id]
        entities.append({"entity_id": _slug(stem), "label": stem.replace("_", " ").title(), "parts": parts})
    return entities


def _group_box(part_ids: list[str], by_id: dict[str, dict[str, Any]]) -> Box:
    boxes = [Box.from_bbox(by_id[part_id]["bbox_px"]) for part_id in part_ids if part_id in by_id]
    box = boxes[0]
    for other in boxes[1:]:
        box = box.union(other)
    return box


def _render_crop(component: dict[str, Any], image_href: str) -> str:
    box = Box.from_bbox(component["bbox_px"])
    clip_id = f"clip-{component['id']}"
    return "\n".join(
        [
            f'<clipPath id="{clip_id}"><rect x="{box.x:g}" y="{box.y:g}" width="{box.width:g}" height="{box.height:g}"/></clipPath>',
            (
                f'<image id="{component["id"]}-raster" data-role="raster-source" href="{image_href}" '
                f'x="0" y="0" width="{component["image_width"]}" height="{component["image_height"]}" '
                f'clip-path="url(#{clip_id})"/>'
            ),
        ]
    )


def _render_text(component: dict[str, Any], role: str, image_href: str) -> str:
    box = Box.from_bbox(component["bbox_px"])
    text = _label_text(component.get("label", ""))
    font_size = max(14, min(36, box.height * 0.58))
    attrs = {
        "id": component["id"],
        "data-role": role,
        "data-source-component-id": component["id"],
        "data-label": text,
        "data-bbox": ",".join(f"{value:g}" for value in box.to_list()),
    }
    text_attrs = {
        "id": f"{component['id']}-text",
        "data-role": "text-content",
        "x": f"{box.x:g}",
        "y": f"{box.y + box.height * 0.72:g}",
        "font-family": "Arial, sans-serif",
        "font-size": f"{font_size:g}",
        "font-weight": "700",
        "visibility": "hidden",
    }
    return f"<g {_attrs(attrs)}>\n{_render_crop(component, image_href)}\n<text {_attrs(text_attrs)}>{html.escape(text)}</text>\n</g>"


def _render_mark(component: dict[str, Any]) -> str:
    box = Box.from_bbox(component["bbox_px"])
    role = _role(component["type"])
    attrs = {
        "id": component["id"],
        "data-role": role,
        "data-source-component-id": component["id"],
        "data-label": _label_text(component.get("label", "")),
        "x": f"{box.x:g}",
        "y": f"{box.y:g}",
        "width": f"{box.width:g}",
        "height": f"{box.height:g}",
        "fill": _dominant_fill(component),
    }
    if component["type"] == "bar":
        attrs.update(_bar_animation(component))
    return f"<rect {_attrs(attrs)}/>"


def _render_component(component: dict[str, Any], image_href: str) -> str:
    component_type = component["type"]
    role = _role(component_type)
    if component_type == "bar":
        return _render_mark(component)
    if component_type == "series":
        box = Box.from_bbox(component["bbox_px"])
        attrs = {
            "id": component["id"],
            "data-role": role,
            "data-source-component-id": component["id"],
            "data-label": _label_text(component.get("label", "")),
            "data-bbox": ",".join(f"{value:g}" for value in box.to_list()),
        }
        attrs.update(_line_animation(component))
        proxy_attrs = {
            "id": f"{component['id']}-animation-proxy",
            "data-role": "animation-proxy",
            "x": f"{box.x:g}",
            "y": f"{box.y:g}",
            "width": f"{box.width:g}",
            "height": f"{box.height:g}",
            "fill": "none",
            "stroke": _dominant_fill(component),
            "stroke-width": "2",
            "opacity": "0",
        }
        return f"<g {_attrs(attrs)}>\n{_render_crop(component, image_href)}\n<rect {_attrs(proxy_attrs)}/>\n</g>"
    if component_type in {"category_label", "value_label", "title"}:
        return _render_text(component, role, image_href)

    attrs = {
        "id": component["id"],
        "data-role": role,
        "data-source-component-id": component["id"],
        "data-label": _label_text(component.get("label", "")),
    }
    return f"<g {_attrs(attrs)}>\n{_render_crop(component, image_href)}\n</g>"


def _render_debug_boxes(components: list[dict[str, Any]]) -> str:
    lines = ['<g id="semantic-debug-layer" data-role="debug-layer" visibility="hidden">']
    for component in components:
        box = Box.from_bbox(component["bbox_px"])
        attrs = {
            "x": f"{box.x:g}",
            "y": f"{box.y:g}",
            "width": f"{box.width:g}",
            "height": f"{box.height:g}",
            "fill": "none",
            "stroke": "#111827",
            "stroke-width": "2",
            "stroke-dasharray": "6 5",
        }
        lines.append(f"<rect {_attrs(attrs)}/>")
    lines.append("</g>")
    return "\n".join(lines)


def _build_scene(annotation: dict[str, Any], image_path: Path) -> tuple[str, dict[str, Any]]:
    clip_id = annotation.get("clip_id") or image_path.parent.parent.name
    width = int(annotation.get("image_width") or 1280)
    height = int(annotation.get("image_height") or 720)
    components = [
        dict(component, image_width=width, image_height=height, clip_id=clip_id)
        for component in annotation.get("objects", [])
    ]
    by_id = _component_map(components)
    image_href = _image_data_uri(image_path)
    entity_specs = _entity_specs(
        clip_id,
        components,
        annotation.get("entity_groups") or annotation.get("qwen_entity_groups"),
    )
    used_ids = {part_id for entity in entity_specs for part_id in entity["parts"]}
    background = "#ffffff"

    scene = {
        "clip_id": clip_id,
        "source_keyframe": str(image_path),
        "image_width": width,
        "image_height": height,
        "annotation_source": SEMANTIC_COMPONENTS_FILENAME,
        "generator": "datavideo.semantic_scene_v1",
        "contains_data_values": False,
        "entities": [],
        "non_entity_components": [],
    }

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" data-role="semantic-chart" '
            f'data-generator="datavideo.semantic_scene_v1" data-source-keyframe="{html.escape(str(image_path), quote=True)}">'
        ),
        f'<rect id="scene-background-fill" data-role="background-fill" x="0" y="0" width="{width}" height="{height}" fill="{background}"/>',
        (
            f'<image id="source-frame-background" data-role="background" href="{image_href}" '
            f'x="0" y="0" width="{width}" height="{height}"/>'
        ),
    ]

    for entity in entity_specs:
        parts = [part_id for part_id in entity["parts"] if part_id in by_id]
        if not parts:
            continue
        box = _group_box(parts, by_id)
        entity_id = entity["entity_id"]
        entity_attrs = {
            "id": f"entity-{entity_id}",
            "data-role": "entity",
            "data-entity-id": entity_id,
            "data-label": entity["label"],
            "data-bbox": ",".join(f"{value:g}" for value in box.to_list()),
        }
        lines.append(f"<g {_attrs(entity_attrs)}>")
        for part_id in parts:
            component = by_id[part_id]
            rendered = _render_component(component, image_href)
            if component["type"] == "bar":
                rendered = rendered.replace(f'id="{part_id}"', f'id="{entity_id}-bar"', 1)
                rendered = rendered.replace(f'data-source-component-id="{part_id}"', f'data-source-component-id="{part_id}" data-entity-id="{entity_id}"', 1)
            elif component["type"] in {"category_label", "value_label"}:
                suffix = "value-label" if component["type"] == "value_label" else "label"
                rendered = rendered.replace(f'id="{part_id}"', f'id="{entity_id}-{suffix}"', 1)
                rendered = rendered.replace(f'data-source-component-id="{part_id}"', f'data-source-component-id="{part_id}" data-entity-id="{entity_id}"', 1)
            elif component["type"] == "icon":
                rendered = rendered.replace(f'id="{part_id}"', f'id="{entity_id}-icon"', 1)
                rendered = rendered.replace(f'data-source-component-id="{part_id}"', f'data-source-component-id="{part_id}" data-entity-id="{entity_id}"', 1)
            elif component["type"] == "series":
                rendered = rendered.replace(f'id="{part_id}"', f'id="{entity_id}-series"', 1)
                rendered = rendered.replace(f'data-source-component-id="{part_id}"', f'data-source-component-id="{part_id}" data-entity-id="{entity_id}"', 1)
            lines.append(rendered)
        lines.append("</g>")
        scene["entities"].append(
            {
                "entity_id": entity_id,
                "label": entity["label"],
                "bbox_px": box.to_list(),
                "parts": parts,
            }
        )

    for component in components:
        if component["id"] in used_ids:
            continue
        lines.append(_render_component(component, image_href))
        scene["non_entity_components"].append(
            {
                "component_id": component["id"],
                "role": _role(component["type"]),
                "label": _label_text(component.get("label", "")),
                "bbox_px": Box.from_bbox(component["bbox_px"]).to_list(),
            }
        )

    lines.append(_render_debug_boxes(components))
    lines.append("</svg>")
    return "\n".join(lines) + "\n", scene


def _render_preview(svg_path: Path, preview_path: Path) -> tuple[bool, str | None]:
    try:
        import cairosvg

        cairosvg.svg2png(url=str(svg_path), write_to=str(preview_path))
        return preview_path.exists(), None
    except Exception as exc:
        pass

    try:
        import subprocess

        subprocess.run(["rsvg-convert", "-o", str(preview_path), str(svg_path)], check=True)
        return preview_path.exists(), None
    except Exception as exc:
        return False, str(exc)


def build_semantic_svg(
    image_path: str | Path,
    out_dir: str | Path,
    cfg: dict[str, Any],
    force: bool = False,
    rebuild_components: bool | None = None,
) -> dict[str, Any]:
    """Build an animation-ready semantic scene graph from component annotations."""
    image_path = Path(image_path)
    out_dir = ensure_dir(out_dir)
    svg_path = out_dir / SEMANTIC_SVG_FILENAME
    preview_path = out_dir / SEMANTIC_PREVIEW_FILENAME
    scene_path = out_dir / SEMANTIC_SCENE_FILENAME
    components_path = out_dir / SEMANTIC_COMPONENTS_FILENAME
    report = {
        "tool": "semantic_scene",
        "generator": "datavideo.semantic_scene_v1",
        "input": str(image_path),
        "annotation": str(components_path),
        "semantic_svg": str(svg_path),
        "semantic_scene": str(scene_path),
        "semantic_preview": str(preview_path),
        "success": False,
        "failure_reason": None,
        "preview_success": False,
        "preview_failure_reason": None,
    }

    try:
        if rebuild_components is None:
            rebuild_components = force
        if rebuild_components or not components_path.exists():
            from .semantic_components import build_semantic_components

            components_report = build_semantic_components(image_path, out_dir, cfg, force=rebuild_components)
            if not components_report.get("success"):
                raise RuntimeError(components_report.get("failure_reason") or "semantic component generation failed")
        annotation = read_json(components_path)
        if force or not svg_path.exists() or not scene_path.exists():
            svg_text, scene = _build_scene(annotation, image_path)
            svg_path.write_text(svg_text, encoding="utf-8")
            write_json(scene_path, scene)
        preview_success, preview_error = _render_preview(svg_path, preview_path)
        report["preview_success"] = preview_success
        report["preview_failure_reason"] = preview_error
        report["success"] = svg_path.exists() and scene_path.exists()
    except Exception as exc:
        report["failure_reason"] = str(exc)
    write_json(out_dir / "svg_report.json", report)
    return report
