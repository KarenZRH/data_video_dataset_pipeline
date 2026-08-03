import xml.etree.ElementTree as ET
from pathlib import Path

from datavideo.semantic import build_semantic_svg
from datavideo.semantic_components import build_semantic_components
from datavideo.svg_trace import trace_svg


def _write_minimal_png(path: Path) -> None:
    path.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
            "0000000c4944415408d763f8ffff3f0005fe02fea73581e20000000049454e44ae426082"
        )
    )


def test_semantic_svg_builds_role_based_scene_graph(tmp_path):
    image = tmp_path / "initial.png"
    _write_minimal_png(image)
    (tmp_path / "semantic_components.json").write_text(
        """
{
  "clip_id": "bar_1",
  "image_width": 1280,
  "image_height": 720,
  "objects": [
    {"id": "car_icon", "type": "icon", "label": "car icon", "bbox_px": [10, 20, 30, 40]},
    {"id": "car_label", "type": "category_label", "label": "label: CAR", "bbox_px": [40, 20, 80, 40]},
    {"id": "car_bar", "type": "bar", "label": "bar: car", "bbox_px": [40, 50, 140, 80]},
    {"id": "plane_icon", "type": "icon", "label": "plane icon", "bbox_px": [10, 90, 30, 110]},
    {"id": "plane_label", "type": "category_label", "label": "label: BOING 747", "bbox_px": [40, 90, 120, 110]},
    {"id": "plane_bar", "type": "bar", "label": "bar: plane", "bbox_px": [40, 120, 80, 150]},
    {"id": "spaceship_icon", "type": "icon", "label": "spaceship icon", "bbox_px": [10, 160, 30, 180]},
    {"id": "spaceship_label", "type": "category_label", "label": "label: SPACESHIP", "bbox_px": [40, 160, 130, 180]},
    {"id": "spaceship_bar", "type": "bar", "label": "bar: spaceship", "bbox_px": [40, 190, 70, 220]}
  ]
}
""",
        encoding="utf-8",
    )

    report = build_semantic_svg(image, tmp_path, {}, force=False)

    assert report["success"] is True
    assert Path(report["semantic_svg"]).name == "semantic.svg"
    assert Path(report["semantic_scene"]).name == "semantic_scene.json"
    root = ET.parse(tmp_path / "semantic.svg").getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}
    background = root.find(".//*[@id='source-frame-background']", ns)
    car = root.find(".//*[@id='entity-car']", ns)
    car_bar = root.find(".//*[@id='car-bar']", ns)
    assert background is not None
    assert background.attrib["data-role"] == "background"
    assert car is not None
    assert car.attrib["data-role"] == "entity"
    assert car_bar is not None
    assert car_bar.attrib["data-role"] == "bar"
    assert car_bar.attrib["data-animation-property"] == "width"
    assert car_bar.attrib["data-anchor"] == "left"

    compatibility_report = trace_svg(image, tmp_path, {}, force=False)
    assert compatibility_report["semantic_svg"] == report["semantic_svg"]


def test_semantic_components_accept_direct_model_output(tmp_path, monkeypatch):
    image = tmp_path / "initial.png"
    _write_minimal_png(image)

    class FakeDirectClient:
        def identify_semantic_components(self, _image_path, _metadata=None):
            return {
                "model_status": "gemini",
                "failure_reason": None,
                "raw_response": "{}",
                "result": {
                    "chart_type": "horizontal_bar",
                    "needs_review": False,
                    "objects": [
                        {
                            "id": "car-icon",
                            "type": "icon",
                            "entity_id": "car",
                            "label": "CAR",
                            "text": None,
                            "text_status": "not_applicable",
                            "bbox_px": [0, 0, 1, 1],
                            "confidence": 0.9,
                            "reason": "visible car icon",
                        },
                        {
                            "id": "car-label",
                            "type": "category_label",
                            "entity_id": "car",
                            "label": "CAR",
                            "text": "CAR",
                            "text_status": "readable",
                            "bbox_px": [0, 0, 1, 1],
                            "confidence": 0.9,
                            "reason": "visible label",
                        },
                        {
                            "id": "car-bar",
                            "type": "bar",
                            "entity_id": "car",
                            "label": "CAR",
                            "text": None,
                            "text_status": "not_applicable",
                            "bbox_px": [0, 0, 1, 1],
                            "confidence": 0.9,
                            "reason": "visible bar",
                            "animation_axis": "x",
                            "anchor": "left",
                        },
                    ],
                    "entity_groups": [
                        {
                            "entity_id": "car",
                            "label": "CAR",
                            "component_ids": ["car-icon", "car-label", "car-bar"],
                            "confidence": 0.9,
                        }
                    ],
                    "warnings": [],
                },
            }

    monkeypatch.setattr("datavideo.semantic_components._render_png", lambda _svg, png: png.write_bytes(b"png") or True)

    report = build_semantic_components(image, tmp_path, {}, client=FakeDirectClient(), force=True)

    assert report["success"] is True
    annotation = (tmp_path / "semantic_components.json").read_text(encoding="utf-8")
    assert "gemini_direct_semantic_components_v1" in annotation
    assert not (tmp_path / "semantic_candidates.json").exists()
    root = ET.parse(tmp_path / "semantic_components.svg").getroot()
    assert root.find(".//*[@id='car-bar']") is not None
