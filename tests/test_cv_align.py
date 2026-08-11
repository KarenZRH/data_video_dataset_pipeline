from pathlib import Path

import cv2
import numpy as np

from datavideo.cv_align import (
    _clean_vision_label,
    _contrast_outline_color,
    _labeled_value_pairs,
    _parse_label_json,
    _ratio_consistency,
    _render_aligned_svg,
    _render_overlay,
    _value_plausibility,
    detect_bars,
    estimate_unlabeled_values,
    locate_text_boxes,
    match_entities,
)


BG = (112, 32, 240)  # BGR dark purple, like the WeChat test clip


def _synthetic_frame(tmp_path: Path) -> Path:
    img = np.full((720, 1280, 3), BG, dtype=np.uint8)
    bars = [
        (172, 236, (253, 208, 54), "36.1%", "Sub-Saharan Africa"),
        (432, 46, (78, 235, 129), "6.9%", "Latin America & Caribbean"),
        (692, 34, (52, 208, 193), "5.1%", "East Asia & Pacific"),
        (954, 6, (84, 168, 255), "1%", "European Union"),
    ]
    baseline = 566
    for x, height, color, value, label in bars:
        cv2.rectangle(img, (x, baseline - height), (x + 158, baseline), color, -1)
        cv2.putText(img, value, (x + 20, baseline - height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(img, label[:12], (x, baseline + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    path = tmp_path / "synthetic.png"
    cv2.imwrite(str(path), img)
    return path


def test_detect_bars_includes_short_bar(tmp_path):
    path = _synthetic_frame(tmp_path)
    boxes = detect_bars(path)
    assert len(boxes) == 4
    assert [b["x"] for b in boxes] == [172, 432, 692, 954]
    heights = [b["h"] for b in boxes]
    assert abs(heights[0] - 236) <= 3
    assert abs(heights[1] - 46) <= 3
    assert abs(heights[2] - 34) <= 3
    assert abs(heights[3] - 6) <= 3


def test_detect_bars_rejects_label_text(tmp_path):
    img = np.full((720, 1280, 3), BG, dtype=np.uint8)
    cv2.rectangle(img, (100, 300), (300, 566), (253, 208, 54), -1)
    cv2.putText(img, "Sub-Saharan Africa", (100, 610), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (253, 208, 54), 2)
    path = tmp_path / "label_only.png"
    cv2.imwrite(str(path), img)
    boxes = detect_bars(path)
    assert len(boxes) == 1
    assert abs(boxes[0]["h"] - 266) <= 2


def test_detect_bars_on_light_background(tmp_path):
    """Light-gray background with colored bars (news-graphic style) must not be
    mistaken for a saturated background that erases the bars."""
    img = np.full((720, 1280, 3), (216, 216, 216), dtype=np.uint8)
    salmon = (95, 112, 249)  # BGR
    cv2.rectangle(img, (272, 260), (606, 645), salmon, -1)
    cv2.rectangle(img, (666, 300), (1006, 645), salmon, -1)
    # value circles above the bars, plus category labels below
    cv2.circle(img, (439, 220), 34, (0, 215, 255), -1)
    cv2.circle(img, (836, 260), 34, (0, 215, 255), -1)
    cv2.putText(img, "cyclists", (320, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 2)
    cv2.putText(img, "drivers", (740, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 2)
    path = tmp_path / "light_bg.png"
    cv2.imwrite(str(path), img)

    boxes = detect_bars(path)
    assert len(boxes) == 2
    assert [b["x"] for b in boxes] == [272, 666]
    assert abs(boxes[0]["h"] - 385) <= 5
    assert abs(boxes[1]["h"] - 345) <= 5


def test_detect_bars_horizontal(tmp_path):
    """Horizontal bars share a left edge and vary in width; they must be
    detected with orientation='horizontal' and sorted top-to-bottom."""
    img = np.full((720, 1280, 3), (216, 216, 216), dtype=np.uint8)
    for cy, w, color in [(180, 380, (30, 144, 255)), (300, 240, (255, 165, 0)), (420, 120, (220, 20, 60))]:
        cv2.rectangle(img, (200, cy - 20), (200 + w, cy + 20), color, -1)
    path = tmp_path / "horizontal.png"
    cv2.imwrite(str(path), img)

    bars = detect_bars(path)
    assert len(bars) == 3
    assert all(b["orientation"] == "horizontal" for b in bars)
    assert [b["x"] for b in bars] == [200, 200, 200]
    assert all(abs(bars[i]["w"] - expected) <= 1 for i, expected in enumerate([380, 240, 120]))
    assert [b["y"] for b in bars] == sorted(b["y"] for b in bars)


def test_match_entities_uses_vision_order_and_creates_frame_entity():
    boxes = [
        {"x": 172, "y": 330, "w": 158, "h": 236},
        {"x": 432, "y": 520, "w": 158, "h": 46},
        {"x": 692, "y": 532, "w": 158, "h": 34},
        {"x": 954, "y": 560, "w": 158, "h": 6},
    ]
    entities = [
        {"id": "east-asia-pacific", "label": "East Asia & Pacific"},
        {"id": "european-union", "label": "European Union"},
        {"id": "latin-america-caribbean", "label": "Latin America & Caribbean"},
    ]
    vision_order = [
        "Sub-Saharan Africa",
        "Latin America & Caribbean",
        "East Asia & Pacific",
        "European Union",
    ]
    aligned, warnings = match_entities(boxes, entities, vision_order)
    assert [a["entity_id"] for a in aligned] == [
        "sub-saharan-africa",
        "latin-america-caribbean",
        "east-asia-pacific",
        "european-union",
    ]
    assert aligned[0]["entity_source"] == "frame"
    assert any("created entity from frame label" in w for w in warnings)


def test_match_entities_alias_matching():
    boxes = [{"x": 1, "y": 300, "w": 100, "h": 100}]
    entities = [{"id": "european-union", "label": "European Union"}]
    aligned, _ = match_entities(boxes, entities, vision_order=["EU"])
    assert aligned[0]["entity_id"] == "european-union"


def test_match_entities_falls_back_to_list_order():
    boxes = [
        {"x": 1, "y": 330, "w": 100, "h": 236},
        {"x": 2, "y": 520, "w": 100, "h": 46},
    ]
    entities = [
        {"id": "ssa", "label": "Sub-Saharan Africa"},
        {"id": "lac", "label": "Latin America & Caribbean"},
    ]
    aligned, _ = match_entities(boxes, entities)
    assert [a["entity_id"] for a in aligned] == ["ssa", "lac"]


def test_ratio_consistency_guards_and_detects_mismatch():
    assert _ratio_consistency([]) == (True, "too few bars to check")
    assert _ratio_consistency([{"label": "a", "h": 100, "value": 36.1}]) == (True, "too few bars to check")

    no_values = [
        {"label": "a", "h": 100, "value": None},
        {"label": "b", "h": 50, "value": None},
    ]
    ok, msg = _ratio_consistency(no_values)
    assert ok and "no numeric values" in msg

    degenerate = [
        {"label": "a", "h": 0, "value": 1.0},
        {"label": "b", "h": 0, "value": 2.0},
    ]
    ok, msg = _ratio_consistency(degenerate)
    assert ok and "degenerate" in msg

    mismatch = [
        {"label": "a", "h": 236, "value": 36.1},
        {"label": "b", "h": 46, "value": 30.0},
    ]
    ok, _ = _ratio_consistency(mismatch)
    assert not ok


def test_clean_vision_label():
    assert _clean_vision_label("1. Sub-Saharan Africa") == "Sub-Saharan Africa"
    assert _clean_vision_label("2) EU") == "EU"
    assert _clean_vision_label("  Latin America & Caribbean  ") == "Latin America & Caribbean"


def test_labeled_value_pairs_supports_dollar_and_comma_labels():
    text = "Less than $20,000: 890, More than $200,000: 1150"
    pairs = _labeled_value_pairs(text)
    assert ("Less than $20,000", "890") in pairs
    assert ("More than $200,000", "1150") in pairs


def test_parse_label_json_keeps_comma_labels_intact():
    text = (
        '```json\n'
        '[{"label": "Less than $20,000"}, {"label": "$40,000"},'
        ' {"label": "More than $200,000"}]\n'
        '```'
    )
    assert _parse_label_json(text) == [
        "Less than $20,000",
        "$40,000",
        "More than $200,000",
    ]


def test_render_aligned_svg_has_value_and_category_boxes(tmp_path):
    aligned = [
        {"x": 172, "y": 330, "w": 158, "h": 236, "entity_id": "ssa", "label": "Sub-Saharan Africa", "value_text": "36.1%"},
        {"x": 953, "y": 560, "w": 157, "h": 6, "entity_id": "eu", "label": "European Union", "value_text": "1%"},
    ]
    out = tmp_path / "aligned.svg"
    assert _render_aligned_svg(aligned, out)
    svg = out.read_text(encoding="utf-8")
    assert 'data-role="value-box"' in svg
    assert 'data-role="category-box"' in svg
    assert "36.1%" in svg
    assert "European Union" in svg


def test_render_overlay_renders_with_boxes(tmp_path):
    img = np.full((720, 1280, 3), BG, dtype=np.uint8)
    frame = tmp_path / "frame.png"
    cv2.imwrite(str(frame), img)
    aligned = [
        {
            "x": 172,
            "y": 330,
            "w": 158,
            "h": 236,
            "entity_id": "ssa",
            "label": "Sub-Saharan Africa",
            "value_text": "36.1%",
        }
    ]
    out = tmp_path / "overlay.png"
    assert _render_overlay(frame, aligned, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_overlay_boxes_original_text_positions(tmp_path):
    img = np.full((720, 1280, 3), BG, dtype=np.uint8)
    frame = tmp_path / "frame.png"
    cv2.imwrite(str(frame), img)
    aligned = [
        {
            "x": 172,
            "y": 330,
            "w": 158,
            "h": 236,
            "entity_id": "ssa",
            "label": "Sub-Saharan Africa",
            "value_text": "36.1%",
        }
    ]
    text_boxes = {
        "ssa": {"value_box": [180, 200, 320, 240], "label_box": [180, 600, 400, 640]}
    }
    out = tmp_path / "overlay.png"
    assert _render_overlay(frame, aligned, out, text_boxes)
    from PIL import Image

    rendered = Image.open(out).convert("RGB")
    # transparent outline (white on the purple background) around the original
    # text positions, and the bar box in red
    assert rendered.getpixel((181, 220)) == (255, 255, 255)
    assert rendered.getpixel((181, 620)) == (255, 255, 255)
    assert rendered.getpixel((173, 400)) == (230, 25, 75)


def test_contrast_outline_color_adapts_to_background():
    dark = np.full((720, 1280, 3), BG, dtype=np.uint8)  # BGR purple
    light = np.full((720, 1280, 3), (216, 216, 216), dtype=np.uint8)  # BGR gray
    assert _contrast_outline_color(dark, [100, 100, 200, 140]) == (255, 255, 255)
    assert _contrast_outline_color(light, [100, 100, 200, 140]) == (20, 20, 20)


def test_locate_text_boxes_cv_fallback(tmp_path):
    img = np.full((720, 1280, 3), (216, 216, 216), dtype=np.uint8)
    cv2.rectangle(img, (272, 260), (606, 645), (95, 112, 249), -1)
    cv2.putText(img, "88%", (420, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
    cv2.putText(img, "cyclists", (330, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), img)
    aligned = [
        {"entity_id": "cyclists", "label": "cyclists", "x": 272, "y": 260, "w": 334, "h": 385}
    ]

    boxes = locate_text_boxes(path, aligned)
    value_box = boxes["cyclists"].get("value_box")
    label_box = boxes["cyclists"].get("label_box")
    assert value_box is not None and value_box[3] <= 270
    assert label_box is not None and label_box[1] >= 645


def test_locate_text_boxes_geometry_selection(tmp_path):
    """The value box must be the text line just above the bar and the label
    box the line just below the baseline, selected purely by geometry."""
    img = np.full((720, 1280, 3), (216, 216, 216), dtype=np.uint8)
    cv2.rectangle(img, (272, 260), (606, 645), (95, 112, 249), -1)
    cv2.putText(img, "88%", (400, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
    cv2.putText(img, "cyclists", (330, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), img)
    aligned = [
        {
            "entity_id": "cyclists",
            "label": "cyclists",
            "x": 272,
            "y": 260,
            "w": 334,
            "h": 385,
            "value_text": "88%",
        }
    ]

    boxes = locate_text_boxes(path, aligned)
    value_box = boxes["cyclists"]["value_box"]
    label_box = boxes["cyclists"]["label_box"]
    assert value_box[3] < 260
    assert label_box is not None and label_box[1] >= 600


def test_locate_text_boxes_horizontal(tmp_path):
    """For horizontal bars the value is at the right end and the label sits
    above the bar."""
    img = np.full((720, 1280, 3), (216, 216, 216), dtype=np.uint8)
    cv2.rectangle(img, (200, 180), (580, 220), (30, 144, 255), -1)
    cv2.putText(img, "A", (210, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
    cv2.putText(img, "380", (600, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), img)
    aligned = [
        {
            "entity_id": "a",
            "label": "A",
            "x": 200,
            "y": 180,
            "w": 380,
            "h": 40,
            "orientation": "horizontal",
            "value_text": "380",
        }
    ]
    boxes = locate_text_boxes(path, aligned)
    value_box = boxes["a"]["value_box"]
    label_box = boxes["a"]["label_box"]
    assert value_box is not None and value_box[0] >= 580
    assert label_box is not None and label_box[3] <= 180


def test_render_aligned_svg_horizontal(tmp_path):
    aligned = [
        {
            "x": 200,
            "y": 180,
            "w": 380,
            "h": 40,
            "entity_id": "a",
            "label": "A",
            "value_text": "380",
            "orientation": "horizontal",
        }
    ]
    out = tmp_path / "aligned.svg"
    assert _render_aligned_svg(aligned, out)
    svg = out.read_text(encoding="utf-8")
    assert 'data-animation-property="width"' in svg
    assert 'data-anchor="left"' in svg
    assert 'data-orientation="horizontal"' in svg


def test_value_plausibility_filters_outliers():
    aligned = [
        {"label": "SSA", "h": 236, "value": 36.1, "value_text": "36.1%"},
        {"label": "LAC", "h": 46, "value": 6.9, "value_text": "6.9%"},
        {"label": "EAP", "h": 34, "value": 5.1, "value_text": "5.1%"},
        {"label": "EU", "h": 6, "value": 1.0, "value_text": "1%"},
    ]
    for item in aligned:
        ok, _ = _value_plausibility(item, aligned)
        assert ok

    bad = {"label": "EU", "h": 6, "value": 248.0, "value_text": "248"}
    ok, message = _value_plausibility(bad, aligned)
    assert not ok
    assert "ratio" in message

    wrong = {"label": "SSA", "h": 236, "value": 30.0, "value_text": "30%"}
    ok, message = _value_plausibility(wrong, aligned)
    assert not ok
    assert "ratio" in message

    # Directly printed (majority-verified) values are trusted even when the
    # measured bar length disagrees slightly with the scale.
    printed = {"label": "SSA", "h": 456, "value": 890.0, "value_text": "890", "value_read_verified": True}
    ok, _ = _value_plausibility(printed, aligned)
    assert ok


def test_estimate_unlabeled_values_linear_scale_vertical():
    aligned = [
        {"label": "A", "h": 200, "w": 20, "value": 100.0, "value_text": "100"},
        {"label": "B", "h": 150, "w": 20, "value": None, "value_text": None},
        {"label": "C", "h": 50, "w": 20, "value": None, "value_text": None},
        {"label": "E", "h": 100, "w": 20, "value": 50.0, "value_text": "50"},
    ]
    count = estimate_unlabeled_values(aligned)
    assert count == 2
    by_label = {a["label"]: a for a in aligned}
    assert by_label["B"]["value_estimated"] is True
    assert by_label["B"]["value_type"] == "estimated"
    assert abs(by_label["B"]["value"] - 75.0) < 1.0
    assert abs(by_label["C"]["value"] - 25.0) < 1.0


def test_estimate_unlabeled_values_linear_scale_horizontal():
    aligned = [
        {"label": "A", "w": 400, "h": 30, "value": 1150.0, "value_text": "1150", "orientation": "horizontal"},
        {"label": "B", "w": 320, "h": 30, "value": None, "value_text": None, "orientation": "horizontal"},
        {"label": "D", "w": 300, "h": 30, "value": 890.0, "value_text": "890", "orientation": "horizontal"},
    ]
    count = estimate_unlabeled_values(aligned)
    assert count == 1
    by_label = {a["label"]: a for a in aligned}
    assert by_label["B"]["value_estimated"] is True
    # linear through (400,1150) and (300,890): slope = 2.6, intercept = 110
    assert abs(by_label["B"]["value"] - (110 + 2.6 * 320)) < 2.0
