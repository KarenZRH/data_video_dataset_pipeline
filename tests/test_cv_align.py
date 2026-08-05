from pathlib import Path

import cv2
import numpy as np

from datavideo.cv_align import (
    _clean_vision_label,
    _ratio_consistency,
    _value_plausibility,
    detect_bars,
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
    assert "0-100" in message

    wrong = {"label": "SSA", "h": 236, "value": 30.0, "value_text": "30%"}
    ok, message = _value_plausibility(wrong, aligned)
    assert not ok
    assert "ratio" in message
