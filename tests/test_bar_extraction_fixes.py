"""Bar-only regression tests for data extraction and SVG label fixes.

Covers:
  * generic metric ("value") no longer appended to category labels;
  * keyless duplicate rows of CV-aligned entities are dropped;
  * horizontal-bar value crops include the right end of the bar.
"""

from __future__ import annotations

import cv2
import numpy as np

from datavideo.cv_align import (
    _NUMBER_TOKEN_RE,
    _assign_table_values,
    _crop_value_region,
    _extract_json_object,
    _looks_like_value_label,
    _segment_bar_plateaus,
    detect_bars,
    match_entities,
    read_chart_table,
)
from datavideo.cv_reconcile import clean_states
from datavideo.semantic_render import entities_from_metadata


def test_entities_from_metadata_skips_generic_metric_in_label():
    metadata = {
        "series": [
            {"name": "McDonald's", "metric": "value", "values": [37000.0]},
            {"name": "Subway", "metric": "locations", "values": [42246.0]},
        ]
    }
    entities = entities_from_metadata(metadata)
    labels = [e["label"] for e in entities]
    assert "McDonald's" in labels
    assert "Subway" in labels
    # Bar category labels are always plain -- never append a metric suffix.
    assert all(" - " not in label for label in labels)


def test_clean_states_drops_keyless_duplicate_of_aligned_entity():
    aligned = [
        {"entity_id": "mcdonald-s", "label": "McDonald's"},
        {"entity_id": "subway", "label": "Subway"},
    ]
    states = [
        {
            "clip_id": "bar_74",
            "state_id": "state_001",
            "state_key": "2019",
            "state_label": "2019",
            "entity_id": "mcdonald-s",
            "entity": "McDonald's",
            "metric": "value",
            "value": 37000,
            "unit": "locations",
            "state_start": 0.25,
            "state_end": 10.0,
            "source_type": "visual_frame_align",
            "confidence": 0.85,
        },
        {
            "clip_id": "bar_74",
            "state_id": "state_004",
            "state_key": "",
            "state_label": "",
            "entity_id": "mcdonald-s",
            "entity": "McDonald's",
            "metric": "value",
            "value": 37000,
            "unit": "locations",
            "state_start": 0.25,
            "state_end": 0.25,
            "source_type": "visual",
            "confidence": 0.8,
        },
    ]
    cleaned = clean_states(states, aligned)
    assert len(cleaned) == 1
    assert cleaned[0]["entity_id"] == "mcdonald-s"
    assert cleaned[0]["source_type"] == "visual_frame_align"


def test_clean_states_drops_numeric_entity_duplicate():
    aligned = [{"entity_id": "country-1", "label": "Country 1"}]
    states = [
        {
            "clip_id": "bar_92",
            "state_id": "state_001",
            "state_key": "2019",
            "state_label": "2019",
            "entity_id": "country-1",
            "entity": "Country 1",
            "metric": "Score",
            "value": 53,
            "unit": "",
            "state_start": 1.0,
            "state_end": 1.0,
            "source_type": "visual",
            "confidence": 0.8,
        },
        {
            "clip_id": "bar_92",
            "state_id": "state_001",
            "state_key": "2019",
            "state_label": "2019",
            "entity_id": "53",
            "entity": "53",
            "metric": "Score",
            "value": 53,
            "unit": "",
            "state_start": 4.5,
            "state_end": 4.5,
            "source_type": "visual",
            "confidence": 0.8,
        },
    ]
    cleaned = clean_states(states, aligned)
    assert len(cleaned) == 1
    assert cleaned[0]["entity_id"] == "country-1"


def test_crop_value_region_horizontal_covers_right_end():
    img = np.full((720, 1280, 3), 255, dtype=np.uint8)
    box = {"x": 100, "y": 300, "w": 500, "h": 40, "orientation": "horizontal"}
    crop = _crop_value_region(img, box)
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    # Crop starts inside the bar tail and extends past the right edge, so a
    # value printed inside or at the end of the bar is included.
    assert crop.shape[1] >= 500 * 0.45 + 60
    # Vertical band is roughly centred on the bar (bar height + small margin).
    assert crop.shape[0] <= box["h"] + 12


def test_number_token_re_keeps_thousands_separators():
    assert _NUMBER_TOKEN_RE.search("43,000").group(0).strip() == "43,000"
    assert _NUMBER_TOKEN_RE.search("36.1%").group(0).strip() == "36.1%"
    assert _NUMBER_TOKEN_RE.search("890").group(0).strip() == "890"


def test_extract_json_object_handles_markdown_fences():
    text = '```json\n{"title": "A", "bars": [{"label": "X", "value": 10}]}\n```'
    obj = _extract_json_object(text)
    assert obj is not None
    assert obj["title"] == "A"
    assert obj["bars"][0]["label"] == "X"


def test_read_chart_table_merges_two_attempts(monkeypatch):
    import datavideo.cv_align as cv_align

    responses = iter(
        [
            (
                '{"title": "GLOBAL LOCATIONS", "unit": "", "bars": ['
                '{"label": "", "value": 43000}, {"label": "McDonald\'s", "value": 37000}]}'
            ),
            (
                '{"title": "GLOBAL LOCATIONS", "unit": "", "bars": ['
                '{"label": "Subway", "value": 43000}, {"label": "McDonald\'s", "value": 37000}]}'
            ),
        ]
    )
    monkeypatch.setattr(cv_align, "_call_vision", lambda *args, **kwargs: next(responses))
    table = read_chart_table("frame.png", {}, orientation="horizontal", attempts=2)
    assert table["title"] == "GLOBAL LOCATIONS"
    assert table["bars"][0]["label"] == "Subway"
    assert table["bars"][0]["value"] == 43000.0
    assert table["bars"][0]["verified"] is True


def test_read_chart_table_disagreeing_value_stays_unverified(monkeypatch):
    import datavideo.cv_align as cv_align

    responses = iter(
        [
            '{"bars": [{"label": "A", "value": 100}]}',
            '{"bars": [{"label": "A", "value": 200}]}',
        ]
    )
    monkeypatch.setattr(cv_align, "_call_vision", lambda *args, **kwargs: next(responses))
    table = read_chart_table("frame.png", {}, orientation="vertical", attempts=2)
    assert table["bars"][0]["verified"] is False
    assert table["bars"][0]["value"] is None


def test_assign_table_values_matches_by_label():
    aligned = [
        {"entity_id": "subway", "label": "Subway"},
        {"entity_id": "mcdonald-s", "label": "McDonald's"},
    ]
    table = {
        "bars": [
            {"label": "Subway", "value": 43000.0, "value_text": "43000", "verified": True},
            {"label": "McDonald's", "value": 37000.0, "value_text": "37000", "verified": True},
        ]
    }
    out = _assign_table_values(aligned, table)
    assert out[0]["value"] == 43000.0 and out[0]["value_read_verified"]
    assert out[1]["value"] == 37000.0 and out[1]["value_read_verified"]


def test_looks_like_value_label_distinguishes_years_from_values():
    assert not _looks_like_value_label("2019")
    assert not _looks_like_value_label("1990")
    assert _looks_like_value_label("43,000")
    assert _looks_like_value_label("36.1%")
    assert _looks_like_value_label("43000")
    assert not _looks_like_value_label("Less than $20,000")
    assert not _looks_like_value_label("5-17")
    assert _looks_like_value_label("$3,000")


def test_match_entities_matches_numeric_year_label():
    boxes = [{"x": 100, "y": 400, "w": 60, "h": 100}]
    entities = [{"id": "2019", "label": "2019"}]
    aligned, _ = match_entities(boxes, entities, ["2019"])
    assert aligned[0]["entity_id"] == "2019"


def test_match_entities_value_label_never_creates_entity():
    boxes = [
        {"x": 50, "y": 62, "w": 386, "h": 36},
        {"x": 50, "y": 160, "w": 300, "h": 36},
    ]
    entities = [
        {"id": "mcdonald-s", "label": "McDonald's"},
        {"id": "subway", "label": "Subway"},
    ]
    # Bar 1's label was misread as its value "43,000"; it must fall back to
    # the still-unmatched recovered entity instead of inventing "43-000".
    aligned, warnings = match_entities(boxes, entities, ["43,000", "McDonald's"])
    assert aligned[0]["entity_id"] == "subway"
    assert not any("43-000" in warning for warning in warnings)


def test_match_entities_creates_frame_entity_for_year_label_when_unmatched():
    boxes = [
        {"x": 100, "y": 400, "w": 60, "h": 100},
        {"x": 220, "y": 400, "w": 60, "h": 100},
    ]
    entities = [{"id": "a", "label": "A"}]
    aligned, warnings = match_entities(boxes, entities, ["2019", "A"])
    assert aligned[1]["entity_id"] == "a"
    assert aligned[0]["label"] == "2019"
    assert aligned[0]["entity_source"] == "frame"
    assert any("2019" in warning for warning in warnings)


def _scan_obs(times, vectors):
    out = []
    for idx, (ts, vector) in enumerate(zip(times, vectors)):
        out.append(
            {
                "timestamp": float(ts),
                "bar_count": len(vector) if vector is not None else 0,
                "vector": vector,
                "path": f"frame_{idx:03d}.jpg",
                "sharpness": 1.0,
            }
        )
    return out


def test_segment_bar_plateaus_static_chart_is_one_state():
    obs = _scan_obs(
        [0.0, 0.5, 1.0, 1.5, 2.0],
        [[0.5, 0.6, 1.0]] * 5,
    )
    plateaus = _segment_bar_plateaus(obs, sample_fps=2.0, min_plateau_seconds=0.8, length_tolerance=0.06)
    assert len(plateaus) == 1
    assert plateaus[0][0]["bar_count"] == 3


def test_segment_bar_plateaus_ignores_transition_frames():
    # Plateau A -> one-frame transition (different values) -> plateau B.
    obs = _scan_obs(
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        [
            [0.5, 0.6, 1.0],
            [0.5, 0.6, 1.0],
            [0.5, 0.6, 1.0],
            [0.7, 0.8, 1.0],  # transition frame (0.5s < min duration)
            [0.2, 0.3, 1.0],
            [0.2, 0.3, 1.0],
            [0.2, 0.3, 1.0],
            [0.2, 0.3, 1.0],
            [0.2, 0.3, 1.0],
        ],
    )
    plateaus = _segment_bar_plateaus(obs, sample_fps=2.0, min_plateau_seconds=0.8, length_tolerance=0.06)
    assert len(plateaus) == 2
    assert plateaus[0][0]["vector"] == [0.5, 0.6, 1.0]
    assert plateaus[1][0]["vector"] == [0.2, 0.3, 1.0]


def test_segment_bar_plateaus_merges_brief_dip():
    # A one-frame dip below min duration between identical shapes is one state.
    obs = _scan_obs(
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        [
            [0.5, 1.0],
            [0.5, 1.0],
            [0.5, 1.0],
            [0.6, 1.0],  # brief dip (0.5s)
            [0.5, 1.0],
            [0.5, 1.0],
            [0.5, 1.0],
        ],
    )
    plateaus = _segment_bar_plateaus(obs, sample_fps=2.0, min_plateau_seconds=0.8, length_tolerance=0.06)
    assert len(plateaus) == 1
    assert plateaus[0][0]["vector"] == [0.5, 1.0]


def test_selection_rank_prefers_regular_full_count_later_frame():
    from datavideo.multichart_assets import _selection_rank

    def mk(ts, count, full, combined=1.0, regularity=1.0):
        return {
            "timestamp": ts,
            "clip_duration": 10.0,
            "cv_bar_count": count,
            "_bar_full_count": full,
            "bar_regularity": regularity,
            "combined_score": combined,
            "score": {
                "target_chart_type_match": True,
                "scene_change_or_title_card": False,
                "structure_complete": True,
                "edge_crop_or_occlusion": False,
                "final_or_most_complete_state": True,
                "completeness": 1.0,
                "state_finality": 1.0,
                "data_marks_readable": True,
            },
        }

    cfg = {"keyframes": {"prefer_late_chart_types": ["bar"]}}
    noisy = mk(1.0, 8, 7, regularity=0.0)  # over-counted / cross-fade early frame
    early_full = mk(1.25, 7, 7)
    late_full = mk(6.75, 7, 7)
    assert _selection_rank(late_full, "bar", cfg) > _selection_rank(noisy, "bar", cfg)
    assert _selection_rank(late_full, "bar", cfg) > _selection_rank(early_full, "bar", cfg)


def test_render_data_driven_uses_cv_geometry_and_style(tmp_path):
    from datavideo.semantic_render import render_data_driven

    metadata = {
        "title": "Test",
        "unit": "%",
        "orientation": "vertical",
        "series": [
            {"name": "A", "values": [10.0]},
            {"name": "B", "values": [20.0]},
        ],
    }
    geometry = [
        {"entity_id": "a", "label": "A", "x": 100, "y": 200, "w": 60, "h": 120},
        {"entity_id": "b", "label": "B", "x": 300, "y": 250, "w": 60, "h": 70},
    ]
    style = {"colors": {"A": "#112233", "B": "#445566"}, "background": "#000000", "rounded_corners": 4}
    report = render_data_driven("bar_x", metadata, tmp_path, geometry=geometry, style=style)
    svg = (tmp_path / "semantic.svg").read_text(encoding="utf-8")
    assert 'x="100.0" y="200.0" width="60.0" height="120.0"' in svg
    assert 'x="300.0" y="250.0" width="60.0" height="70.0"' in svg
    assert 'fill="#112233"' in svg and 'fill="#445566"' in svg
    assert 'fill="#000000"' in svg
    assert 'rx="4.0"' in svg
    assert report["success"] is True


def test_render_data_driven_scales_geometry(tmp_path):
    from datavideo.semantic_render import render_data_driven

    metadata = {
        "title": "T",
        "unit": "",
        "orientation": "horizontal",
        "series": [{"name": "A", "values": [10.0]}],
    }
    geometry = [{"label": "A", "x": 100, "y": 50, "w": 200, "h": 30}]
    render_data_driven("bar_x", metadata, tmp_path, geometry=geometry, geometry_scale=(2.0, 2.0))
    svg = (tmp_path / "semantic.svg").read_text(encoding="utf-8")
    assert 'x="200.0" y="100.0" width="400.0" height="60.0"' in svg


def test_render_data_driven_value_position_left(tmp_path):
    from datavideo.semantic_render import render_data_driven

    metadata = {
        "title": "T",
        "unit": "B",
        "orientation": "horizontal",
        "series": [{"name": "A", "values": [51.6]}],
    }
    style = {"show_values": True, "value_position": "left"}
    render_data_driven("bar_x", metadata, tmp_path, style=style)
    svg = (tmp_path / "semantic.svg").read_text(encoding="utf-8")
    assert 'data-role="value-label"' in svg
    assert 'text-anchor="end"' in svg


def test_match_chart_style_parses_vision_json(monkeypatch):
    import datavideo.semantic_render as semantic_render

    responses = iter(
        [
            '{"colors": {"A": "#ff0000"}, "background": "#000000", "gridlines": true, '
            '"rounded_corners": 4, "value_position": "inside", "legend": "none", "title": "T"}'
        ]
    )
    monkeypatch.setattr(semantic_render, "_call_vision", lambda *args, **kwargs: next(responses))
    style = semantic_render.match_chart_style("frame.png", {})
    assert style["colors"]["A"] == "#ff0000"
    assert style["rounded_corners"] == 4
    assert style["value_position"] == "inside"


def test_analyze_chart_parses_vision_json(monkeypatch):
    import datavideo.cv_align as cv_align

    responses = iter(
        [
            '{"chart_type": "bar", "orientation": "horizontal", "axis_unit": "$", '
            '"bars": [{"label": "A", "value": "51.6B", "x": 0.1, "y": 0.2, "w": 0.4, "h": 0.5}, '
            '{"label": "B", "value": 20, "x": 0.6, "y": 0.2, "w": 0.4, "h": 0.6}], '
            '"style": {"colors": {"A": "#ff0000"}, "background": "#ffffff", "gridlines": true, '
            '"rounded_corners": 0, "show_values": false, "value_position": "right", '
            '"legend": "none", "title": "T"}}'
        ]
    )
    monkeypatch.setattr(cv_align, "_call_vision", lambda *args, **kwargs: next(responses))
    result = cv_align.analyze_chart("frame.png", {})
    assert result["orientation"] == "horizontal"
    assert len(result["bars"]) == 2
    assert result["bars"][0]["label"] == "A" and result["bars"][0]["x"] == 0.1
    assert result["bars"][0]["value"] == 51.6
    assert result["bars"][0]["value_text"] == "51.6B"
    assert result["style"]["colors"]["A"] == "#ff0000"


def test_supplement_bars_from_vision_appends_missing_bars(tmp_path):
    import cv2
    import numpy as np
    from datavideo.cv_align import supplement_bars_from_vision

    img = np.full((200, 400, 3), 255, dtype=np.uint8)
    frame_path = tmp_path / "frame.png"
    cv2.imwrite(str(frame_path), img)
    cv_report = {
        "orientation": "vertical",
        "bars": [
            {"entity_id": "rise", "label": "Rise", "value": 7.0, "x": 10, "y": 100, "w": 40, "h": 100},
            {"entity_id": "fall", "label": "Fall", "value": 42.0, "x": 60, "y": 100, "w": 40, "h": 100},
        ],
    }
    vision_bars = [
        {"label": "Rise", "value": 7.0, "x": 0.1, "y": 0.2, "w": 0.1, "h": 0.3},
        {"label": "Fall", "value": 42.0, "x": 0.3, "y": 0.2, "w": 0.1, "h": 0.3},
        {"label": "Unsure", "value": 22.0, "value_text": "22%", "x": 0.5, "y": 0.2, "w": 0.1, "h": 0.3},
        {"label": None, "value": 43000.0, "value_text": "43,000", "x": 0.7, "y": 0.2, "w": 0.1, "h": 0.3},
    ]
    added = supplement_bars_from_vision(cv_report, vision_bars, frame_path)
    assert added == 2
    labels = [b["label"] for b in cv_report["bars"]]
    assert "Unsure" in labels
    assert "43,000" in labels  # unlabelled bar keeps its printed value text
    uns = [b for b in cv_report["bars"] if b["label"] == "Unsure"][0]
    assert uns["value"] == 22.0
    assert uns["value_text"] == "22%"
    assert uns["x"] == 200 and uns["w"] == 40  # 0.5*400, 0.1*400


def test_detect_bars_respects_preferred_orientation(tmp_path):
    import cv2
    import numpy as np

    img = np.full((360, 640, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (160, 300), (30, 144, 255), -1)
    cv2.rectangle(img, (200, 150), (260, 300), (255, 165, 0), -1)
    cv2.rectangle(img, (300, 200), (360, 300), (220, 20, 60), -1)
    path = tmp_path / "v.png"
    cv2.imwrite(str(path), img)
    assert len(detect_bars(str(path), prefer_orientation="vertical")) >= 3
    assert len(detect_bars(str(path), prefer_orientation="horizontal")) < 3
