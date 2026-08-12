from datavideo.semantic_render import (
    entities_from_metadata,
    _format_value,
    _infer_unit,
    _timestamp_evidenced,
    metadata_from_dynamic,
    render_data_driven,
    resolve_render_title,
)


def _metadata() -> dict:
    return {
        "title": "Illiteracy Rate (2017)",
        "unit": "%",
        "series": [
            {"name": "Sub-Saharan Africa", "values": [36.1]},
            {"name": "European Union", "values": [1]},
        ],
    }


def test_components_svg_has_value_and_category_boxes(tmp_path):
    report = render_data_driven("clip_x", _metadata(), tmp_path)
    svg = (tmp_path / "semantic_components.svg").read_text(encoding="utf-8")
    assert 'data-role="title-box"' in svg
    assert 'data-role="value-box"' in svg
    assert 'data-role="category-box"' in svg
    assert "36.1%" in svg
    assert "Sub-Saharan Africa" in svg
    assert 'stroke="#d62728"' in svg  # red-bordered bars
    assert (tmp_path / "semantic_components_preview.png").exists()
    assert report["semantic_components_svg"] == str(tmp_path / "semantic_components.svg")
    assert report["components_preview_success"] is True


def test_format_value_handles_unit_and_currency_prefix():
    assert _format_value(36.1, "%") == "36.1%"
    assert _format_value(890, None) == "890"
    assert _format_value(300, "$") == "$300"
    assert _format_value(1150, "") == "1150"


def test_render_data_driven_uses_entities_with_unit_strings(tmp_path):
    metadata = {
        "title": "Transportation mode, by share of use",
        "unit": "%",
        "series": [],
        "entities": [
            {"label": "Auto", "value": "75%", "unit": "%"},
            {"label": "Public transit", "value": "5%", "unit": "%"},
        ],
    }

    report = render_data_driven("bar_6", metadata, tmp_path)
    svg = (tmp_path / "semantic.svg").read_text(encoding="utf-8")

    assert report["success"] is True
    assert report["entity_count"] == 2
    assert 'data-role="bar"' in svg
    assert "Auto" in svg
    assert "75%" in svg


def test_render_data_driven_empty_metadata_returns_failure_without_crashing(tmp_path):
    report = render_data_driven("empty", {}, tmp_path)

    assert report["success"] is False
    assert report["failure_reason"] == "no_recoverable_entities"
    assert (tmp_path / "semantic.svg").exists()


def test_infer_unit_never_defaults_to_percent():
    assert _infer_unit([{"unit": "%"}]) == "%"
    assert _infer_unit([{"unit": None}]) == ""
    assert _infer_unit([{"unit": None}], visible_text=["$20,000", "890"]) == ""
    assert _infer_unit([{"unit": None}], visible_text=["48%", "Sub-Saharan Africa"]) == "%"
    assert _infer_unit([{"unit": "$"}]) == "$"


def test_render_dynamic_states_does_not_invent_percent_unit(tmp_path):
    from datavideo.semantic_render import render_dynamic_states

    dynamic = {
        "states": [
            {"entity_id": "a", "entity": "A", "metric": "Score", "value": 890.0, "unit": None, "state_key": "2019", "state_start": 0.0, "source_type": "visual_frame_align"},
            {"entity_id": "b", "entity": "B", "metric": "Score", "value": 1150.0, "unit": None, "state_key": "2019", "state_start": 0.0, "source_type": "visual_frame_align"},
        ]
    }
    reports = render_dynamic_states("clip_x", dynamic, tmp_path, visible_text=[])
    assert reports and reports[0]["success"] is True
    svg = (tmp_path / "semantic_states" / "2019" / "semantic.svg").read_text(encoding="utf-8")
    assert "890%" not in svg
    assert "1150%" not in svg
    assert '>890<' in svg or 'data-value="890"' in svg


def test_timestamp_evidenced_requires_visible_year():
    assert _timestamp_evidenced("2017", ["Illiteracy Rate 2017", "Sub-Saharan Africa"]) is True
    assert _timestamp_evidenced("1990", ["Illiteracy Rate 1990"]) is True
    assert _timestamp_evidenced("2019", ["compliance with traffic laws", "88%", "cyclists", "drivers"]) is False
    assert _timestamp_evidenced("cyclists", ["cyclists", "drivers"]) is False


def test_resolve_render_title_keeps_original_unless_year_conflicts():
    assert resolve_render_title("compliance with traffic laws", "Value") == "compliance with traffic laws"
    assert resolve_render_title("Illiteracy Rate 1990", "Illiteracy Rate (2017)") == "Illiteracy Rate (2017)"
    assert resolve_render_title("Illiteracy Rate 2017", "Illiteracy Rate (2017)") == "Illiteracy Rate 2017"
    assert resolve_render_title("", "Value") == "Value"


def test_metadata_from_dynamic_prefers_cv_aligned_state():
    dynamic = {
        "states": [
            {"entity_id": "ssa", "entity": "Sub-Saharan Africa", "metric": "Illiteracy Rate", "value": 48.0, "unit": "%", "state_key": "1990", "state_start": 0.0, "source_type": "visual", "confidence": 0.8},
            {"entity_id": "lac", "entity": "Latin America & Caribbean", "metric": "Illiteracy Rate", "value": 15.5, "unit": "%", "state_key": "1990", "state_start": 0.0, "source_type": "visual", "confidence": 0.8},
            {"entity_id": "ssa", "entity": "Sub-Saharan Africa", "metric": "Illiteracy Rate", "value": 36.1, "unit": "%", "state_key": "2017", "state_start": 3.75, "source_type": "visual_frame_align", "confidence": 0.85},
            {"entity_id": "lac", "entity": "Latin America & Caribbean", "metric": "Illiteracy Rate", "value": 6.9, "unit": "%", "state_key": "2017", "state_start": 3.75, "source_type": "visual_frame_align", "confidence": 0.85},
        ]
    }
    meta = metadata_from_dynamic(dynamic, visible_text=["Illiteracy Rate 2017", "Sub-Saharan Africa"])
    assert meta is not None
    assert meta["title"] == "Illiteracy Rate (2017)"
    assert [e["label"] for e in meta["entities"]] == ["Sub-Saharan Africa", "Latin America & Caribbean"]
    assert [e["value"] for e in meta["entities"]] == [36.1, 6.9]


def test_metadata_from_dynamic_keeps_static_grouped_bar_metrics():
    dynamic = {
        "chart_type": "bar",
        "states": [
            {"entity_id": "auto", "entity": "Auto", "metric": "National average", "value": 75.0, "unit": "%", "state_key": None, "state_id": "state_001", "source_type": "visual", "confidence": 1.0},
            {"entity_id": "auto", "entity": "Auto", "metric": "Transit-oriented developments", "value": 45.0, "unit": "%", "state_key": None, "state_id": "state_002", "source_type": "visual", "confidence": 1.0},
            {"entity_id": "bike", "entity": "Bike", "metric": "National average", "value": 2.0, "unit": "%", "state_key": None, "state_id": "state_003", "source_type": "visual", "confidence": 1.0},
            {"entity_id": "bike", "entity": "Bike", "metric": "Transit-oriented developments", "value": 5.0, "unit": "%", "state_key": None, "state_id": "state_004", "source_type": "visual", "confidence": 1.0},
            {"entity_id": "public-transit", "entity": "Public transit", "metric": "National average", "value": 5.0, "unit": "%", "state_key": None, "state_id": "state_005", "source_type": "visual", "confidence": 1.0},
            {"entity_id": "public-transit", "entity": "Public transit", "metric": "Transit-oriented developments", "value": 30.0, "unit": "%", "state_key": None, "state_id": "state_006", "source_type": "visual", "confidence": 1.0},
            {"entity_id": "walk", "entity": "Walk", "metric": "National average", "value": 5.0, "unit": "%", "state_key": None, "state_id": "state_007", "source_type": "visual", "confidence": 1.0},
            {"entity_id": "walk", "entity": "Walk", "metric": "Transit-oriented developments", "value": 20.0, "unit": "%", "state_key": None, "state_id": "state_008", "source_type": "visual", "confidence": 1.0},
        ],
    }

    meta = metadata_from_dynamic(dynamic, visible_text=["National average", "Transit-oriented developments"])

    assert meta is not None
    assert len(meta["series"]) == 8
    assert {row["metric"] for row in meta["series"]} == {"National average", "Transit-oriented developments"}
    assert len(entities_from_metadata(meta)) == 8


def test_render_data_driven_draws_static_grouped_bar_from_dynamic_metadata(tmp_path):
    dynamic = {
        "chart_type": "bar",
        "states": [
            {"entity_id": "auto", "entity": "Auto", "metric": "National average", "value": 75.0, "unit": "%", "state_key": None, "state_id": "state_001", "source_type": "visual", "confidence": 1.0},
            {"entity_id": "auto", "entity": "Auto", "metric": "Transit-oriented developments", "value": 45.0, "unit": "%", "state_key": None, "state_id": "state_002", "source_type": "visual", "confidence": 1.0},
            {"entity_id": "bike", "entity": "Bike", "metric": "National average", "value": 2.0, "unit": "%", "state_key": None, "state_id": "state_003", "source_type": "visual", "confidence": 1.0},
            {"entity_id": "bike", "entity": "Bike", "metric": "Transit-oriented developments", "value": 5.0, "unit": "%", "state_key": None, "state_id": "state_004", "source_type": "visual", "confidence": 1.0},
            {"entity_id": "public-transit", "entity": "Public transit", "metric": "National average", "value": 5.0, "unit": "%", "state_key": None, "state_id": "state_005", "source_type": "visual", "confidence": 1.0},
            {"entity_id": "public-transit", "entity": "Public transit", "metric": "Transit-oriented developments", "value": 30.0, "unit": "%", "state_key": None, "state_id": "state_006", "source_type": "visual", "confidence": 1.0},
            {"entity_id": "walk", "entity": "Walk", "metric": "National average", "value": 5.0, "unit": "%", "state_key": None, "state_id": "state_007", "source_type": "visual", "confidence": 1.0},
            {"entity_id": "walk", "entity": "Walk", "metric": "Transit-oriented developments", "value": 20.0, "unit": "%", "state_key": None, "state_id": "state_008", "source_type": "visual", "confidence": 1.0},
        ],
    }
    meta = metadata_from_dynamic(dynamic, visible_text=["Transportation mode, by share of use"])

    report = render_data_driven("bar_6", meta, tmp_path)
    svg = (tmp_path / "semantic.svg").read_text(encoding="utf-8")

    assert report["success"] is True
    assert report["entity_count"] == 8
    assert svg.count('data-role="bar"') == 8
    assert "Auto - National average" in svg
    assert "Auto - Transit-oriented developments" in svg


def test_metadata_from_dynamic_drops_hallucinated_entity_and_unseen_year():
    dynamic = {
        "states": [
            {"entity_id": "cycling", "entity": "cycling", "metric": "drivers", "value": 88.0, "unit": "%", "state_key": "2019", "state_start": 3.75, "source_type": "visual", "confidence": 0.8},
            {"entity_id": "cycling", "entity": "cycling", "metric": "drivers", "value": 85.0, "unit": "%", "state_key": "2019", "state_start": 3.75, "source_type": "visual", "confidence": 0.8},
            {"entity_id": "cyclists", "entity": "cyclists", "metric": "drivers", "value": 88.0, "unit": "%", "state_key": "2019", "state_start": 3.75, "source_type": "visual_frame_align", "confidence": 0.85},
            {"entity_id": "drivers", "entity": "drivers", "metric": "drivers", "value": 85.0, "unit": "%", "state_key": "2019", "state_start": 3.75, "source_type": "visual_frame_align", "confidence": 0.85},
        ]
    }
    meta = metadata_from_dynamic(dynamic, visible_text=["compliance with traffic laws", "88%", "cyclists", "drivers"])
    assert meta is not None
    assert "(2019)" not in meta["title"]
    assert [e["label"] for e in meta["entities"]] == ["cyclists", "drivers"]
    assert [e["value"] for e in meta["entities"]] == [88.0, 85.0]


def test_render_data_driven_horizontal_layout(tmp_path):
    metadata = {
        "title": "Horizontal Test",
        "unit": "km",
        "orientation": "horizontal",
        "series": [
            {"name": "A", "values": [300.0]},
            {"name": "B", "values": [200.0]},
            {"name": "C", "values": [100.0]},
        ],
    }
    report = render_data_driven("clip_h", metadata, tmp_path)
    svg = (tmp_path / "semantic.svg").read_text(encoding="utf-8")
    assert 'data-orientation="horizontal"' in svg
    assert 'data-animation-property="width"' in svg
    assert 'data-anchor="left"' in svg
    assert report["entity_count"] == 3
    assert (tmp_path / "semantic_preview.png").exists()
