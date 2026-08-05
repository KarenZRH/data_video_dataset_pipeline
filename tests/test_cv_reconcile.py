from datavideo.cv_reconcile import reconcile_dynamic_data


def _dynamic():
    return {
        "clip_id": "combined_1",
        "states": [
            {
                "clip_id": "combined_1",
                "state_id": "state_001",
                "state_key": "2017",
                "state_label": "2017",
                "entity_id": "east-asia-pacific",
                "entity": "East Asia & Pacific",
                "metric": "Illiteracy Rate",
                "value": 4.9,
                "unit": "%",
                "state_start": 3.75,
                "state_end": 3.75,
                "source_type": "visual",
                "evidence_frames": [],
                "confidence": 0.8,
                "review_status": "machine",
            },
            {
                "clip_id": "combined_1",
                "state_id": "state_001",
                "state_key": "2017",
                "state_label": "2017",
                "entity_id": "european-union",
                "entity": "European Union",
                "metric": "Illiteracy Rate",
                "value": 1.0,
                "unit": "%",
                "state_start": 3.75,
                "state_end": 3.75,
                "source_type": "visual",
                "evidence_frames": [],
                "confidence": 0.8,
                "review_status": "machine",
            },
            {
                "clip_id": "combined_1",
                "state_id": "state_001",
                "state_key": "2017",
                "state_label": "2017",
                "entity_id": "latin-america-caribbean",
                "entity": "Latin America & Caribbean",
                "metric": "Illiteracy Rate",
                "value": 6.8,
                "unit": "%",
                "state_start": 3.75,
                "state_end": 3.75,
                "source_type": "visual",
                "evidence_frames": [],
                "confidence": 0.8,
                "review_status": "machine",
            },
        ],
    }


def _cv_report():
    return {
        "detected_bar_count": 4,
        "bars": [
            {"x": 172, "y": 330, "h": 236, "entity_id": "sub-saharan-africa", "label": "Sub-Saharan Africa", "value": 36.1, "value_text": "36.1%"},
            {"x": 432, "y": 520, "h": 46, "entity_id": "latin-america-caribbean", "label": "Latin America & Caribbean", "value": 6.9, "value_text": "6.9%"},
            {"x": 692, "y": 532, "h": 34, "entity_id": "east-asia-pacific", "label": "East Asia & Pacific", "value": 5.1, "value_text": "5.1%"},
            {"x": 954, "y": 560, "h": 6, "entity_id": "european-union", "label": "European Union", "value": 1.0, "value_text": "1%"},
        ],
    }


def test_reconcile_adds_missing_entity_and_updates_values(tmp_path):
    result = reconcile_dynamic_data(
        _dynamic(),
        _cv_report(),
        clip_id="combined_1",
        keyframe_timestamp=3.75,
        image_path="keyframes/initial.png",
        out_dir=tmp_path,
    )
    assert result is not None
    states = result["dynamic"]["states"]
    by_id = {row["entity_id"]: row for row in states}
    assert set(by_id) == {"east-asia-pacific", "european-union", "latin-america-caribbean", "sub-saharan-africa"}
    assert by_id["sub-saharan-africa"]["value"] == 36.1
    assert by_id["latin-america-caribbean"]["value"] == 6.9
    assert by_id["east-asia-pacific"]["value"] == 5.1
    assert by_id["european-union"]["value"] == 1.0
    assert len(result["dynamic"]["final_data_table"]) == 4
    assert (tmp_path / "dynamic_data.json").exists()
    assert (tmp_path / "final_data_table.csv").exists()
    assert result["updated_bar_count"] == 4


def test_reconcile_returns_none_when_no_frame_values(tmp_path):
    report = {"bars": [{"x": 1, "entity_id": "east-asia-pacific", "value": None}]}
    assert (
        reconcile_dynamic_data(
            _dynamic(),
            report,
            clip_id="combined_1",
            keyframe_timestamp=3.75,
            image_path="keyframes/initial.png",
            out_dir=tmp_path,
        )
        is None
    )


def test_reconcile_skips_implausible_values(tmp_path):
    report = {
        "bars": [
            {
                "x": 954,
                "y": 560,
                "h": 6,
                "entity_id": "european-union",
                "label": "European Union",
                "value": 248.0,
                "value_text": "248",
                "value_plausible": False,
                "plausibility_message": "outside plausible 0-100 range",
            },
            {
                "x": 172,
                "y": 330,
                "h": 236,
                "entity_id": "sub-saharan-africa",
                "label": "Sub-Saharan Africa",
                "value": 36.1,
                "value_text": "36.1%",
                "value_plausible": True,
            },
        ]
    }
    result = reconcile_dynamic_data(
        _dynamic(),
        report,
        clip_id="combined_1",
        keyframe_timestamp=3.75,
        image_path="keyframes/initial.png",
        out_dir=tmp_path,
    )
    assert result is not None
    by_id = {row["entity_id"]: row for row in result["dynamic"]["states"]}
    assert by_id["european-union"]["value"] == 1.0  # unchanged from recovered table
    assert by_id["sub-saharan-africa"]["value"] == 36.1
    assert result["updated_bar_count"] == 1
    assert result["skipped_bar_count"] == 1
    assert result["skipped_bars"][0]["entity_id"] == "european-union"
