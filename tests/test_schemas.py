from datavideo.schemas import SCENE_STATES, chart_result, object_hash


def test_chart_result_strict_keys():
    row = chart_result(is_chart=True, confidence=0.8, reason="bar chart visible")
    assert set(row) == {
        "is_chart",
        "chart_types",
        "chart_visible",
        "chart_completeness",
        "occlusion",
        "scene_state",
        "confidence",
        "reason",
    }
    assert row["scene_state"] in SCENE_STATES


def test_object_hash_stable():
    assert object_hash({"b": 2, "a": 1}) == object_hash({"a": 1, "b": 2})
