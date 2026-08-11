from datavideo.animation import reconcile_intent_with_data


def _dynamic_states(*pairs):
    states = []
    for state_id, state_key, state_label, state_start, values in pairs:
        for entity_id, entity, value, source_type in values:
            states.append(
                {
                    "state_id": state_id,
                    "state_key": state_key,
                    "state_label": state_label,
                    "entity_id": entity_id,
                    "entity": entity,
                    "metric": "Illiteracy Rate",
                    "value": value,
                    "unit": "%",
                    "state_start": state_start,
                    "source_type": source_type,
                    "confidence": 0.85,
                }
            )
    return {"states": states}


def test_reconcile_flips_wrong_grow_direction():
    animation = {
        "target_chart_type": "bar",
        "overall_description": "The bars grow taller.",
        "major_actions": [{"action": "bar_grow", "description": "grows", "evidence_timestamps": [0.0, 3.75]}],
        "confidence": 0.95,
        "model_status": "qwen",
    }
    dynamic = _dynamic_states(
        ("state_001", "1990", "1990", 0.0, [("a", "A", 48.0, "visual"), ("b", "B", 18.0, "visual")]),
        ("state_002", "2017", "2017", 3.75, [("a", "A", 36.1, "visual_frame_align"), ("b", "B", 5.1, "visual_frame_align")]),
    )

    result = reconcile_intent_with_data(animation, dynamic)

    assert result["reconciled_with_data"] is True
    assert result["data_direction"] == "decrease"
    assert [action["action"] for action in result["major_actions"]] == ["bar_shrink"]
    assert "下降" in result["overall_description"]
    assert "1990年" in result["overall_description"]
    assert "2017年" in result["overall_description"]


def test_reconcile_handles_mixed_direction():
    animation = {
        "target_chart_type": "bar",
        "overall_description": "something changes",
        "major_actions": [],
        "confidence": 0.5,
        "model_status": "qwen",
    }
    dynamic = _dynamic_states(
        ("state_001", "1990", "1990", 0.0, [("a", "A", 10.0, "visual"), ("b", "B", 10.0, "visual")]),
        ("state_002", "2017", "2017", 1.0, [("a", "A", 20.0, "visual_frame_align"), ("b", "B", 5.0, "visual_frame_align")]),
    )

    result = reconcile_intent_with_data(animation, dynamic)

    assert result["data_direction"] == "mixed"
    assert {action["action"] for action in result["major_actions"]} == {"bar_grow", "bar_shrink"}


def test_reconcile_keeps_report_when_single_state():
    animation = {
        "target_chart_type": "bar",
        "overall_description": "static",
        "major_actions": [],
        "confidence": 0.5,
        "model_status": "qwen",
    }
    dynamic = _dynamic_states(("state_001", "1990", "1990", 0.0, [("a", "A", 10.0, "visual")]))

    assert reconcile_intent_with_data(animation, dynamic) is animation


def test_reconcile_keeps_report_when_no_shared_entities():
    animation = {
        "target_chart_type": "bar",
        "overall_description": "static",
        "major_actions": [],
        "confidence": 0.5,
        "model_status": "qwen",
    }
    dynamic = _dynamic_states(
        ("state_001", "1990", "1990", 0.0, [("a", "A", 10.0, "visual")]),
        ("state_002", "2017", "2017", 1.0, [("c", "C", 20.0, "visual")]),
    )

    assert reconcile_intent_with_data(animation, dynamic) is animation
