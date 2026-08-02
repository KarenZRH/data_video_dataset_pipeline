from datavideo.qwen_vl import QwenVLClient, _normalize_data_video_result


def test_qwen_unavailable_is_not_positive(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    client = QwenVLClient({"chart_type": "bar", "model": {"env_var": "MODEL_PATH"}})
    response = client.classify_frames(["missing.jpg"])
    assert response["model_status"] == "qwen_unavailable"
    assert response["result"]["is_chart"] is False
    assert response["result"]["scene_state"] == "uncertain"


def test_data_video_negative_reason_overrides_copied_positive_fields():
    result = _normalize_data_video_result(
        {
            "is_data_video_clip_candidate": True,
            "contains_data_marks": True,
            "data_mark_types": ["horizontal_bar"],
            "chart_types": ["bar"],
            "chart_readable": False,
            "chart_completeness": 0.45,
            "scene_state": "chart_animating",
            "animation_cue": "bar_growing_left_to_right",
            "confidence": 0.62,
            "reason": "画面中没有明显的数据可视化元素，因此不符合数据可视化动画的标准。",
        }
    )
    assert result["is_data_video_clip_candidate"] is False
    assert result["contains_data_marks"] is False
    assert result["scene_state"] == "non_chart"
    assert result["sanitized"] is True
