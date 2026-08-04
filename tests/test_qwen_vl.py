from datavideo.qwen_vl import QwenVLClient


def test_qwen_semantic_components_unavailable(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    client = QwenVLClient({"model": {"env_var": "MODEL_PATH"}})

    response = client.identify_semantic_components("missing.jpg")

    assert response["model_status"] == "qwen_unavailable"
    assert response["result"]["chart_type"] == "unknown"
    assert response["result"]["entities"] == []
    assert response["failure_reason"]


def test_qwen_quality_review_unavailable(monkeypatch):
    monkeypatch.setenv("DATAVIDEO_SKIP_QWEN", "1")
    client = QwenVLClient({"model": {"env_var": "MODEL_PATH"}})

    response = client.review_quality(["missing.jpg"], "review this")

    assert response["model_status"] == "qwen_unavailable"
    assert response["result"]["needs_review"] is True
    assert response["result"]["issue_codes"] == ["qc_vlm_unavailable"]
