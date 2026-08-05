from pathlib import Path

from datavideo.qwen_vl import QwenVLClient, _json_from_text
from datavideo.semantic_components import _clip_id_for_out_dir


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


def test_qwen_uses_legacy_model_path_when_no_variant(monkeypatch):
    monkeypatch.delenv("QWEN_MODEL_VARIANT", raising=False)
    monkeypatch.setenv("MODEL_PATH", "/models/qwen-7b")

    client = QwenVLClient({"model": {"env_var": "MODEL_PATH"}})

    assert client.model_path == "/models/qwen-7b"
    assert client.model_path_source == "MODEL_PATH"


def test_qwen_variant_selects_model_specific_env(monkeypatch):
    monkeypatch.setenv("QWEN_MODEL_VARIANT", "qwen3b")
    monkeypatch.setenv("MODEL_PATH", "/models/default")
    monkeypatch.setenv("MODEL_3B_PATH", "/models/qwen-3b")
    monkeypatch.setenv("MODEL_7B_PATH", "/models/qwen-7b")

    client = QwenVLClient(
        {
            "model": {
                "env_var": "MODEL_PATH",
                "variant_env": "QWEN_MODEL_VARIANT",
                "variants": {
                    "qwen3b": {"env_var": "MODEL_3B_PATH"},
                    "qwen7b": {"env_var": "MODEL_7B_PATH"},
                },
            }
        }
    )

    assert client.model_path == "/models/qwen-3b"
    assert client.model_path_source == "MODEL_3B_PATH"


def test_qwen_explicit_config_variant_ignores_unrelated_env_variant(monkeypatch):
    monkeypatch.setenv("QWEN_MODEL_VARIANT", "qwen3b")
    monkeypatch.setenv("MODEL_7B_PATH", "/models/qwen-7b")

    client = QwenVLClient({"model": {"variant": "qwen7b", "env_var": "MODEL_7B_PATH"}})

    assert client.model_cfg["selected_variant"] == "qwen7b"
    assert client.model_path == "/models/qwen-7b"
    assert client.model_path_source == "MODEL_7B_PATH"


def test_qwen_json_repair_handles_code_fence_and_truncation():
    payload = """```json\n{\"chart_type\":\"unknown\",\"needs_review\":false,\"objects\":[],\"entity_groups\":[],\"warnings\":[\"x\"]\n```"""

    result = _json_from_text(payload)

    assert result["chart_type"] == "unknown"
    assert result["warnings"] == ["x"]


def test_semantic_component_clip_id_prefers_parent_clip(tmp_path):
    state_out_dir = tmp_path / "bar_2" / "semantic_states" / "state_001_1990"
    state_out_dir.mkdir(parents=True)

    assert _clip_id_for_out_dir(state_out_dir) == "bar_2"
    assert _clip_id_for_out_dir(tmp_path / "bar_2") == "bar_2"
