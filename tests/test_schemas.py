from datavideo.manifest import load_config
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


def test_load_config_extends_deep_merges(tmp_path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "\n".join(
            [
                "sample_id: base",
                "model:",
                "  env_var: MODEL_PATH",
                "  prompt_version: base_prompt",
                "clip_data:",
                "  max_frames: 10",
                "  max_new_tokens: 4096",
            ]
        ),
        encoding="utf-8",
    )
    child.write_text(
        "\n".join(
            [
                "extends: base.yaml",
                "sample_id: child",
                "model:",
                "  env_var: MODEL_3B_PATH",
                "clip_data:",
                "  max_frames: 6",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(child)

    assert cfg["sample_id"] == "child"
    assert cfg["model"]["env_var"] == "MODEL_3B_PATH"
    assert cfg["model"]["prompt_version"] == "base_prompt"
    assert cfg["clip_data"]["max_frames"] == 6
    assert cfg["clip_data"]["max_new_tokens"] == 4096
