from pathlib import Path

from PIL import Image

from datavideo.schemas import read_json, write_jsonl
from datavideo_multichart_v2.animation import detect_animation
from datavideo_multichart_v2.pipeline import _write_candidate_report
from datavideo_multichart_v2.qwen import ANIMATION_PROMPT, _normalize_animation


class FakeAnimationClient:
    model_path = "fake-qwen"

    def __init__(self, *, related: bool = True):
        self.related = related
        self.calls = []

    def describe_animation(self, image_paths, clip_context, frame_context):
        self.calls.append(
            {
                "image_paths": image_paths,
                "clip_context": clip_context,
                "frame_context": frame_context,
            }
        )
        return {
            "result": {
                "is_target_chart_related": self.related,
                "overall_description": "蓝色柱子持续下降，随后数值标签出现。",
                "major_actions": [
                    {
                        "action": "bar_shrink",
                        "description": "蓝色柱子持续下降。",
                        "evidence_timestamps": [0.1, 1.4],
                    }
                ],
                "confidence": 0.88,
            },
            "raw_response": "{}",
            "model_status": "qwen",
            "failure_reason": None,
        }


def _manifest(tmp_path: Path, clip_id: str, frame_count: int = 9) -> Path:
    frame_dir = tmp_path / "processed" / clip_id / "visual_frames" / "keyframe_candidates"
    frame_dir.mkdir(parents=True)
    rows = []
    for idx in range(frame_count):
        path = frame_dir / f"frame_{idx:03d}.jpg"
        Image.new("RGB", (32, 24), (idx * 10, 20, 30)).save(path)
        rows.append(
            {
                "frame_id": path.stem,
                "path": str(path),
                "timestamp": idx * 0.25,
                "fps": 4,
            }
        )
    manifest = frame_dir.parent / "keyframe_frame_manifest.jsonl"
    write_jsonl(manifest, rows)
    return manifest


def test_detect_animation_uses_one_complete_ordered_call(tmp_path):
    manifest = _manifest(tmp_path, "bar_1")
    client = FakeAnimationClient()
    out_dir = tmp_path / "generated" / "bar_1"

    report = detect_animation(
        {
            "processed_root": str(tmp_path / "processed"),
            "animation": {"sample_fps": 2, "types": ["bar_shrink", "element_appear", "other"]},
            "model": {"prompt_version": "test"},
        },
        {"clip_id": "bar_1", "chart_type": "bar"},
        {"score_manifest": str(manifest)},
        out_dir,
        client=client,
        force=True,
    )

    assert len(client.calls) == 1
    assert len(client.calls[0]["image_paths"]) == 5
    assert [row["timestamp"] for row in client.calls[0]["frame_context"]] == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert client.calls[0]["clip_context"]["target_chart_type"] == "bar"
    assert report["frame_count"] == 5
    assert report["visual_start"] == 0.0
    assert report["visual_end"] == 2.0
    assert report["overall_description"] == "蓝色柱子持续下降，随后数值标签出现。"
    assert report["major_actions"][0]["evidence_timestamps"] == [0.0, 1.5]
    assert report["prompt_version"] == "test_animation_v6"
    assert read_json(out_dir / "animation_detection.json") == report
    raw = read_json(out_dir / "animation_detection_raw.json")
    assert len(raw["image_paths"]) == 5


def test_detect_animation_allows_empty_actions_for_unrelated_clip(tmp_path):
    manifest = _manifest(tmp_path, "line_1", frame_count=5)
    report = detect_animation(
        {
            "processed_root": str(tmp_path / "processed"),
            "animation": {"sample_fps": 2},
            "model": {"prompt_version": "test"},
        },
        {"clip_id": "line_1", "chart_type": "line"},
        {"score_manifest": str(manifest)},
        tmp_path / "generated" / "line_1",
        client=FakeAnimationClient(related=False),
        force=True,
    )

    assert report["is_target_chart_related"] is False
    assert report["major_actions"] == []


def test_candidate_report_exposes_clip_level_animation_description(tmp_path):
    clip_root = tmp_path / "generated" / "bar_1"
    animation = {
        "overall_description": "蓝色柱子持续下降。",
        "major_actions": [{"action": "bar_shrink"}],
        "confidence": 0.88,
        "is_target_chart_related": True,
    }

    report = _write_candidate_report(
        clip_root,
        {"output_stem": "bar_1", "chart_type": "bar", "chart_index": 1},
        {},
        {},
        {},
        {},
        animation,
        {},
        {},
    )

    assert report["clip"]["animation_description"] == "蓝色柱子持续下降。"
    assert report["clip"]["animation_action_count"] == 1
    assert report["clip"]["animation_confidence"] == 0.88
    assert report["clip"]["is_target_chart_related"] is True


def test_animation_prompt_requires_direct_mark_comparison_and_consistency():
    assert "a difference between ordered frames is animation evidence" in ANIMATION_PROMPT
    assert "compare their absolute length or height, not only their rank" in ANIMATION_PROMPT
    assert "Never compare bars or lines from different chart identities" in ANIMATION_PROMPT
    assert '"target_mark_dimensions_change": boolean' in ANIMATION_PROMPT
    assert "A description containing decrease, shrink, shorter" in ANIMATION_PROMPT


def test_animation_normalization_resolves_concrete_change_flag_conflict():
    result = _normalize_animation(
        {
            "target_marks_visible": True,
            "target_mark_dimensions_change": True,
            "printed_values_or_time_states_change": False,
            "target_components_appear_or_disappear": False,
            "is_target_chart_related": False,
            "overall_description": "柱子长度发生变化。",
            "major_actions": [],
            "confidence": 0.9,
        }
    )

    assert result["is_target_chart_related"] is True
