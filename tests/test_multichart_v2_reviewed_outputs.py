from pathlib import Path

from datavideo.review_db import save_review
from datavideo.schemas import read_json, read_jsonl, write_json, write_jsonl
from datavideo_multichart_v2.reviewed_outputs import REVIEW_STAGE, apply_latest_reviews


def _cfg(tmp_path: Path) -> dict:
    return {
        "sample_id": "datavideo_multichart_v2_test",
        "generated_root": str(tmp_path / "generated_v2"),
        "reviewed_dir": str(tmp_path / "reviewed"),
        "processed_root": str(tmp_path / "processed"),
        "review_db": str(tmp_path / "review.db"),
    }


def test_apply_latest_reviews_writes_reviewed_narration(tmp_path):
    cfg = _cfg(tmp_path)
    generated = Path(cfg["generated_root"])
    clip_root = generated / "bar_1"
    (clip_root / "keyframes").mkdir(parents=True)
    (clip_root / "clip.mp4").write_bytes(b"mp4")
    (clip_root / "keyframes" / "initial.png").write_bytes(b"png")
    write_json(clip_root / "svg_report.json", {})
    write_json(clip_root / "chart_metadata.json", {})
    write_json(clip_root / "chart_data_validation.json", {})
    write_json(clip_root / "chart_data_clip_raw.json", {})
    write_json(clip_root / "animation_detection.json", {"overall_description": "machine animation"})
    write_json(clip_root / "animation_detection_raw.json", {})
    write_json(
        generated / "multichart_v2_run_report.json",
        {
            "clips": [
                {
                    "clip": {"clip_id": "bar_1", "start_time": "00:00:01", "end_time": "00:00:03"},
                    "keyframes": {"assets": {"initial": str(clip_root / "keyframes" / "initial.png")}},
                }
            ]
        },
    )

    processed = Path(cfg["processed_root"]) / "bar_1" / "narration"
    processed.mkdir(parents=True)
    write_json(processed.parent / "intervals.json", {"requires_context_redownload": False})
    write_json(processed / "transcript_provenance.json", {"narration_status": "provisional"})
    write_jsonl(
        processed / "selected_full_sentences.jsonl",
        [
            {
                "text": "hello world",
                "start_context": 0.0,
                "end_context": 1.0,
                "confidence": 0.9,
            }
        ],
    )

    save_review(
        cfg["review_db"],
        {
            "sample_id": cfg["sample_id"],
            "stage": REVIEW_STAGE,
            "decision": "approved",
            "original_value": {"clip": {"clip_id": "bar_1"}},
            "reviewed_value": {
                "clip_id": "bar_1",
                "clip": {"start_seconds": 0.0, "end_seconds": 2.0},
                "narration": {
                    "clip_id": "bar_1",
                    "status": "provisional",
                    "sentences": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
                    "full_text": "hello world",
                    "machine_source": str(processed / "selected_full_sentences.jsonl"),
                },
                "animation": {
                    "overall_description": "reviewed animation",
                    "major_actions": [{"action": "bar_grow", "description": "bar grows", "evidence_timestamps": [0.5]}],
                },
            },
            "reviewer": "local",
        },
    )

    report = apply_latest_reviews(cfg)

    assert report["accepted_count"] == 1
    final_rows = read_jsonl(Path(cfg["reviewed_dir"]) / "final_multichart_v2_clips.jsonl")
    assert final_rows[0]["narration_text"] == "hello world"
    assert final_rows[0]["animation_description"] == "reviewed animation"
    reviewed_clip = read_json(Path(cfg["reviewed_dir"]) / "clips" / "bar_1" / "clip.json")
    assert reviewed_clip["narration_status"] == "provisional"
    assert reviewed_clip["narration_text"] == "hello world"
    assert read_json(Path(cfg["reviewed_dir"]) / "clips" / "bar_1" / "narration_reviewed.json")["full_text"] == "hello world"
    assert (Path(cfg["reviewed_dir"]) / "clips" / "bar_1" / "narration" / "selected_full_sentences.jsonl").exists()
