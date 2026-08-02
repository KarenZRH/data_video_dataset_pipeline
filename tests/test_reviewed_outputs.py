from pathlib import Path

from datavideo.review_db import save_review
from datavideo.reviewed_outputs import apply_latest_reviews
from datavideo.schemas import read_jsonl, write_json, write_jsonl


def _cfg(tmp_path: Path) -> dict:
    return {
        "sample_id": "bar_test",
        "generated_dir": str(tmp_path / "generated"),
        "reviewed_dir": str(tmp_path / "reviewed"),
        "processed_dir": str(tmp_path / "processed"),
        "review_db": str(tmp_path / "review.db"),
    }


def test_apply_latest_reviews_builds_clean_reviewed_set(tmp_path):
    cfg = _cfg(tmp_path)
    generated = Path(cfg["generated_dir"])
    clips_root = generated / "clips"
    for clip_id in ["bar_final_000", "bar_final_001"]:
        root = clips_root / clip_id
        (root / "keyframes").mkdir(parents=True)
        (root / "clip.mp4").write_bytes(b"mp4")
        (root / "keyframes" / "initial.png").write_bytes(b"png")
        (root / "trace.svg").write_text("<svg />\n", encoding="utf-8")
        (root / "trace_preview.png").write_bytes(b"png")
        write_json(root / "svg_report.json", {"success": True})
        write_json(root / "chart_metadata.json", {})
        write_json(root / "chart_data_validation.json", {})
        write_json(root / "chart_data_raw.json", {})

    write_jsonl(
        generated / "final_bar_clips.jsonl",
        [
            {"clip_id": "bar_final_000", "start": 1.0, "end": 2.0},
            {"clip_id": "bar_final_001", "start": 3.0, "end": 4.0},
        ],
    )

    save_review(
        cfg["review_db"],
        {
            "sample_id": "bar_test",
            "stage": "stage1_review",
            "decision": "需要修改",
            "original_value": {"clip": {"clip_id": "bar_final_000"}},
            "reviewed_value": {
                "clip_id": "bar_final_000",
                "clip": {"start": 1.0, "end": 2.0},
                "chart_data": [{"index": 0, "label": "A", "value": "10"}],
            },
            "reviewer": "local",
        },
    )
    save_review(
        cfg["review_db"],
        {
            "sample_id": "bar_test",
            "stage": "stage1_review",
            "decision": "排除",
            "original_value": {"clip": {"clip_id": "bar_final_001"}},
            "reviewed_value": {"clip_id": "bar_final_001"},
            "reviewer": "local",
        },
    )

    report = apply_latest_reviews(cfg)

    assert report["accepted_count"] == 1
    assert report["excluded_count"] == 1
    assert read_jsonl(Path(cfg["reviewed_dir"]) / "final_bar_clips.jsonl")[0]["clip_id"] == "bar_final_000"
    assert (Path(cfg["reviewed_dir"]) / "clips" / "bar_final_000" / "clip.mp4").exists()
    assert "A,10" in (Path(cfg["reviewed_dir"]) / "clips" / "bar_final_000" / "chart_data.csv").read_text()
    assert not (Path(cfg["reviewed_dir"]) / "clips" / "bar_final_001").exists()
