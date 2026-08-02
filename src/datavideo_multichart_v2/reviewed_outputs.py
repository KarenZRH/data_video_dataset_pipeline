from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from datavideo.review_db import latest_reviews_by_clip
from datavideo.schemas import ensure_dir, read_json, write_csv, write_json, write_jsonl


APPROVED_DECISIONS = {"approved"}
BLOCKING_DECISIONS = {"needs_revision", "saved", "需要修改", "保存", "通过"}
EXCLUDE_DECISIONS = {"excluded", "排除"}
REVIEW_STAGE = "multichart_v2_review"
CLIP_REVIEW_STAGE = "multichart_v2_clip_review"


def compose_reviewed_intervals(cfg: dict[str, Any], clip_id: str) -> dict[str, Any]:
    processed_root = Path(cfg.get("processed_root", "data/processed")) / clip_id
    return read_json(processed_root / "intervals.json")


def apply_latest_reviews(cfg: dict[str, Any]) -> dict[str, Any]:
    generated = Path(cfg.get("generated_root", cfg.get("generated_dir", "data/generated_v2")))
    reviewed = ensure_dir(cfg.get("reviewed_dir", "data/reviewed/datavideo_multichart_v2"))
    reviewed_clips = reviewed / "clips"
    if reviewed_clips.exists():
        shutil.rmtree(reviewed_clips)
    ensure_dir(reviewed_clips)

    run_report_path = generated / "multichart_v2_run_report.json"
    run_report = read_json(run_report_path) if run_report_path.exists() else {"clips": []}
    source_reports = run_report.get("clips", []) if isinstance(run_report.get("clips"), list) else []
    asset_reviews = latest_reviews_by_clip(cfg["review_db"], cfg["sample_id"], stage=REVIEW_STAGE)

    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    unreviewed: list[dict[str, Any]] = []

    for source_report in source_reports:
        source_clip = source_report.get("clip", {})
        clip_id = source_clip.get("clip_id")
        if not clip_id:
            continue
        asset_review = asset_reviews.get(clip_id)
        if asset_review is None:
            unreviewed.append({"clip_id": clip_id, "reason": "no_asset_review"})
            continue
        decision = asset_review["decision"]
        if decision in EXCLUDE_DECISIONS:
            excluded.append(_review_summary(clip_id, asset_review))
            continue
        if decision in BLOCKING_DECISIONS:
            unreviewed.append({"clip_id": clip_id, "reason": f"blocking_decision:{decision}"})
            continue
        if decision not in APPROVED_DECISIONS:
            unreviewed.append({"clip_id": clip_id, "reason": f"unsupported_decision:{decision}"})
            continue
        processed_root = Path(cfg.get("processed_root", "data/processed")) / clip_id
        machine_intervals = read_json(processed_root / "intervals.json")

        source_root = generated / clip_id
        clip_root = ensure_dir(reviewed_clips / clip_id)
        _copy_review_assets(source_root, clip_root)
        _copy_context_assets(processed_root, clip_root)

        asset_value = asset_review.get("reviewed_value", {}) if asset_review else {}
        reviewed_clip_value = asset_value.get("clip", {})
        reviewed_clip = {**source_clip, **reviewed_clip_value}
        reviewed_clip["clip_id"] = clip_id
        reviewed_clip["asset_review_decision"] = asset_review["decision"] if asset_review else None
        reviewed_clip["clip_review_decision"] = None
        reviewed_clip["review_decision"] = asset_review["decision"]
        reviewed_clip["review_id"] = asset_review["id"]
        reviewed_clip["reviewed_at"] = asset_review["reviewed_at"]
        reviewed_clip["reviewer"] = asset_review["reviewer"]
        reviewed_clip["notes"] = asset_review["notes"]
        reviewed_clip["clip_mp4"] = str(clip_root / "clip.mp4")
        reviewed_clip["chart_data_csv"] = str(clip_root / "chart_data.csv")
        reviewed_clip["final_keyframe"] = str(clip_root / "keyframes" / "final.png")
        reviewed_animation = asset_value.get("animation") or {}
        reviewed_actions = reviewed_animation.get("major_actions") if isinstance(reviewed_animation.get("major_actions"), list) else []
        reviewed_clip["animation_action_count"] = len(reviewed_actions)
        reviewed_clip["animation_description"] = reviewed_animation.get("overall_description")
        reviewed_narration = asset_value.get("narration") or {}
        reviewed_clip["narration_status"] = reviewed_narration.get("status") or _narration_status(processed_root, machine_intervals)
        reviewed_clip["narration_text"] = reviewed_narration.get("full_text") or ""

        final_intervals = compose_reviewed_intervals(cfg, clip_id)
        write_json(clip_root / "intervals.json", final_intervals)
        _write_reviewed_clip_video(source_root, clip_root)
        _copy_narration_outputs(processed_root, clip_root, final_intervals)
        _write_final_keyframe(source_root, clip_root, asset_value.get("keyframe", {}))
        _write_reviewed_chart_data(clip_root, asset_value.get("chart_data") or [])
        _write_reviewed_animation(clip_root, asset_value.get("animation") or {})
        _write_reviewed_narration(clip_root, reviewed_narration)
        write_json(clip_root / "review.json", {"asset_review": asset_review, "deprecated_clip_review": None})
        write_json(clip_root / "clip.json", reviewed_clip)
        accepted.append(reviewed_clip)

    write_jsonl(reviewed / "final_multichart_v2_clips.jsonl", accepted)
    write_jsonl(reviewed / "excluded_clips.jsonl", excluded)
    write_jsonl(reviewed / "unreviewed_clips.jsonl", unreviewed)
    report = {
        "sample_id": cfg["sample_id"],
        "source": str(run_report_path),
        "accepted_count": len(accepted),
        "excluded_count": len(excluded),
        "unreviewed_count": len(unreviewed),
        "reviewed_dir": str(reviewed),
    }
    write_json(reviewed / "reviewed_report.json", report)
    return report


def _copy_review_assets(source_root: Path, clip_root: Path) -> None:
    for name in [
        "keyframes",
        "trace.svg",
        "trace_preview.png",
        "svg_report.json",
        "chart_data_clip_raw.json",
        "chart_metadata.json",
        "chart_data_validation.json",
        "animation_detection.json",
        "animation_detection_raw.json",
        "clip_report.json",
        "data_events.jsonl",
    ]:
        src = source_root / name
        dst = clip_root / name
        if not src.exists():
            continue
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _copy_context_assets(processed_root: Path, clip_root: Path) -> None:
    raw_narration = processed_root / "narration"
    if raw_narration.exists():
        dst = clip_root / "context_narration"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(raw_narration, dst)


def _write_reviewed_clip_video(source_root: Path, clip_root: Path) -> None:
    out = clip_root / "clip.mp4"
    source_video = source_root / "clip.mp4"
    if source_video.exists():
        shutil.copy2(source_video, out)


def _copy_narration_outputs(processed_root: Path, clip_root: Path, intervals: dict[str, Any]) -> None:
    src_dir = processed_root / "narration"
    dst_dir = ensure_dir(clip_root / "narration")
    if src_dir.exists():
        for src in src_dir.iterdir():
            if src.is_file():
                shutil.copy2(src, dst_dir / src.name)
    write_json(
        dst_dir / "narration_status.json",
        {
            "status": _narration_status(processed_root, intervals),
            "requires_context_redownload": bool(intervals.get("requires_context_redownload")),
            "review_status": "machine_asr_provisional",
        },
    )


def _narration_status(processed_root: Path, intervals: dict[str, Any]) -> str:
    if intervals.get("requires_context_redownload"):
        return "incomplete_context"
    provenance = processed_root / "narration" / "transcript_provenance.json"
    if provenance.exists():
        return str(read_json(provenance).get("narration_status") or "provisional")
    return "missing"


def _write_final_keyframe(source_root: Path, clip_root: Path, keyframe_review: dict[str, Any]) -> None:
    asset = keyframe_review.get("asset") or keyframe_review.get("path")
    if not asset:
        asset = source_root / "keyframes" / "initial.png"
    src = Path(asset)
    if not src.is_absolute():
        src = Path.cwd() / src
    if not src.exists():
        src = source_root / "keyframes" / "initial.png"
    dst = ensure_dir(clip_root / "keyframes") / "final.png"
    if src.exists():
        shutil.copy2(src, dst)
    write_json(clip_root / "keyframes" / "final_keyframe.json", keyframe_review)


def _write_reviewed_chart_data(clip_root: Path, rows: list[dict[str, Any]]) -> None:
    clean_rows = _normalize_chart_rows(rows)
    csv_path = clip_root / "chart_data.csv"
    if clean_rows:
        write_csv(csv_path, clean_rows)
    elif csv_path.exists():
        csv_path.unlink()
    write_json(clip_root / "chart_data_reviewed.json", {"rows": clean_rows})


def _write_reviewed_animation(clip_root: Path, animation: dict[str, Any]) -> None:
    write_json(clip_root / "animation_reviewed.json", animation)


def _write_reviewed_narration(clip_root: Path, narration: dict[str, Any]) -> None:
    write_json(clip_root / "narration_reviewed.json", narration)
    write_json(clip_root / "narration" / "narration_reviewed.json", narration)


def _normalize_chart_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        clean = {key: value for key, value in row.items() if not str(key).startswith("_")}
        if any(value not in (None, "") for value in clean.values()):
            clean_rows.append(clean)
    return clean_rows


def _review_summary(clip_id: str, review: dict[str, Any]) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        "decision": review["decision"],
        "review_id": review["id"],
        "reviewed_at": review["reviewed_at"],
        "reviewer": review["reviewer"],
        "notes": review["notes"],
    }
