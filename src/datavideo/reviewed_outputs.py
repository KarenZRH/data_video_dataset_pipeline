from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .media import extract_clip
from .review_db import latest_reviews_by_clip
from .schemas import ensure_dir, read_jsonl, write_csv, write_json, write_jsonl


ACCEPT_DECISIONS = {"通过", "需要修改", "保存"}
EXCLUDE_DECISIONS = {"排除"}


def apply_latest_reviews(cfg: dict[str, Any]) -> dict[str, Any]:
    generated = Path(cfg["generated_dir"])
    reviewed = ensure_dir(cfg["reviewed_dir"])
    reviewed_clips = reviewed / "clips"
    if reviewed_clips.exists():
        shutil.rmtree(reviewed_clips)
    ensure_dir(reviewed_clips)

    source_clips = read_jsonl(generated / "final_bar_clips.jsonl")
    reviews = latest_reviews_by_clip(cfg["review_db"], cfg["sample_id"])
    normalized = Path(cfg["processed_dir"]) / "normalized.mp4"

    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    unreviewed: list[dict[str, Any]] = []

    for source_clip in source_clips:
        clip_id = source_clip["clip_id"]
        review = reviews.get(clip_id)
        if review is None:
            unreviewed.append({"clip_id": clip_id, "reason": "no_review"})
            continue
        if review["decision"] in EXCLUDE_DECISIONS:
            excluded.append(_review_summary(clip_id, review))
            continue
        if review["decision"] not in ACCEPT_DECISIONS:
            unreviewed.append({"clip_id": clip_id, "reason": f"unsupported_decision:{review['decision']}"})
            continue

        clip_root = ensure_dir(reviewed_clips / clip_id)
        source_root = generated / "clips" / clip_id
        if source_root.exists():
            _copy_review_assets(source_root, clip_root)

        reviewed_value = review.get("reviewed_value", {})
        reviewed_clip = {**source_clip, **reviewed_value.get("clip", {})}
        reviewed_clip["clip_id"] = clip_id
        reviewed_clip["review_decision"] = review["decision"]
        reviewed_clip["review_id"] = review["id"]
        reviewed_clip["reviewed_at"] = review["reviewed_at"]
        reviewed_clip["reviewer"] = review["reviewer"]
        reviewed_clip["notes"] = review["notes"]
        reviewed_clip["clip_mp4"] = str(clip_root / "clip.mp4")
        reviewed_clip["chart_data_csv"] = str(clip_root / "chart_data.csv")

        _write_reviewed_clip_video(normalized, source_root, clip_root, source_clip, reviewed_clip)
        chart_rows = reviewed_value.get("chart_data") or []
        write_csv(clip_root / "chart_data.csv", _normalize_chart_rows(chart_rows))
        write_json(clip_root / "review.json", review)
        write_json(clip_root / "clip.json", reviewed_clip)
        accepted.append(reviewed_clip)

    write_jsonl(reviewed / "final_bar_clips.jsonl", accepted)
    write_jsonl(reviewed / "excluded_clips.jsonl", excluded)
    write_jsonl(reviewed / "unreviewed_clips.jsonl", unreviewed)
    report = {
        "sample_id": cfg["sample_id"],
        "source": str(generated / "final_bar_clips.jsonl"),
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
        "semantic.svg",
        "semantic_preview.png",
        "svg_report.json",
        "chart_data_raw.json",
        "chart_metadata.json",
        "chart_data_validation.json",
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


def _write_reviewed_clip_video(
    normalized: Path,
    source_root: Path,
    clip_root: Path,
    source_clip: dict[str, Any],
    reviewed_clip: dict[str, Any],
) -> None:
    source_start = float(source_clip["start"])
    source_end = float(source_clip["end"])
    reviewed_start = float(reviewed_clip["start"])
    reviewed_end = float(reviewed_clip["end"])
    out = clip_root / "clip.mp4"
    if reviewed_start != source_start or reviewed_end != source_end:
        extract_clip(normalized, reviewed_start, reviewed_end, out, force=True)
        return
    source_video = source_root / "clip.mp4"
    if source_video.exists():
        shutil.copy2(source_video, out)
    else:
        extract_clip(normalized, reviewed_start, reviewed_end, out, force=True)


def _normalize_chart_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_rows = []
    for idx, row in enumerate(rows):
        clean = {key: value for key, value in row.items() if not str(key).startswith("_")}
        clean.setdefault("index", idx)
        clean_rows.append(clean)
    return clean_rows or [{"index": 0, "label": None, "value": None}]


def _review_summary(clip_id: str, review: dict[str, Any]) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        "decision": review["decision"],
        "review_id": review["id"],
        "reviewed_at": review["reviewed_at"],
        "reviewer": review["reviewer"],
        "notes": review["notes"],
    }
