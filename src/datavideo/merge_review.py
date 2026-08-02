from __future__ import annotations

from pathlib import Path
from typing import Any

from .frames import extract_frames, write_contact_sheet
from .media import extract_clip
from .model_client import make_qwen_client
from .schemas import ensure_dir, read_jsonl, write_json, write_jsonl


DATA_CUE_TERMS = {
    "bar",
    "race",
    "progress",
    "distance",
    "comparison",
    "rank",
    "label",
    "text",
    "growth",
    "line",
}


def _cue_tokens(clip: dict[str, Any]) -> set[str]:
    text = " ".join(clip.get("animation_cues", []) + clip.get("scene_states", [])).lower()
    return {term for term in DATA_CUE_TERMS if term in text}


def has_data_encoding_hint(clip: dict[str, Any]) -> bool:
    tokens = _cue_tokens(clip)
    return bool(tokens & {"bar", "race", "progress", "distance", "comparison", "rank", "label"})


def _same_visual_group(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if b["start"] - a["end"] > 2.0:
        return False
    at = _cue_tokens(a)
    bt = _cue_tokens(b)
    if at & bt:
        return True
    if has_data_encoding_hint(a) and has_data_encoding_hint(b):
        return True
    return False


def merge_atomic_clips(clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [clip for clip in sorted(clips, key=lambda row: row["start"]) if has_data_encoding_hint(clip)]
    merged: list[dict[str, Any]] = []
    for clip in eligible:
        if not merged or not _same_visual_group(merged[-1]["atomic_clips"][-1], clip):
            merged.append(
                {
                    "clip_id": f"merged_{len(merged):03d}",
                    "start": clip["start"],
                    "end": clip["end"],
                    "atomic_clip_ids": [clip["clip_id"]],
                    "atomic_clips": [clip],
                    "merge_reasons": ["data_encoding_hint"],
                }
            )
            continue
        current = merged[-1]
        gap = round(clip["start"] - current["end"], 3)
        current["end"] = max(current["end"], clip["end"])
        current["atomic_clip_ids"].append(clip["clip_id"])
        current["atomic_clips"].append(clip)
        current["merge_reasons"].append(f"gap<={gap}s_same_scene_or_mark")

    for row in merged:
        cues = sorted({cue for clip in row["atomic_clips"] for cue in clip.get("animation_cues", [])})
        states = sorted({state for clip in row["atomic_clips"] for state in clip.get("scene_states", [])})
        confidences = [float(clip.get("confidence", 0.0) or 0.0) for clip in row["atomic_clips"]]
        row["start"] = round(row["start"], 3)
        row["end"] = round(row["end"], 3)
        row["duration"] = round(row["end"] - row["start"], 3)
        row["animation_cues"] = cues
        row["scene_states"] = states
        row["confidence"] = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        row["data_encoding_hint"] = True
    return merged


def review_merged_clips(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    generated = ensure_dir(Path(cfg["generated_dir"]) / "merged_clip_review")
    normalized = Path(cfg["processed_dir"]) / "normalized.mp4"
    refined = read_jsonl(Path(cfg["generated_dir"]) / "refined_clips.jsonl")
    merged = merge_atomic_clips(refined)
    write_jsonl(generated / "merged_clips_pre_review.jsonl", merged)

    client = make_qwen_client(cfg)
    reviewed = []
    for idx, clip in enumerate(merged):
        review_id = f"merged_{idx:03d}"
        clip["clip_id"] = review_id
        clip_dir = ensure_dir(generated / review_id)
        clip_path = extract_clip(normalized, clip["start"], clip["end"], clip_dir / f"{review_id}.mp4", force=force)
        frame_dir = Path(cfg["processed_dir"]) / "frames" / "merged_review_2fps" / review_id
        frames = extract_frames(
            normalized,
            frame_dir,
            2,
            cfg["sampling"]["short_side"],
            "merged",
            force=force,
            start=clip["start"],
            end=clip["end"],
        )
        scores = {row["frame_id"]: clip["confidence"] for row in frames}
        label = ",".join(clip["animation_cues"][:4])
        labels = {row["frame_id"]: label for row in frames}
        sheet_path = write_contact_sheet(frames, clip_dir / f"{review_id}_contact_sheet.jpg", max_cols=4, thumb_width=360, scores=scores, labels=labels)
        qwen = client.review_merged_clip_contact_sheet(str(sheet_path))
        item = {
            **clip,
            "clip_mp4": str(clip_path),
            "contact_sheet": str(sheet_path),
            "qwen_review": qwen["result"],
            "raw_response": qwen["raw_response"],
            "model_status": qwen["model_status"],
            "failure_reason": qwen["failure_reason"],
            "model_path": client.model_path,
            "model_version": client.model_version,
            "prompt_version": f"{cfg['model']['prompt_version']}_merged_review_v1",
            "config_hash": cfg["config_hash"],
        }
        item["final_decision"] = "keep" if qwen["result"].get("decision") == "keep" else "exclude"
        write_json(clip_dir / f"{review_id}.json", item)
        reviewed.append(item)

    kept = [row for row in reviewed if row["final_decision"] == "keep"]
    write_jsonl(generated / "merged_clips_reviewed.jsonl", reviewed)
    write_jsonl(Path(cfg["generated_dir"]) / "merged_clips.jsonl", kept)
    report = {
        "sample_id": cfg["sample_id"],
        "atomic_clip_count": len(refined),
        "merged_pre_review_count": len(merged),
        "merged_kept_count": len(kept),
        "merged_output": str(Path(cfg["generated_dir"]) / "merged_clips.jsonl"),
        "review_dir": str(generated),
        "model_path": client.model_path,
        "model_version": client.model_version,
        "config_hash": cfg["config_hash"],
    }
    write_json(generated / "merged_clip_review_report.json", report)
    return report
