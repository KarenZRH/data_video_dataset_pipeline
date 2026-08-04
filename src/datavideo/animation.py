from __future__ import annotations

from pathlib import Path
from typing import Any

from datavideo.schemas import ensure_dir, read_json, read_jsonl, write_json
from datavideo.visual_provenance import assert_qwen_visual_inputs

from .multichart_qwen import MultichartQwenClient


def _clip_id(row: dict[str, Any]) -> str:
    return str(row.get("output_stem") or row.get("clip_id") or f"{row['chart_type']}_{row['chart_index']}")


def _load_frame_rows(cfg: dict[str, Any], clip_id: str, keyframes: dict[str, Any]) -> list[dict[str, Any]]:
    score_manifest = Path(str(keyframes.get("score_manifest", "")))
    if score_manifest.is_file():
        rows = read_jsonl(score_manifest)
    else:
        manifest = (
            Path(cfg.get("processed_root", "data/processed"))
            / clip_id
            / "visual_frames"
            / "keyframe_frame_manifest.jsonl"
        )
        rows = read_jsonl(manifest) if manifest.is_file() else []

    frames = []
    for row in rows:
        path = Path(str(row.get("path", "")))
        if not path.is_file():
            continue
        frames.append({**row, "path": str(path), "timestamp": float(row.get("timestamp", 0.0))})
    return sorted(frames, key=lambda item: float(item["timestamp"]))


def _sample_complete_sequence(frames: list[dict[str, Any]], sample_fps: float) -> list[dict[str, Any]]:
    if len(frames) <= 1:
        return frames
    interval = 1.0 / sample_fps
    start = float(frames[0]["timestamp"])
    end = float(frames[-1]["timestamp"])
    target_count = int((end - start) / interval) + 1
    targets = [start + idx * interval for idx in range(target_count)]
    if not targets or end - targets[-1] > interval * 0.5:
        targets.append(end)

    selected = []
    seen: set[str] = set()
    for target in targets:
        frame = min(frames, key=lambda item: abs(float(item["timestamp"]) - target))
        identity = str(frame.get("frame_id") or frame["path"])
        if identity not in seen:
            selected.append(frame)
            seen.add(identity)
    return selected


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _normalize_actions(actions: Any, sampled_timestamps: list[float]) -> list[dict[str, Any]]:
    if not isinstance(actions, list):
        return []
    normalized = []
    for item in actions:
        if not isinstance(item, dict):
            continue
        evidence = []
        values = item.get("evidence_timestamps")
        if isinstance(values, list) and sampled_timestamps:
            for value in values:
                try:
                    requested = float(value)
                except (TypeError, ValueError):
                    continue
                timestamp = min(sampled_timestamps, key=lambda candidate: abs(candidate - requested))
                if timestamp not in evidence:
                    evidence.append(timestamp)
        normalized.append(
            {
                "action": str(item.get("action", "other") or "other"),
                "description": str(item.get("description", "") or ""),
                "evidence_timestamps": evidence,
            }
        )
    return normalized


def detect_animation(
    cfg: dict[str, Any],
    row: dict[str, Any],
    keyframes: dict[str, Any],
    out_dir: str | Path,
    *,
    client: MultichartQwenClient | None = None,
    force: bool = False,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    report_path = out_dir / "animation_detection.json"
    raw_path = out_dir / "animation_detection_raw.json"
    if report_path.is_file() and not force:
        cached = read_json(report_path)
        if str(cached.get("prompt_version", "")).endswith("_animation_v6"):
            return cached

    clip_id = _clip_id(row)
    frames = _load_frame_rows(cfg, clip_id, keyframes)
    if not frames:
        raise RuntimeError(f"No visual frame samples available for animation detection for {clip_id}")

    animation_cfg = cfg.get("animation", {})
    sample_fps = max(0.1, float(animation_cfg.get("sample_fps", 2.0)))
    sampled_frames = _sample_complete_sequence(frames, sample_fps)
    image_paths = [frame["path"] for frame in sampled_frames]
    assert_qwen_visual_inputs(image_paths)

    timestamps = [round(float(frame["timestamp"]), 3) for frame in sampled_frames]
    frame_context = [
        {
            "image_index": idx,
            "frame_id": frame.get("frame_id"),
            "timestamp": timestamp,
        }
        for idx, (frame, timestamp) in enumerate(zip(sampled_frames, timestamps), start=1)
    ]
    clip_context = {
        "clip_id": clip_id,
        "target_chart_type": row.get("chart_type"),
        "title": row.get("raw_video_title"),
        "sample_fps": sample_fps,
        "frame_count": len(sampled_frames),
        "visual_start": timestamps[0],
        "visual_end": timestamps[-1],
        "allowed_animation_types": animation_cfg.get("types", []),
        "video_role": "visual_clip",
    }

    scorer = client or MultichartQwenClient(cfg)
    response = scorer.describe_animation(image_paths, clip_context, frame_context)
    result = response.get("result") if isinstance(response.get("result"), dict) else {}
    related = bool(result.get("is_target_chart_related", False))
    description = str(result.get("overall_description", "") or "")
    if not description:
        description = "检测到与目标图表相关的动画。" if related else "没有检测到与目标图表相关的动画。"

    report = {
        "clip_id": clip_id,
        "target_chart_type": row.get("chart_type"),
        "sample_fps": sample_fps,
        "frame_count": len(sampled_frames),
        "visual_start": timestamps[0],
        "visual_end": timestamps[-1],
        "is_target_chart_related": related,
        "overall_description": description,
        "major_actions": _normalize_actions(result.get("major_actions", []), timestamps) if related else [],
        "confidence": _clamp01(result.get("confidence", 0.0)),
        "model_status": response.get("model_status"),
        "failure_reason": response.get("failure_reason"),
        "model_path": scorer.model_path,
        "prompt_version": f"{cfg.get('model', {}).get('prompt_version', 'qwen_multichart_assets_v2')}_animation_v6",
    }
    write_json(
        raw_path,
        {
            "clip_context": clip_context,
            "frame_context": frame_context,
            "image_paths": image_paths,
            "response": response,
        },
    )
    write_json(report_path, report)
    return report
