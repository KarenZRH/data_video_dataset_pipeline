from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .frames import extract_frames, write_contact_sheet, write_frame_manifest
from .media import extract_clip, ffprobe, normalize_video
from .model_client import make_qwen_client
from .schemas import ensure_dir, read_jsonl, write_json, write_jsonl


POSITIVE_STATES = {"chart_entering", "stable_chart", "chart_animating", "chart_leaving", "transition", "uncertain"}


def _duration(video: str | Path) -> float:
    return float(ffprobe(video)["format"]["duration"])


def _is_candidate(result: dict[str, Any]) -> bool:
    if result.get("is_data_video_clip_candidate"):
        return True
    if result.get("contains_data_marks") and result.get("scene_state") in POSITIVE_STATES:
        return True
    return False


def _confidence(result: dict[str, Any]) -> float:
    try:
        return float(result.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _classify_frames(
    cfg: dict[str, Any],
    client: Any,
    frames: list[dict[str, Any]],
    out_path: Path | None,
    force: bool,
) -> list[dict[str, Any]]:
    if out_path and out_path.exists() and not force:
        return read_jsonl(out_path)
    rows = []
    max_n = int(cfg["model"].get("max_frames_per_call", 3))
    for i in range(0, len(frames), max_n):
        group = frames[i : i + max_n]
        response = client.detect_data_video_clip_candidate([row["path"] for row in group])
        for frame in group:
            rows.append(
                {
                    "sample_id": cfg["sample_id"],
                    "frame_id": frame["frame_id"],
                    "timestamp": frame["timestamp"],
                    "image_path": frame["path"],
                    "result": response["result"],
                    "raw_response": response["raw_response"],
                    "model_status": response["model_status"],
                    "model_path": client.model_path,
                    "model_version": client.model_version or (Path(client.model_path).name if client.model_path else None),
                    "prompt_version": cfg["model"]["prompt_version"],
                    "config_hash": cfg["config_hash"],
                    "failure_reason": response["failure_reason"],
                }
            )
    if out_path:
        write_jsonl(out_path, rows)
    return rows


def merge_candidates(
    frame_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    *,
    max_gap_seconds: float,
    expand_seconds: float,
    min_duration: float,
    duration: float,
) -> list[dict[str, Any]]:
    by_id = {row["frame_id"]: row for row in result_rows}
    positives = []
    for frame in frame_rows:
        result = by_id.get(frame["frame_id"], {}).get("result", {})
        if _is_candidate(result):
            positives.append(
                {
                    **frame,
                    "confidence": _confidence(result),
                    "scene_state": result.get("scene_state", "uncertain"),
                    "animation_cue": result.get("animation_cue", "unknown"),
                    "chart_completeness": result.get("chart_completeness", 0.0),
                }
            )

    clips = []
    if not positives:
        return clips
    start = end = positives[0]["timestamp"]
    rows = [positives[0]]
    last = positives[0]["timestamp"]
    for row in positives[1:]:
        if row["timestamp"] - last <= max_gap_seconds:
            end = row["timestamp"]
            rows.append(row)
        else:
            clips.append(_clip(start, end, rows, expand_seconds, min_duration, duration, len(clips)))
            start = end = row["timestamp"]
            rows = [row]
        last = row["timestamp"]
    clips.append(_clip(start, end, rows, expand_seconds, min_duration, duration, len(clips)))
    return clips


def _clip(start: float, end: float, rows: list[dict[str, Any]], expand: float, min_duration: float, duration: float, idx: int) -> dict[str, Any]:
    s = max(0.0, start - expand)
    e = min(duration, end + expand)
    if e - s < min_duration:
        pad = (min_duration - (e - s)) / 2
        s = max(0.0, s - pad)
        e = min(duration, e + pad)
    confidences = [_confidence({"confidence": row.get("confidence")}) for row in rows]
    states = sorted({row.get("scene_state", "uncertain") for row in rows})
    cues = sorted({row.get("animation_cue", "unknown") for row in rows})
    return {
        "clip_id": f"candidate_{idx:03d}",
        "start": round(s, 3),
        "end": round(e, 3),
        "source_start": round(start, 3),
        "source_end": round(end, 3),
        "confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        "scene_states": states,
        "animation_cues": cues,
        "positive_frame_count": len(rows),
        "status": "uncertain" if any(c <= 0.35 for c in confidences) else "candidate",
    }


def _sheet_scores_and_labels(results: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, str]]:
    scores = {}
    labels = {}
    for row in results:
        result = row.get("result", {})
        scores[row["frame_id"]] = _confidence(result)
        labels[row["frame_id"]] = f"{result.get('scene_state', 'uncertain')} | {result.get('animation_cue', 'unknown')}"
    return scores, labels


def _write_review_readme(review_dir: Path) -> None:
    text = """# Data-video Clip Candidate Review

本轮只检查 data-video clip detection，不检查关键帧、SVG、图表数据或完整质检页面。

每个候选包含：

- `candidate_xxx.mp4`: 标准化视频中截出的候选片段
- `candidate_xxx_contact_sheet.jpg`: 该候选的 8 FPS 细采样总览，标注时间戳、confidence、scene_state、animation_cue
- `candidate_xxx.json`: 候选边界、置信度、状态、模型路径、配置 hash 和对应文件

人工判断是否命中你关心的 data-video clip 时，优先看：

1. 是否围绕同一数据表达/数据可视化意图展开，而不只是孤立静态画面。
2. 是否包含数据 mark 的出现、增长、缩短、强调、排序或离开。
3. 是否应该把进入前的视觉铺垫也纳入 clip 起点。
4. 是否为非数据的装饰图、普通字幕、照片、地图或 UI 进度条。

本轮目标是高召回：多给候选可以接受，漏掉 1:05-1:18 这类 data-video 片段不可接受。
"""
    (review_dir / "README.md").write_text(text, encoding="utf-8")


def write_review_package(
    cfg: dict[str, Any],
    normalized: str | Path,
    refined: list[dict[str, Any]],
    *,
    force: bool = False,
    model_path: str | None = None,
    model_version: str | None = None,
) -> list[dict[str, Any]]:
    processed = ensure_dir(cfg["processed_dir"])
    review_dir = ensure_dir(Path(cfg["reviewed_dir"]) / "candidates")
    _write_review_readme(review_dir)
    review_items = []
    for idx, clip in enumerate(refined):
        clip_id = f"candidate_{idx:03d}"
        clip["review_id"] = clip_id
        clip_path = review_dir / f"{clip_id}.mp4"
        extract_clip(normalized, clip["start"], clip["end"], clip_path, force=force)
        review_frame_dir = processed / "frames" / "review_8fps" / clip_id
        fine_frames = extract_frames(
            normalized,
            review_frame_dir,
            cfg["sampling"]["fine_fps"],
            cfg["sampling"]["short_side"],
            "fine",
            force=force,
            start=clip["start"],
            end=clip["end"],
        )
        scores = {row["frame_id"]: float(clip.get("confidence", 0.0) or 0.0) for row in fine_frames}
        label = f"{','.join(clip.get('scene_states', ['uncertain']))} | {','.join(clip.get('animation_cues', ['unknown']))}"
        labels = {row["frame_id"]: label for row in fine_frames}
        sheet_path = review_dir / f"{clip_id}_contact_sheet.jpg"
        write_contact_sheet(fine_frames, sheet_path, max_cols=4, thumb_width=360, scores=scores, labels=labels)
        item = {
            "sample_id": cfg["sample_id"],
            "candidate_id": clip_id,
            "clip": clip,
            "clip_mp4": str(clip_path),
            "contact_sheet": str(sheet_path),
            "model_path": model_path,
            "model_version": model_version,
            "prompt_version": cfg["model"]["prompt_version"],
            "config_hash": cfg["config_hash"],
        }
        write_json(review_dir / f"{clip_id}.json", item)
        review_items.append(item)
    return review_items


def detect_data_video_clips(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    processed = ensure_dir(cfg["processed_dir"])
    generated = ensure_dir(cfg["generated_dir"])
    review_dir = ensure_dir(Path(cfg["reviewed_dir"]) / "candidates")
    _write_review_readme(review_dir)

    media = normalize_video(cfg, force=force)
    normalized = Path(media["video"])
    duration = _duration(normalized)
    client = make_qwen_client(cfg)

    coarse_dir = processed / "frames" / "coarse_2fps"
    coarse_frames = extract_frames(
        normalized,
        coarse_dir,
        cfg["sampling"]["coarse_fps"],
        cfg["sampling"]["short_side"],
        "coarse",
        force=force,
    )
    write_frame_manifest(generated / "frame_manifest.jsonl", coarse_frames)
    coarse_results = _classify_frames(cfg, client, coarse_frames, generated / "qwen_frame_results.jsonl", force)
    coarse_candidates = merge_candidates(
        coarse_frames,
        coarse_results,
        max_gap_seconds=float(cfg["detection"]["max_gap_seconds"]),
        expand_seconds=float(cfg["detection"]["clip_expand_seconds"]),
        min_duration=float(cfg["detection"]["min_clip_seconds"]),
        duration=duration,
    )
    write_jsonl(generated / "coarse_candidates.jsonl", coarse_candidates)

    refined = []
    for coarse in coarse_candidates:
        fine_dir = processed / "frames" / "fine_8fps" / coarse["clip_id"]
        fine_frames = extract_frames(
            normalized,
            fine_dir,
            cfg["sampling"]["fine_fps"],
            cfg["sampling"]["short_side"],
            "fine",
            force=force,
            start=coarse["start"],
            end=coarse["end"],
        )
        fine_results = _classify_frames(cfg, client, fine_frames, None, force=True)
        fine_clips = merge_candidates(
            fine_frames,
            fine_results,
            max_gap_seconds=0.5,
            expand_seconds=float(cfg["detection"]["fine_expand_seconds"]),
            min_duration=float(cfg["detection"]["min_clip_seconds"]),
            duration=duration,
        )
        if fine_clips:
            for local_idx, fine in enumerate(fine_clips):
                clip_id = coarse["clip_id"] if local_idx == 0 else f"{coarse['clip_id']}_{local_idx:02d}"
                merged = {**fine, "clip_id": clip_id, "coarse_clip": coarse}
                refined.append(merged)
        else:
            refined.append({**coarse, "failure_reason": "no fine positive frames; kept coarse candidate"})

    write_jsonl(generated / "refined_clips.jsonl", refined)

    review_items = write_review_package(
        cfg,
        normalized,
        refined,
        force=force,
        model_path=client.model_path,
        model_version=client.model_version,
    )

    report = {
        "sample_id": cfg["sample_id"],
        "normalized_video": str(normalized),
        "coarse_frame_count": len(coarse_frames),
        "coarse_candidate_count": len(coarse_candidates),
        "refined_candidate_count": len(refined),
        "review_dir": str(review_dir),
        "candidates": review_items,
        "model_path": client.model_path,
        "model_version": client.model_version,
        "prompt_version": cfg["model"]["prompt_version"],
        "config_hash": cfg["config_hash"],
    }
    write_json(generated / "detect_report.json", report)
    return report
