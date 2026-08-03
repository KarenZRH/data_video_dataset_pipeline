from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from datavideo.frames import extract_frames, write_frame_manifest
from datavideo.keyframes import _clamp01, _image_motion_scores, extract_still
from datavideo.media import ffprobe, normalize_video
from datavideo.semantic import build_semantic_svg
from datavideo.schemas import ensure_dir, read_json, read_jsonl, write_csv, write_json, write_jsonl

from .qwen import MultichartQwenClient


def _clip_id(row: dict[str, Any]) -> str:
    return str(row.get("output_stem") or f"{row['chart_type']}_{row['chart_index']}")


def _processed_clip_cfg(cfg: dict[str, Any], row: dict[str, Any], clip_id: str) -> dict[str, Any]:
    return {
        **cfg,
        "sample_id": clip_id,
        "chart_type": row["chart_type"],
        "video_path": row["output_path"],
        "processed_dir": str(Path(cfg.get("processed_root", "data/processed")) / clip_id),
    }


def _duration_seconds(video: str | Path) -> float:
    return float(ffprobe(video)["format"]["duration"])


def _sample_fps(cfg: dict[str, Any]) -> float:
    requested = cfg.get("keyframes", {}).get("sample_fps", cfg.get("sampling", {}).get("fine_fps", 4))
    try:
        return max(1.0, min(8.0, float(requested)))
    except (TypeError, ValueError):
        return 4.0


def _clip_context(row: dict[str, Any], duration: float) -> dict[str, Any]:
    return {
        "clip_id": _clip_id(row),
        "chart_type": row.get("chart_type"),
        "title": row.get("raw_video_title"),
        "channel": row.get("channel"),
        "year": row.get("year"),
        "source_youtube_url": row.get("youtube_url"),
        "source_time_range": {"start": row.get("start_time"), "end": row.get("end_time")},
        "clip_duration_seconds": round(duration, 3),
    }


def _heuristic_keyframe_score(motion_score: float) -> dict[str, Any]:
    staticness = 1.0 - motion_score
    return {
        "same_chart": True,
        "scene_change": False,
        "complete_chart": True,
        "data_marks_readable": True,
        "labels_readable": False,
        "legend_or_axes_readable": False,
        "staticness": staticness,
        "completeness": 0.65,
        "chart_identity_consistency": 0.75,
        "data_extraction_suitability": 0.55,
        "motion_score": motion_score,
        "reason": "heuristic fallback from sampled clip frame and image motion",
    }


def _combined_keyframe_score(score: dict[str, Any]) -> float:
    return (
        1.6 * _clamp01(score.get("data_extraction_suitability"))
        + 1.4 * _clamp01(score.get("completeness"))
        + 1.1 * _clamp01(score.get("chart_identity_consistency"))
        + 0.7 * _clamp01(score.get("staticness"))
        + (0.7 if score.get("complete_chart") else -1.5)
        + (0.5 if score.get("data_marks_readable") else -1.0)
        + (0.25 if score.get("labels_readable") else 0.0)
        + (0.25 if score.get("legend_or_axes_readable") else 0.0)
        + (0.6 if score.get("same_chart") else -2.0)
        - (2.0 if score.get("scene_change") else 0.0)
        - 0.7 * _clamp01(score.get("motion_score"), default=1.0)
    )


def _selection_rank(row: dict[str, Any]) -> tuple[bool, bool, bool, bool, float, float, float, float]:
    score = row["score"]
    duration = float(row.get("clip_duration", 0.0))
    midpoint_distance = abs(float(row["timestamp"]) - duration / 2.0) if duration else 0.0
    return (
        bool(score.get("same_chart")),
        not bool(score.get("scene_change")),
        bool(score.get("complete_chart")),
        bool(score.get("data_marks_readable")),
        _clamp01(score.get("data_extraction_suitability")),
        float(row["combined_score"]),
        -_clamp01(score.get("motion_score"), default=1.0),
        -midpoint_distance,
    )


def select_keyframe(
    normalized_video: str | Path,
    row: dict[str, Any],
    out_dir: str | Path,
    cfg: dict[str, Any],
    *,
    client: MultichartQwenClient | None = None,
    force: bool = False,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    manifest_path = out_dir / "keyframe_manifest.json"
    clip_id = _clip_id(row)
    if manifest_path.exists() and not force:
        cached = read_json(manifest_path)
        initial = Path(cached.get("assets", {}).get("initial", ""))
        if cached.get("clip_id") == clip_id and initial.exists():
            return cached

    duration = _duration_seconds(normalized_video)
    frame_dir = ensure_dir(Path(cfg.get("processed_root", "data/processed")) / clip_id / "frames" / "keyframe_candidates")
    fps = _sample_fps(cfg)
    frames = extract_frames(
        normalized_video,
        frame_dir,
        fps,
        int(cfg.get("sampling", {}).get("short_side", 768)),
        "keyframe_candidate",
        force=force,
    )
    if not frames:
        raise RuntimeError(f"No keyframe candidates sampled for {clip_id}")
    write_frame_manifest(frame_dir.parent / "keyframe_frame_manifest.jsonl", frames)

    motion_scores = _image_motion_scores(frames)
    scorer = client or MultichartQwenClient(cfg)
    context = _clip_context(row, duration)
    scored_rows = []
    for frame in frames:
        motion_score = motion_scores.get(frame["frame_id"], 0.0)
        model_score = scorer.score_keyframe_candidate(frame["path"], context)
        score = model_score["result"]
        if model_score["model_status"] != "qwen" and not cfg.get("model", {}).get("allow_heuristic_fallback", False):
            raise RuntimeError(model_score["failure_reason"] or "Qwen keyframe scoring unavailable")
        if model_score["model_status"] != "qwen":
            score = _heuristic_keyframe_score(motion_score)
        else:
            score["motion_score"] = max(_clamp01(score.get("motion_score"), default=1.0), motion_score)
        combined = _combined_keyframe_score(score)
        scored_rows.append(
            {
                **frame,
                "clip_duration": duration,
                "score": score,
                "combined_score": round(combined, 4),
                "raw_response": model_score["raw_response"],
                "model_status": model_score["model_status"],
                "failure_reason": model_score["failure_reason"],
            }
        )

    selected = max(scored_rows, key=_selection_rank)
    timestamp = float(selected["timestamp"])
    asset = str(extract_still(normalized_video, timestamp, out_dir / "initial.png", force=force))
    manifest = {
        "clip_id": clip_id,
        "chart_type": row["chart_type"],
        "timestamps": {"initial": timestamp},
        "assets": {"initial": asset},
        "selection_method": "sampled_frame_multichart_static_data_keyframe",
        "source_frame_id": selected["frame_id"],
        "sample_fps": fps,
        "clip_context": context,
        "selected_score": selected["score"],
        "combined_score": selected["combined_score"],
        "score_manifest": str(out_dir / "keyframe_scores.jsonl"),
        "requirements": {
            "same_chart": True,
            "scene_change": False,
            "complete_chart": True,
            "data_marks_readable": True,
            "description": "Select the frame that best represents the chart as a static visualization for SVG tracing and data extraction across chart types.",
        },
    }
    write_jsonl(out_dir / "keyframe_scores.jsonl", scored_rows)
    write_json(manifest_path, manifest)
    return manifest


def recover_chart_data(
    cfg: dict[str, Any],
    image_path: str | Path,
    row: dict[str, Any],
    out_dir: str | Path,
    *,
    client: MultichartQwenClient | None = None,
    force: bool = False,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    raw_path = out_dir / "chart_data_raw.json"
    validation_path = out_dir / "chart_data_validation.json"
    csv_path = out_dir / "chart_data.csv"
    if raw_path.exists() and validation_path.exists() and not force:
        raw = read_json(raw_path)
        validation = read_json(validation_path)
        return {
            "data": raw.get("response", {}).get("data"),
            "metadata": raw.get("metadata", {}),
            "validation": validation,
            "csv_path": str(csv_path) if csv_path.exists() else None,
        }

    scorer = client or MultichartQwenClient(cfg)
    context = {
        "clip_id": _clip_id(row),
        "chart_type": row.get("chart_type"),
        "title": row.get("raw_video_title"),
        "channel": row.get("channel"),
        "year": row.get("year"),
    }
    response = scorer.recover_chart_data(str(image_path), context)
    data = response["data"]
    rows = data.get("rows") if isinstance(data, dict) else []
    has_extractable = bool(data.get("has_extractable_data")) and bool(rows)

    if has_extractable:
        write_csv(csv_path, rows)
    elif csv_path.exists() and force:
        csv_path.unlink()

    metadata = {
        "title": data.get("title"),
        "chart_type": data.get("chart_type", row.get("chart_type")),
        "unit": data.get("unit"),
        "x_axis": data.get("x_axis"),
        "y_axis": data.get("y_axis"),
        "series": data.get("series", []),
        "model_status": response["model_status"],
        "failure_reason": response["failure_reason"],
        "skip_reason": data.get("skip_reason") if not has_extractable else None,
    }
    validation = {
        "valid_schema": isinstance(data, dict) and isinstance(rows, list),
        "has_extractable_data": has_extractable,
        "value_count": len(rows) if has_extractable else 0,
        "csv_path": str(csv_path) if has_extractable else None,
        "uncertain_fields": sorted(set(data.get("uncertain_fields", []))),
        "do_not_use_for_training_without_review": bool(data.get("uncertain_fields")),
    }
    write_json(
        raw_path,
        {
            "response": response,
            "metadata": metadata,
            "model_path": scorer.model_path,
            "prompt_version": cfg["model"]["prompt_version"],
        },
    )
    write_json(out_dir / "chart_metadata.json", metadata)
    write_json(validation_path, validation)
    return {"data": data, "metadata": metadata, "validation": validation, "csv_path": validation["csv_path"]}


def _cached_report(path: Path, row: dict[str, Any], force: bool) -> dict[str, Any] | None:
    if force or not path.exists():
        return None
    cached = read_json(path)
    cached_clip = cached.get("clip", {})
    expected_id = _clip_id(row)
    if cached_clip.get("clip_id") != expected_id or cached_clip.get("video_id") != row.get("video_id"):
        return None
    keyframe = Path(cached.get("keyframes", {}).get("assets", {}).get("initial", ""))
    semantic_svg_path = Path(cached.get("semantic", {}).get("semantic_svg", ""))
    if keyframe.exists() and semantic_svg_path.exists():
        return cached
    return None


def run_pipeline(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    jsonl_path = Path(cfg.get("raw_clips_jsonl", "data/raw/datavideo_clips.jsonl"))
    processed_root = ensure_dir(cfg.get("processed_root", "data/processed"))
    generated_root = ensure_dir(cfg.get("generated_root", "data/generated"))
    rows = read_jsonl(jsonl_path)
    max_clips = cfg.get("max_clips")
    if max_clips is not None:
        rows = rows[: int(max_clips)]
    client = MultichartQwenClient(cfg)

    clip_reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in rows:
        clip_id = _clip_id(row)
        clip_root = ensure_dir(generated_root / clip_id)
        clip_report_path = clip_root / "clip_report.json"
        cached = _cached_report(clip_report_path, row, force)
        if cached is not None:
            clip_reports.append(cached)
            continue

        try:
            processed_cfg = _processed_clip_cfg({**cfg, "processed_root": str(processed_root)}, row, clip_id)
            media = normalize_video(processed_cfg, force=force)
            normalized_video = Path(media["video"])
            clip_video = clip_root / "clip.mp4"
            if force or not clip_video.exists():
                shutil.copy2(normalized_video, clip_video)

            keyframe_dir = ensure_dir(clip_root / "keyframes")
            keyframes = select_keyframe(
                normalized_video,
                row,
                keyframe_dir,
                {**cfg, "processed_root": str(processed_root)},
                client=client,
                force=force,
            )
            initial = keyframes["assets"]["initial"]
            semantic = build_semantic_svg(initial, clip_root, cfg, force=force)
            chart_data = recover_chart_data(cfg, initial, row, clip_root, client=client, force=force)

            clip_payload = {
                **row,
                "clip_id": clip_id,
                "processed_dir": str(processed_root / clip_id),
                "generated_dir": str(clip_root),
            }
            report = {
                "clip": clip_payload,
                "media": media,
                "clip_video": str(clip_video),
                "keyframes": keyframes,
                "semantic": semantic,
                "chart_data": chart_data,
            }
            write_json(clip_report_path, report)
            clip_reports.append(report)
        except Exception as exc:
            failure = {"clip_id": clip_id, "clip": row, "failure_reason": str(exc)}
            write_json(clip_root / "clip_report_failed.json", failure)
            failures.append(failure)

    refined_rows = [report["clip"] for report in clip_reports]
    run_report = {
        "source": str(jsonl_path),
        "clip_count": len(rows),
        "completed_clip_count": len(clip_reports),
        "failure_count": len(failures),
        "processed_root": str(processed_root),
        "generated_root": str(generated_root),
        "clips": clip_reports,
        "failures": failures,
        "config_hash": cfg.get("config_hash"),
    }
    write_json(generated_root / "multichart_run_report.json", run_report)
    write_jsonl(generated_root / "multichart_clips.jsonl", refined_rows)
    return run_report
