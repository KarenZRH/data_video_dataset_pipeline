from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from datavideo.frames import extract_frames, write_frame_manifest
from datavideo.keyframes import _clamp01, _image_motion_scores, extract_still
from datavideo.media import ffprobe, normalize_video
from datavideo.schemas import ensure_dir, read_json, read_jsonl, write_csv, write_json, write_jsonl
from datavideo.svg_trace import trace_svg
from datavideo.visual_provenance import assert_qwen_visual_inputs

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


def _safe_still_timestamp(timestamp: float, duration: float, cfg: dict[str, Any]) -> float:
    margin = float(cfg.get("keyframes", {}).get("tail_frame_min_margin_seconds", 0.25))
    return max(0.0, min(float(timestamp), max(0.0, duration - margin)))


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
        "source_time_range": {"start": row.get("start_seconds", row.get("start_time")), "end": row.get("end_seconds", row.get("end_time"))},
        "clip_duration_seconds": round(duration, 3),
        "video_role": "visual_clip",
    }


def _heuristic_keyframe_score(motion_score: float) -> dict[str, Any]:
    staticness = 1.0 - motion_score
    return {
        "target_chart_type_match": True,
        "same_chart": True,
        "scene_change_or_title_card": False,
        "scene_change": False,
        "structure_complete": True,
        "complete_chart": True,
        "final_or_most_complete_state": True,
        "data_marks_readable": True,
        "printed_text_readable": False,
        "labels_readable": False,
        "edge_crop_or_occlusion": False,
        "has_directly_printed_values": False,
        "staticness": staticness,
        "completeness": 0.65,
        "state_finality": 0.65,
        "edge_integrity": 1.0,
        "data_text_visibility": 0.5,
        "chart_identity_consistency": 0.75,
        "data_extraction_suitability": 0.55,
        "motion_score": motion_score,
        "state_summary": "heuristic fallback",
        "reason": "heuristic fallback from sampled clip frame and image motion",
    }


_CHART_TYPE_TERMS = {
    "map": {"map", "geographic", "region"},
    "bar": {"bar", "bars"},
    "line": {"line"},
    "area": {"area"},
    "donut": {"donut"},
    "pie": {"pie"},
    "timeline": {"timeline", "time line", "event", "events"},
    "treemap": {"treemap", "tree map", "rectangular", "rectangle", "segments"},
    "scatter": {"scatter", "points"},
    "sankey": {"sankey", "flow", "flows", "nodes"},
    "pictograph": {"pictograph", "icon", "icons"},
    "combined": {"combined"},
}


def _reason_type_penalty(score: dict[str, Any], chart_type: str) -> float:
    text = f"{score.get('reason', '')} {score.get('state_summary', '')}".lower()
    if not text:
        return 0.0
    target_terms = _CHART_TYPE_TERMS.get(chart_type, {chart_type})
    other_terms = set().union(*[terms for key, terms in _CHART_TYPE_TERMS.items() if key != chart_type])
    mentions_target = any(term in text for term in target_terms)
    mentions_other = any(term in text for term in other_terms)
    if mentions_other and not mentions_target:
        return 3.0
    return 0.0


def _combined_keyframe_score(score: dict[str, Any]) -> float:
    return (
        1.8 * _clamp01(score.get("completeness"))
        + 1.4 * _clamp01(score.get("state_finality"))
        + 1.2 * _clamp01(score.get("edge_integrity"))
        + 0.9 * _clamp01(score.get("data_text_visibility"))
        + 0.7 * _clamp01(score.get("chart_identity_consistency"))
        + (1.0 if score.get("target_chart_type_match", score.get("same_chart")) else -4.0)
        + (0.8 if score.get("structure_complete", score.get("complete_chart")) else -2.0)
        + (0.7 if score.get("final_or_most_complete_state") else -0.8)
        + (0.5 if score.get("data_marks_readable") else -1.0)
        + (0.4 if score.get("printed_text_readable", score.get("labels_readable")) else 0.0)
        + (0.3 if score.get("has_directly_printed_values") else 0.0)
        - (2.0 if score.get("edge_crop_or_occlusion") else 0.0)
        - (2.5 if score.get("scene_change_or_title_card", score.get("scene_change")) else 0.0)
        - 0.2 * _clamp01(score.get("motion_score"), default=1.0)
    )


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _state_enabled_chart_types(cfg: dict[str, Any]) -> set[str]:
    values = cfg.get("keyframes", {}).get("state_enabled_chart_types", [])
    return {str(value).lower() for value in _as_list(values)}


def _prefer_late_chart_types(cfg: dict[str, Any]) -> set[str]:
    values = cfg.get("keyframes", {}).get("prefer_late_chart_types", ["bar"])
    return {str(value).lower() for value in _as_list(values)}


def _selection_rank(row: dict[str, Any], chart_type: str, cfg: dict[str, Any]) -> tuple[bool, bool, bool, bool, bool, float, float, bool, float, float]:
    score = row["score"]
    duration = float(row.get("clip_duration", 0.0))
    timestamp = float(row["timestamp"])
    time_position = timestamp / duration if duration else 0.0
    late_priority = time_position if chart_type.lower() in _prefer_late_chart_types(cfg) else 0.0
    return (
        bool(score.get("target_chart_type_match", score.get("same_chart"))),
        not bool(score.get("scene_change_or_title_card", score.get("scene_change"))),
        bool(score.get("structure_complete", score.get("complete_chart"))),
        not bool(score.get("edge_crop_or_occlusion")),
        bool(score.get("final_or_most_complete_state")),
        _clamp01(score.get("completeness")),
        _clamp01(score.get("state_finality")),
        bool(score.get("data_marks_readable")),
        late_priority,
        float(row["combined_score"]),
    )


def _add_tail_candidate_frames(
    normalized_video: str | Path,
    frame_dir: Path,
    frames: list[dict[str, Any]],
    duration: float,
    fps: float,
    cfg: dict[str, Any],
    force: bool,
) -> list[dict[str, Any]]:
    min_tail_margin = float(cfg.get("keyframes", {}).get("tail_frame_min_margin_seconds", 0.25))
    offsets = [float(x) for x in (0.75, 0.5, min_tail_margin)]
    existing_times = [float(frame["timestamp"]) for frame in frames]
    rows = list(frames)
    for idx, offset in enumerate(offsets, start=1):
        timestamp = max(0.0, min(duration - min_tail_margin, duration - offset))
        if any(abs(timestamp - t) <= (0.5 / fps) for t in existing_times):
            continue
        path = frame_dir / f"keyframe_candidate_tail_{idx:02d}.jpg"
        extract_still(normalized_video, timestamp, path, force=force)
        rows.append(
            {
                "frame_id": path.stem,
                "path": str(path),
                "timestamp": timestamp,
                "fps": fps,
                "sample_type": "keyframe_candidate_tail",
            }
        )
        existing_times.append(timestamp)
    return sorted(rows, key=lambda item: float(item["timestamp"]))


def _quality_rows(scored_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in scored_rows:
        score = row["score"]
        if not score.get("target_chart_type_match", score.get("same_chart")):
            continue
        if score.get("scene_change_or_title_card", score.get("scene_change")):
            continue
        if not score.get("structure_complete", score.get("complete_chart")):
            continue
        if score.get("edge_crop_or_occlusion"):
            continue
        rows.append(row)
    return rows or scored_rows


def _pick_evenly(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    if limit <= 1:
        return [rows[-1]]
    selected = []
    for idx in range(limit):
        pos = round(idx * (len(rows) - 1) / (limit - 1))
        selected.append(rows[pos])
    seen = set()
    unique = []
    for row in selected:
        if row["frame_id"] not in seen:
            unique.append(row)
            seen.add(row["frame_id"])
    return unique


def _select_state_rows(scored_rows: list[dict[str, Any]], primary: dict[str, Any], cfg: dict[str, Any], chart_type: str) -> list[dict[str, Any]]:
    if chart_type.lower() not in _state_enabled_chart_types(cfg):
        return []
    max_states = int(cfg.get("keyframes", {}).get("max_state_keyframes", 3))
    if max_states <= 0:
        return []
    rows = sorted(_quality_rows(scored_rows), key=lambda item: float(item["timestamp"]))
    if len(rows) <= 1:
        return [primary]
    candidates = _pick_evenly(rows, max_states)
    if primary["frame_id"] not in {row["frame_id"] for row in candidates}:
        others = [row for row in candidates if row["frame_id"] != primary["frame_id"]]
        candidates = _pick_evenly(sorted(others, key=lambda item: float(item["timestamp"])), max_states - 1) + [primary]
    return sorted(candidates, key=lambda item: float(item["timestamp"]))


def _select_clip_data_rows(scored_rows: list[dict[str, Any]], state_rows: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    max_frames = int(cfg.get("clip_data", {}).get("max_frames", 10))
    rows = sorted(_quality_rows(scored_rows), key=lambda item: float(item["timestamp"]))
    selected = _pick_evenly(rows, max(1, max_frames))
    by_id = {row["frame_id"]: row for row in selected}
    for row in state_rows:
        by_id[row["frame_id"]] = row
    return sorted(by_id.values(), key=lambda item: float(item["timestamp"]))[:max_frames]


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
    frame_dir = ensure_dir(Path(cfg.get("processed_root", "data/processed")) / clip_id / "visual_frames" / "keyframe_candidates")
    fps = _sample_fps(cfg)
    frames = extract_frames(
        normalized_video,
        frame_dir,
        fps,
        int(cfg.get("sampling", {}).get("short_side", 768)),
        "keyframe_candidate",
        force=force,
    )
    frames = _add_tail_candidate_frames(normalized_video, frame_dir, frames, duration, fps, cfg, force)
    if not frames:
        raise RuntimeError(f"No keyframe candidates sampled for {clip_id}")
    write_frame_manifest(frame_dir.parent / "keyframe_frame_manifest.jsonl", frames)
    assert_qwen_visual_inputs([frame["path"] for frame in frames])

    motion_scores = _image_motion_scores(frames)
    scorer = client or MultichartQwenClient(cfg)
    context = _clip_context(row, duration)
    scored_rows = []
    for frame in frames:
        motion_score = motion_scores.get(frame["frame_id"], 0.0)
        model_score = scorer.score_keyframe_candidate(frame["path"], context)
        score = model_score["result"]
        if model_score["model_status"] != "qwen":
            score = _heuristic_keyframe_score(motion_score)
            score["target_chart_type_match"] = False
            score["same_chart"] = False
            score["structure_complete"] = False
            score["complete_chart"] = False
            score["state_finality"] = 0.0
            score["reason"] = model_score["failure_reason"] or score["reason"]
        else:
            score["motion_score"] = max(_clamp01(score.get("motion_score"), default=1.0), motion_score)
        combined = _combined_keyframe_score(score) - _reason_type_penalty(score, str(row.get("chart_type", "")))
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

    chart_type = str(row.get("chart_type", ""))
    selected = max(scored_rows, key=lambda item: _selection_rank(item, chart_type, cfg))
    timestamp = _safe_still_timestamp(float(selected["timestamp"]), duration, cfg)
    asset = str(extract_still(normalized_video, timestamp, out_dir / "initial.png", force=True))
    state_rows = _select_state_rows(scored_rows, selected, cfg, chart_type)
    states_dir = ensure_dir(out_dir / "states")
    if force:
        for stale_state in states_dir.glob("state_*.png"):
            stale_state.unlink()
    states = []
    for idx, state_row in enumerate(state_rows, start=1):
        state_name = f"state_{idx:03d}"
        state_timestamp = _safe_still_timestamp(float(state_row["timestamp"]), duration, cfg)
        state_path = str(extract_still(normalized_video, state_timestamp, states_dir / f"{state_name}.png", force=True))
        states.append(
            {
                "name": state_name,
                "timestamp": state_timestamp,
                "asset": state_path,
                "source_frame_id": state_row["frame_id"],
                "combined_score": state_row["combined_score"],
                "state_summary": state_row["score"].get("state_summary"),
                "reason": state_row["score"].get("reason"),
            }
        )
    manifest = {
        "clip_id": clip_id,
        "chart_type": row["chart_type"],
        "timestamps": {"initial": timestamp},
        "assets": {"initial": asset, "states": [state["asset"] for state in states]},
        "states": states,
        "selection_method": "v2_simple_complete_final_state_keyframe",
        "source_video_role": "visual_clip",
        "source_video": str(normalized_video),
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
            "edge_crop_or_occlusion": False,
            "final_or_most_complete_state": True,
            "description": "Select a target-type frame with complete visible structure, no important edge crop, and a final or most complete chart state.",
        },
    }
    write_jsonl(out_dir / "keyframe_scores.jsonl", scored_rows)
    write_json(manifest_path, manifest)
    return manifest


def recover_clip_data(
    cfg: dict[str, Any],
    keyframes: dict[str, Any],
    row: dict[str, Any],
    out_dir: str | Path,
    *,
    client: MultichartQwenClient | None = None,
    force: bool = False,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    raw_path = out_dir / "chart_data_clip_raw.json"
    validation_path = out_dir / "chart_data_validation.json"
    csv_path = out_dir / "chart_data.csv"
    manual_csv_path = out_dir / "manual_data_stub.csv"
    events_path = out_dir / "data_events.jsonl"
    if raw_path.exists() and validation_path.exists() and not force:
        raw = read_json(raw_path)
        validation = read_json(validation_path)
        return {
            "data": raw.get("response", {}).get("data"),
            "metadata": raw.get("metadata", {}),
            "validation": validation,
            "csv_path": str(csv_path) if csv_path.exists() else None,
            "manual_csv_path": str(manual_csv_path) if manual_csv_path.exists() else None,
            "events_path": str(events_path) if events_path.exists() else None,
        }

    scorer = client or MultichartQwenClient(cfg)
    score_manifest = Path(keyframes.get("score_manifest", ""))
    scored_rows = read_jsonl(score_manifest) if score_manifest.exists() else []
    frame_rows = _select_clip_data_rows(scored_rows, scored_rows[:0], cfg) if scored_rows else []
    state_assets = {Path(state.get("asset", "")).resolve(): state for state in keyframes.get("states", [])}
    image_paths = []
    frame_context = []
    for idx, frame in enumerate(frame_rows, start=1):
        image_paths.append(frame["path"])
        frame_context.append(
            {
                "image_index": idx,
                "source_frame": frame["frame_id"],
                "time_seconds": round(float(frame["timestamp"]), 3),
                "state_summary": frame.get("score", {}).get("state_summary"),
            }
        )
    for state_path, state in state_assets.items():
        if str(state_path) not in {str(Path(path).resolve()) for path in image_paths} and state_path.exists():
            image_paths.append(str(state_path))
            frame_context.append(
                {
                    "image_index": len(image_paths),
                    "source_frame": state.get("source_frame_id"),
                    "time_seconds": round(float(state.get("timestamp", 0.0)), 3),
                    "state_summary": state.get("state_summary"),
                }
            )
    max_frames = int(cfg.get("clip_data", {}).get("max_frames", 10))
    if len(image_paths) > max_frames:
        paired = list(zip(image_paths, frame_context))
        paired = _pick_evenly(
            [{"path": path, "frame_context": context, "timestamp": context.get("time_seconds") or 0.0, "frame_id": str(idx)} for idx, (path, context) in enumerate(paired)],
            max_frames,
        )
        image_paths = [row["path"] for row in paired]
        frame_context = [row["frame_context"] for row in paired]
        for idx, context in enumerate(frame_context, start=1):
            context["image_index"] = idx
    if not image_paths:
        initial = Path(keyframes.get("assets", {}).get("initial", ""))
        if initial.exists():
            image_paths = [str(initial)]
            frame_context = [{"image_index": 1, "source_frame": keyframes.get("source_frame_id"), "time_seconds": keyframes.get("timestamps", {}).get("initial")}]
    if not image_paths:
        raise RuntimeError(f"No frames available for clip-level data recovery for {_clip_id(row)}")
    assert_qwen_visual_inputs(image_paths)

    context = {
        "clip_id": _clip_id(row),
        "chart_type": row.get("chart_type"),
        "title": row.get("raw_video_title"),
        "channel": row.get("channel"),
        "year": row.get("year"),
        "source_time_range": {"start": row.get("start_seconds", row.get("start_time")), "end": row.get("end_seconds", row.get("end_time"))},
        "video_role": "visual_clip",
        "rule": "Recover only directly printed values from the whole clip frame sequence; do not estimate geometric values.",
    }
    response = scorer.recover_clip_data(image_paths, context, frame_context)
    data = response["data"]
    rows = data.get("rows") if isinstance(data, dict) else []
    manual_rows = data.get("manual_stub_rows") if isinstance(data, dict) else []
    has_extractable = bool(data.get("has_extractable_data")) and bool(rows)

    if has_extractable:
        write_csv(csv_path, rows)
    elif csv_path.exists() and force:
        csv_path.unlink()
    if manual_rows:
        write_csv(manual_csv_path, manual_rows)
    elif manual_csv_path.exists() and force:
        manual_csv_path.unlink()
    write_jsonl(events_path, rows or manual_rows or [])

    metadata = {
        "title": data.get("title"),
        "chart_type": data.get("chart_type", row.get("chart_type")),
        "unit": data.get("unit"),
        "x_axis": data.get("x_axis"),
        "y_axis": data.get("y_axis"),
        "series": data.get("series", []),
        "visible_text": data.get("visible_text", []),
        "needs_manual_data": bool(data.get("needs_manual_data")) or bool(manual_rows),
        "model_status": response["model_status"],
        "failure_reason": response["failure_reason"],
        "skip_reason": data.get("skip_reason") if not has_extractable else None,
    }
    validation = {
        "valid_schema": isinstance(data, dict) and isinstance(rows, list),
        "has_extractable_data": has_extractable,
        "value_count": len(rows) if has_extractable else 0,
        "csv_path": str(csv_path) if has_extractable else None,
        "manual_csv_path": str(manual_csv_path) if manual_rows else None,
        "manual_stub_count": len(manual_rows) if isinstance(manual_rows, list) else 0,
        "source_frame_count": len(image_paths),
        "uncertain_fields": sorted(set(data.get("uncertain_fields", []))),
        "do_not_use_for_training_without_review": bool(data.get("uncertain_fields")) or bool(manual_rows),
    }
    write_json(
        raw_path,
        {
            "response": response,
            "metadata": metadata,
            "frame_context": frame_context,
            "image_paths": image_paths,
            "model_path": scorer.model_path,
            "prompt_version": cfg["model"]["prompt_version"],
        },
    )
    write_json(out_dir / "chart_metadata.json", metadata)
    write_json(validation_path, validation)
    return {
        "data": data,
        "metadata": metadata,
        "validation": validation,
        "csv_path": validation["csv_path"],
        "manual_csv_path": validation["manual_csv_path"],
        "events_path": str(events_path),
    }


def _cached_report(path: Path, row: dict[str, Any], force: bool) -> dict[str, Any] | None:
    if force or not path.exists():
        return None
    cached = read_json(path)
    cached_clip = cached.get("clip", {})
    expected_id = _clip_id(row)
    if cached_clip.get("clip_id") != expected_id or cached_clip.get("video_id") != row.get("video_id"):
        return None
    keyframe = Path(cached.get("keyframes", {}).get("assets", {}).get("initial", ""))
    trace_svg_path = Path(cached.get("trace", {}).get("trace_svg", ""))
    if keyframe.exists() and trace_svg_path.exists():
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
            trace = trace_svg(initial, clip_root, cfg, force=force)
            chart_data = recover_clip_data(cfg, keyframes, row, clip_root, client=client, force=force)

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
                "trace": trace,
                "chart_data": chart_data,
            }
            write_json(clip_report_path, report)
            failed_path = clip_root / "clip_report_failed.json"
            if failed_path.exists():
                failed_path.unlink()
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
    write_json(generated_root / "multichart_v2_run_report.json", run_report)
    write_jsonl(generated_root / "multichart_v2_clips.jsonl", refined_rows)
    return run_report
