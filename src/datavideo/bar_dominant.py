from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from .frames import extract_frames, write_contact_sheet
from .model_client import make_qwen_client
from .schemas import ensure_dir, read_jsonl, write_json, write_jsonl


SCENE_BREAK_STATES = {"chart_leaving", "transition", "non_chart", "uncertain"}
DEFAULT_START_TIMES_PATH = Path("data/raw/start_time.jsonl")


def _confidence(result: dict[str, Any]) -> float:
    try:
        return float(result.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _is_bar_positive(result: dict[str, Any]) -> bool:
    return bool(
        result.get("is_bar_chart_dominant_candidate")
        and result.get("bar_marks_visible")
        and result.get("bar_marks_dominant")
        and result.get("has_data_encoding_evidence")
    )


def _norm_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ").replace("-", " ")


def _norm_set(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {_norm_token(value) for value in values if _norm_token(value)}


def _bar_orientation(values: Any) -> str:
    tokens = _norm_set(values)
    joined = " ".join(sorted(tokens))
    has_horizontal = "horizontal" in joined
    has_vertical = "vertical" in joined
    if has_horizontal and not has_vertical:
        return "horizontal"
    if has_vertical and not has_horizontal:
        return "vertical"
    return "mixed_or_unknown"


def _dominant_chart_family(values: Any) -> set[str]:
    families = set()
    for token in _norm_set(values):
        if "bar" in token:
            families.add("bar")
        elif token:
            families.add(token)
    return families


def _identity_values(cand: dict[str, Any], field: str) -> set[str]:
    values = cand.get(field)
    if isinstance(values, list):
        return _norm_set(values)
    value = _norm_token(values)
    return {value} if value else set()


def _candidate_identity(row: dict[str, Any]) -> str:
    identity = _norm_token(row.get("chart_identity"))
    if identity:
        return identity
    parts = [
        _norm_token(row.get("chart_title")),
        ",".join(sorted(_norm_set(row.get("axis_labels")))),
        ",".join(sorted(_norm_set(row.get("category_labels")))),
        ",".join(sorted(_dominant_chart_family(row.get("chart_types")))),
        _bar_orientation(row.get("mark_types")),
    ]
    return "|".join(part for part in parts if part)


def _classify_bar_frames(
    cfg: dict[str, Any],
    client: Any,
    frames: list[dict[str, Any]],
    out_path: Path | None,
    force: bool,
) -> list[dict[str, Any]]:
    cached_rows = read_jsonl(out_path) if out_path and out_path.exists() and not force else []
    cached_by_id = {row["frame_id"]: row for row in cached_rows}
    missing_frames = [frame for frame in frames if frame["frame_id"] not in cached_by_id]
    new_rows = []
    max_n = int(cfg["model"].get("max_frames_per_call", 3))
    for i in range(0, len(missing_frames), max_n):
        group = missing_frames[i : i + max_n]
        response = client.detect_bar_dominant_frames([row["path"] for row in group])
        for frame in group:
            new_rows.append(
                {
                    "sample_id": cfg["sample_id"],
                    "frame_id": frame["frame_id"],
                    "timestamp": frame["timestamp"],
                    "image_path": frame["path"],
                    "result": response["result"],
                    "raw_response": response["raw_response"],
                    "model_status": response["model_status"],
                    "failure_reason": response["failure_reason"],
                    "model_path": client.model_path,
                    "model_version": client.model_version,
                    "prompt_version": f"{cfg['model']['prompt_version']}_bar_dominant_frame_v1",
                    "config_hash": cfg["config_hash"],
                }
            )
    by_id = {**cached_by_id, **{row["frame_id"]: row for row in new_rows}}
    rows = [by_id[frame["frame_id"]] for frame in frames if frame["frame_id"] in by_id]
    if out_path:
        persisted_rows = sorted(by_id.values(), key=lambda row: float(row.get("timestamp", 0.0)))
        write_jsonl(out_path, persisted_rows)
    return rows


def _load_search_anchor(cfg: dict[str, Any]) -> dict[str, Any] | None:
    search_cfg = cfg.get("target_search", {})
    path = Path(search_cfg.get("start_times_path", DEFAULT_START_TIMES_PATH))
    if not path.exists():
        return None
    matches = [row for row in read_jsonl(path) if row.get("video_id") == cfg["sample_id"]]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple start-time rows found for sample_id={cfg['sample_id']} in {path}")
    row = matches[0]
    try:
        clip_start = float(row["clip_start_sec"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid clip_start_sec for sample_id={cfg['sample_id']} in {path}") from exc
    configured_video = str(Path(cfg["video_path"]))
    anchor_video = str(Path(row.get("video_path", configured_video)))
    return {
        **row,
        "clip_start_sec": clip_start,
        "start_times_path": str(path),
        "video_path_matches_config": anchor_video == configured_video,
    }


def _frames_in_window(frames: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [frame for frame in frames if start <= float(frame["timestamp"]) <= end]


def _positive_reaches_right_boundary(
    frames: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    *,
    max_gap: float,
) -> bool:
    if not frames:
        return False
    by_id = {row["frame_id"]: row.get("result", {}) for row in result_rows}
    positive_times = [
        float(frame["timestamp"])
        for frame in frames
        if _is_bar_positive(by_id.get(frame["frame_id"], {}))
    ]
    return bool(positive_times and float(frames[-1]["timestamp"]) - positive_times[-1] <= max_gap)


def _classify_search_window(
    cfg: dict[str, Any],
    client: Any,
    all_frames: list[dict[str, Any]],
    out_path: Path,
    force: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    anchor = _load_search_anchor(cfg)
    if anchor is None:
        results = _classify_bar_frames(cfg, client, all_frames, out_path, force)
        return all_frames, results, {"mode": "full_video", "anchor_found": False}

    search_cfg = cfg.get("target_search", {})
    before = float(search_cfg.get("before_seconds", 2.0))
    after = float(search_cfg.get("after_seconds", 20.0))
    extension = float(search_cfg.get("extension_seconds", 10.0))
    boundary_gap = float(search_cfg.get("boundary_positive_gap_seconds", 1.0))
    if before < 0 or after <= 0 or extension <= 0 or boundary_gap < 0:
        raise ValueError("target_search durations must be positive (before/boundary gap may be zero)")

    start = max(0.0, anchor["clip_start_sec"] - before)
    initial_end = anchor["clip_start_sec"] + after
    end = initial_end
    video_end = float(all_frames[-1]["timestamp"]) if all_frames else 0.0
    scanned_frames: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    extension_count = 0
    first_pass = True
    while True:
        scanned_frames = _frames_in_window(all_frames, start, end)
        results = _classify_bar_frames(cfg, client, scanned_frames, out_path, force and first_pass)
        first_pass = False
        if end >= video_end or not _positive_reaches_right_boundary(
            scanned_frames, results, max_gap=boundary_gap
        ):
            break
        end += extension
        extension_count += 1

    search_report = {
        "mode": "anchored_window",
        "anchor_found": True,
        "clip_start_sec": anchor["clip_start_sec"],
        "initial_window_start": round(start, 3),
        "initial_window_end": round(initial_end, 3),
        "final_window_end": round(min(end, video_end), 3),
        "extension_count": extension_count,
        "scanned_frame_count": len(scanned_frames),
        "start_times_path": anchor["start_times_path"],
        "anchor_video_path": anchor.get("video_path"),
        "video_path_matches_config": anchor["video_path_matches_config"],
    }
    return scanned_frames, results, search_report


def _frames_for_candidate_identity(frames: list[dict[str, Any]], candidate: dict[str, Any], max_frames: int) -> list[dict[str, Any]]:
    start = float(candidate.get("source_start", candidate["start"]))
    end = float(candidate.get("source_end", candidate["end"]))
    nearby = [row for row in frames if start <= float(row["timestamp"]) <= end]
    if len(nearby) <= max_frames:
        return nearby
    if max_frames <= 1:
        return [nearby[len(nearby) // 2]]
    picks = []
    for idx in range(max_frames):
        pos = round(idx * (len(nearby) - 1) / (max_frames - 1))
        picks.append(nearby[pos])
    return picks


def _enrich_bar_candidate_identities(
    cfg: dict[str, Any],
    client: Any,
    frames: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    max_n = int(cfg["model"].get("max_frames_per_call", 3))
    enriched = []
    for candidate in candidates:
        identity_frames = _frames_for_candidate_identity(frames, candidate, max_n)
        if not identity_frames:
            enriched.append(candidate)
            continue
        response = client.identify_bar_candidate_frames([row["path"] for row in identity_frames])
        result = response["result"]
        row = {
            **candidate,
            "chart_identities": [result["chart_identity"]] if result.get("chart_identity") else [],
            "chart_titles": [result["chart_title"]] if result.get("chart_title") else [],
            "axis_labels": result.get("axis_labels", []),
            "category_labels": result.get("category_labels", []),
            "chart_types": result.get("chart_types", candidate.get("chart_types", [])),
            "mark_types": result.get("mark_types", candidate.get("mark_types", [])),
            "identity_model_status": response["model_status"],
            "identity_failure_reason": response["failure_reason"],
            "identity_raw_response": response["raw_response"],
            "identity_source_frame_ids": [frame["frame_id"] for frame in identity_frames],
        }
        animation_cue = result.get("animation_cue")
        if animation_cue and animation_cue != "unknown":
            row["animation_cues"] = sorted(set(row.get("animation_cues", [])) | {animation_cue})
        enriched.append(row)
    return enriched


def select_bar_candidates(
    frame_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    *,
    max_gap: float = 1.0,
    expand: float = 0.5,
    min_duration: float = 1.0,
) -> list[dict[str, Any]]:
    by_id = {row["frame_id"]: row for row in result_rows}
    positives = []
    for frame in frame_rows:
        result = by_id.get(frame["frame_id"], {}).get("result", {})
        if _is_bar_positive(result):
            positives.append(
                {
                    **frame,
                    "confidence": _confidence(result),
                    "scene_state": result.get("scene_state", "uncertain"),
                    "animation_cue": result.get("animation_cue", "unknown"),
                    "chart_identity": _candidate_identity(result),
                    "chart_title": result.get("chart_title", ""),
                    "axis_labels": result.get("axis_labels", []),
                    "category_labels": result.get("category_labels", []),
                    "chart_types": result.get("chart_types", []),
                    "mark_types": result.get("mark_types", []),
                }
            )
    candidates = []
    if not positives:
        return candidates
    start = end = positives[0]["timestamp"]
    rows = [positives[0]]
    last = positives[0]["timestamp"]
    for row in positives[1:]:
        if row["timestamp"] - last <= max_gap:
            end = row["timestamp"]
            rows.append(row)
        else:
            candidates.append(_candidate(start, end, rows, len(candidates), expand, min_duration))
            start = end = row["timestamp"]
            rows = [row]
        last = row["timestamp"]
    candidates.append(_candidate(start, end, rows, len(candidates), expand, min_duration))
    return candidates


def _candidate(
    start: float,
    end: float,
    rows: list[dict[str, Any]],
    idx: int,
    expand: float,
    min_duration: float,
) -> dict[str, Any]:
    s = max(0.0, start - expand)
    e = end + expand
    if e - s < min_duration:
        pad = (min_duration - (e - s)) / 2
        s = max(0.0, s - pad)
        e += pad
    confidences = [float(row.get("confidence", 0.0) or 0.0) for row in rows]
    return {
        "clip_id": f"bar_candidate_{idx:03d}",
        "start": round(s, 3),
        "end": round(e, 3),
        "source_start": round(start, 3),
        "source_end": round(end, 3),
        "confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        "positive_frame_count": len(rows),
        "scene_states": sorted({row.get("scene_state", "uncertain") for row in rows}),
        "animation_cues": sorted({row.get("animation_cue", "unknown") for row in rows}),
        "chart_identities": sorted({row.get("chart_identity", "") for row in rows if row.get("chart_identity")}),
        "chart_titles": sorted({row.get("chart_title", "") for row in rows if row.get("chart_title")}),
        "axis_labels": sorted({item for row in rows for item in row.get("axis_labels", [])}),
        "category_labels": sorted({item for row in rows for item in row.get("category_labels", [])}),
        "chart_types": sorted({item for row in rows for item in row.get("chart_types", [])}),
        "mark_types": sorted({item for row in rows for item in row.get("mark_types", [])}),
    }


def _merge_block_reasons(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    reasons = []
    left_identities = _identity_values(left, "chart_identities")
    right_identities = _identity_values(right, "chart_identities")
    if left_identities and right_identities and left_identities.isdisjoint(right_identities):
        reasons.append("different_chart_identity")

    left_titles = _identity_values(left, "chart_titles")
    right_titles = _identity_values(right, "chart_titles")
    if left_titles and right_titles and left_titles.isdisjoint(right_titles):
        reasons.append("title_changed")

    left_axes = _identity_values(left, "axis_labels")
    right_axes = _identity_values(right, "axis_labels")
    if left_axes and right_axes and left_axes.isdisjoint(right_axes):
        reasons.append("axis_labels_changed")

    left_categories = _identity_values(left, "category_labels")
    right_categories = _identity_values(right, "category_labels")
    if left_categories and right_categories and left_categories.isdisjoint(right_categories):
        reasons.append("category_set_changed")

    left_families = _dominant_chart_family(left.get("chart_types", []))
    right_families = _dominant_chart_family(right.get("chart_types", []))
    if left_families and right_families and left_families.isdisjoint(right_families):
        reasons.append("chart_type_changed")

    left_orientation = _bar_orientation(left.get("mark_types", []))
    right_orientation = _bar_orientation(right.get("mark_types", []))
    if (
        left_orientation != "mixed_or_unknown"
        and right_orientation != "mixed_or_unknown"
        and left_orientation != right_orientation
    ):
        reasons.append("bar_orientation_changed")

    states = set(left.get("scene_states", [])) | set(right.get("scene_states", []))
    if states & SCENE_BREAK_STATES:
        reasons.append("scene_break_state")

    cues = {_norm_token(cue) for cue in left.get("animation_cues", []) + right.get("animation_cues", [])}
    if any("scene" in cue or "cut" in cue or "transition" in cue for cue in cues):
        reasons.append("global_scene_change_cue")
    return reasons


def _new_merged_candidate(cand: dict[str, Any], idx: int, reason: str, block_reasons: list[str] | None = None) -> dict[str, Any]:
    row = {
        "clip_id": f"bar_merged_{idx:03d}",
        "start": cand["start"],
        "end": cand["end"],
        "candidate_ids": [cand["clip_id"]],
        "candidates": [cand],
        "merge_reasons": [reason],
        "chart_identities": list(cand.get("chart_identities", [])),
        "chart_titles": list(cand.get("chart_titles", [])),
        "axis_labels": list(cand.get("axis_labels", [])),
        "category_labels": list(cand.get("category_labels", [])),
        "scene_states": list(cand.get("scene_states", [])),
        "animation_cues": list(cand.get("animation_cues", [])),
        "chart_types": list(cand.get("chart_types", [])),
        "mark_types": list(cand.get("mark_types", [])),
    }
    if block_reasons:
        row["merge_block_reasons"] = block_reasons
    return row


def _load_or_build_frame_manifest(cfg: dict[str, Any], generated_root: Path) -> list[dict[str, Any]]:
    frame_manifest = generated_root / "frame_manifest.jsonl"
    if frame_manifest.exists():
        return read_jsonl(frame_manifest)

    processed = Path(cfg["processed_dir"])
    source_manifest = processed / "frame_manifest.jsonl"
    if source_manifest.exists():
        frames = read_jsonl(source_manifest)
        write_jsonl(frame_manifest, frames)
        return frames

    coarse_dir = processed / "frames" / "coarse_2fps"
    fps = float(cfg.get("sampling", {}).get("coarse_fps", 2))
    frames = []
    for idx, path in enumerate(sorted(coarse_dir.glob("coarse_*.jpg"))):
        frames.append(
            {
                "frame_id": path.stem,
                "path": str(path),
                "timestamp": idx / fps,
                "fps": fps,
                "sample_type": "coarse",
            }
        )
    if not frames:
        raise FileNotFoundError(f"No frame manifest or coarse frames found under {processed}")
    write_jsonl(frame_manifest, frames)
    return frames


def merge_bar_candidates(candidates: list[dict[str, Any]], *, max_gap: float = 2.0) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for cand in sorted(candidates, key=lambda row: row["start"]):
        if not merged or cand["start"] - merged[-1]["end"] > max_gap:
            merged.append(_new_merged_candidate(cand, len(merged), "new_bar_dominant_scene"))
            continue
        current = merged[-1]
        block_reasons = _merge_block_reasons(current, cand)
        if block_reasons:
            merged.append(_new_merged_candidate(cand, len(merged), "new_chart_identity", block_reasons))
            continue
        current["end"] = max(current["end"], cand["end"])
        current["candidate_ids"].append(cand["clip_id"])
        current["candidates"].append(cand)
        current["merge_reasons"].append("gap<=2s_same_chart_identity_continuation")
        for field in [
            "chart_identities",
            "chart_titles",
            "axis_labels",
            "category_labels",
            "scene_states",
            "animation_cues",
            "chart_types",
            "mark_types",
        ]:
            current[field] = sorted(set(current.get(field, [])) | set(cand.get(field, [])))
    for row in merged:
        confidences = [float(c.get("confidence", 0.0) or 0.0) for c in row["candidates"]]
        row["start"] = round(row["start"], 3)
        row["end"] = round(row["end"], 3)
        row["duration"] = round(row["end"] - row["start"], 3)
        row["confidence"] = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        row["animation_cues"] = sorted({cue for c in row["candidates"] for cue in c.get("animation_cues", [])})
        row["chart_identities"] = sorted({identity for c in row["candidates"] for identity in c.get("chart_identities", [])})
        row["chart_titles"] = sorted({title for c in row["candidates"] for title in c.get("chart_titles", [])})
        row["axis_labels"] = sorted({label for c in row["candidates"] for label in c.get("axis_labels", [])})
        row["category_labels"] = sorted({label for c in row["candidates"] for label in c.get("category_labels", [])})
        row["chart_types"] = sorted({t for c in row["candidates"] for t in c.get("chart_types", [])})
        row["mark_types"] = sorted({t for c in row["candidates"] for t in c.get("mark_types", [])})
    return merged


def run_bar_dominant_pipeline(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    generated_root = ensure_dir(Path(cfg["generated_dir"]))
    if force and generated_root.exists():
        for path in [
            generated_root / "bar_candidates.jsonl",
            generated_root / "bar_merged_clips.jsonl",
            generated_root / "final_bar_clips.jsonl",
            generated_root / "qwen_bar_frame_results.jsonl",
            generated_root / "bar_merged_clips_pre_review.jsonl",
            generated_root / "bar_merged_clips_reviewed.jsonl",
            generated_root / "bar_dominant_report.json",
        ]:
            if path.exists():
                path.unlink()
        for path in generated_root.glob("bar_final_*"):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        old_root = generated_root / "bar_dominant"
        if old_root.exists():
            shutil.rmtree(old_root)
    normalized = Path(cfg["processed_dir"]) / "normalized.mp4"
    all_frames = _load_or_build_frame_manifest(cfg, generated_root)
    client = make_qwen_client(cfg)

    frames, frame_results, search_report = _classify_search_window(
        cfg,
        client,
        all_frames,
        generated_root / "qwen_bar_frame_results.jsonl",
        force,
    )
    positive_frame_count = sum(
        1 for row in frame_results if _is_bar_positive(row.get("result", {}))
    )
    search_report["positive_frame_count"] = positive_frame_count
    search_report["status"] = (
        "positive_frames_found" if positive_frame_count else "no_positive_in_search_window"
    )
    candidates = select_bar_candidates(frames, frame_results)
    candidates = _enrich_bar_candidate_identities(cfg, client, frames, candidates)
    write_jsonl(generated_root / "bar_candidates.jsonl", candidates)
    merged = merge_bar_candidates(candidates)

    reviewed = []
    final = []
    with tempfile.TemporaryDirectory(prefix="bar_dominant_review_") as tmp:
        tmp_root = Path(tmp)
        for idx, clip in enumerate(merged):
            merged_id = f"bar_merged_{idx:03d}"
            clip["clip_id"] = merged_id
            review_dir = tmp_root / merged_id
            frame_dir = tmp_root / f"{merged_id}_frames"
            sheet_frames = extract_frames(
                normalized,
                frame_dir,
                2,
                cfg["sampling"]["short_side"],
                "bar_review",
                force=True,
                start=clip["start"],
                end=clip["end"],
            )
            labels = {row["frame_id"]: ",".join(clip.get("animation_cues", [])[:3]) for row in sheet_frames}
            scores = {row["frame_id"]: clip.get("confidence", 0.0) for row in sheet_frames}
            sheet_path = write_contact_sheet(
                sheet_frames,
                review_dir / f"{merged_id}_contact_sheet.jpg",
                max_cols=4,
                thumb_width=360,
                scores=scores,
                labels=labels,
            )
            review = client.review_bar_dominant_clip_contact_sheet(str(sheet_path))
            item = {
                **clip,
                "qwen_review": review["result"],
                "raw_response": review["raw_response"],
                "model_status": review["model_status"],
                "failure_reason": review["failure_reason"],
                "model_path": client.model_path,
                "model_version": client.model_version,
                "prompt_version": f"{cfg['model']['prompt_version']}_bar_dominant_clip_review_v1",
                "config_hash": cfg["config_hash"],
            }
            if review["result"].get("decision") == "trim":
                item = _trim_item(item)
            item["final_decision"] = "keep" if item["qwen_review"].get("decision") in {"keep", "trim"} else "exclude"
            reviewed.append(item)
            if item["final_decision"] == "keep":
                final_id = f"bar_final_{len(final):03d}"
                item["clip_id"] = final_id
                item["source_merged_clip_id"] = merged_id
                final.append(item)
    write_jsonl(generated_root / "bar_merged_clips.jsonl", reviewed)
    write_jsonl(generated_root / "final_bar_clips.jsonl", final)
    report = {
        "sample_id": cfg["sample_id"],
        "frame_count": len(frames),
        "available_frame_count": len(all_frames),
        "target_search": search_report,
        "bar_candidate_count": len(candidates),
        "bar_merged_count": len(merged),
        "final_bar_clip_count": len(final),
        "output_dir": str(generated_root),
        "final_output": str(generated_root / "final_bar_clips.jsonl"),
        "model_path": client.model_path,
        "model_version": client.model_version,
        "config_hash": cfg["config_hash"],
    }
    write_json(generated_root / "bar_dominant_report.json", report)
    return report


def _trim_item(item: dict[str, Any]) -> dict[str, Any]:
    review = item["qwen_review"]
    try:
        suggested_start = float(review.get("suggested_start"))
        suggested_end = float(review.get("suggested_end"))
    except (TypeError, ValueError):
        return item
    if item["start"] <= suggested_start < suggested_end <= item["end"]:
        trimmed_start = suggested_start
        trimmed_end = suggested_end
    elif 0.0 <= suggested_start < suggested_end <= item["end"] - item["start"]:
        trimmed_start = item["start"] + suggested_start
        trimmed_end = item["start"] + suggested_end
    else:
        return item
    if trimmed_start < trimmed_end:
        item["original_start"] = item["start"]
        item["original_end"] = item["end"]
        item["start"] = round(trimmed_start, 3)
        item["end"] = round(trimmed_end, 3)
        item["duration"] = round(item["end"] - item["start"], 3)
    return item
