from __future__ import annotations

import re
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


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _numeric_year(value: Any) -> float | None:
    """Extract a 4-digit year (e.g. ``2000`` from ``2000-01``) if present."""
    match = re.search(r"(?:19|20)\d{2}", str(value or ""))
    return float(match.group(0)) if match else None


def reconcile_intent_with_data(
    animation: dict[str, Any],
    dynamic: dict[str, Any],
) -> dict[str, Any]:
    """Cross-check the animation/intent report against recovered dynamic data.

    The vision-based animation report can contradict the recovered data table
    (e.g. bars visually growing while the printed values decrease). When at
    least two data states with shared entities are available, this deterministic
    pass rewrites ``overall_description`` and ``major_actions`` from the
    first->last state deltas so the submitted intent never contradicts the data
    table. The original model fields are preserved and reconciliation metadata
    is added.
    """
    if not isinstance(animation, dict) or not isinstance(dynamic, dict):
        return animation
    states = dynamic.get("states")
    if not isinstance(states, list) or len(states) < 2:
        return animation

    groups: dict[str, list[dict[str, Any]]] = {}
    order: dict[str, float] = {}
    for row in states:
        if not isinstance(row, dict):
            continue
        key = str(row.get("state_key") or row.get("state_label") or row.get("state_id") or "")
        if not key:
            continue
        start = _as_float(row.get("state_start"))
        order[key] = min(order.get(key, start if start is not None else 0.0), start if start is not None else 0.0)
        groups.setdefault(key, []).append(row)
    # State keys may be year labels ("2000-01") or plain indices/timestamps.
    # Sort by parsed year first so "2010-11" cannot sort before "2000-01";
    # non-year keys fall back to their state start timestamp.
    keys = sorted(
        groups,
        key=lambda item: (
            _numeric_year(item) is None,
            _numeric_year(item) or 0.0,
            order[item],
            item,
        ),
    )
    if len(keys) < 2:
        return animation
    first_key, last_key = keys[0], keys[-1]

    def by_entity(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            entity_id = str(row.get("entity_id") or "")
            if not entity_id or entity_id == "unknown":
                continue
            value = _as_float(row.get("value"))
            if value is None:
                continue
            current = out.get(entity_id)
            current_conf = _as_float((current or {}).get("confidence")) or 0.0
            row_conf = _as_float(row.get("confidence")) or 0.0
            if current is None or row_conf > current_conf:
                out[entity_id] = row
        return out

    first_map = by_entity(groups[first_key])
    last_map = by_entity(groups[last_key])
    common = [entity_id for entity_id in first_map if entity_id in last_map]
    if not common:
        return animation

    deltas = []
    for entity_id in common:
        first_value = _as_float(first_map[entity_id].get("value"))
        last_value = _as_float(last_map[entity_id].get("value"))
        if first_value is None or last_value is None or abs(last_value - first_value) < 1e-9:
            continue
        deltas.append(
            {
                "entity_id": entity_id,
                "entity": str(first_map[entity_id].get("entity") or entity_id),
                "first_value": first_value,
                "last_value": last_value,
                "direction": "up" if last_value > first_value else "down",
            }
        )
    if not deltas:
        return animation

    metric = str(first_map[common[0]].get("metric") or "指标")
    unit = str(first_map[common[0]].get("unit") or "")
    placeholder_metrics = {"", "指标", "value", "Value", "metric", "Metric", "unknown"}
    has_metric = metric.strip() not in placeholder_metrics

    def group_start(rows: list[dict[str, Any]]) -> float | None:
        starts = [start for start in (_as_float(row.get("state_start")) for row in rows) if start is not None]
        return min(starts) if starts else None

    first_start = group_start(groups[first_key])
    last_start = group_start(groups[last_key])
    evidence = [ts for ts in (first_start, last_start) if ts is not None]
    evidence = [round(ts, 3) for ts in evidence]

    chart_type = str(animation.get("target_chart_type") or "")
    upward_action = "line_draw_upward" if chart_type in {"line", "area"} else "bar_grow"
    downward_action = "line_draw_downward" if chart_type in {"line", "area"} else "bar_shrink"

    def state_label(key: str) -> str:
        return f"{key}年" if str(key).isdigit() else str(key)

    def describe(items: list[dict[str, Any]], action: str, verb: str) -> dict[str, Any]:
        if len(items) == 1:
            item = items[0]
            metric_part = f"的{metric}" if has_metric else ""
            description = (
                f"从{state_label(first_key)}到{state_label(last_key)}，"
                f"{item['entity']}{metric_part}由{item['first_value']:g}{unit}"
                f"变为{item['last_value']:g}{unit}（{verb}）。"
            )
        else:
            names = "、".join(item["entity"] for item in items)
            metric_part = f"的{metric}" if has_metric else ""
            description = (
                f"从{state_label(first_key)}到{state_label(last_key)}，"
                f"{names}{metric_part}整体{verb}。"
            )
        return {"action": action, "description": description, "evidence_timestamps": evidence}

    upward = [delta for delta in deltas if delta["direction"] == "up"]
    downward = [delta for delta in deltas if delta["direction"] == "down"]
    actions = []
    if upward:
        actions.append(describe(upward, upward_action, "上升"))
    if downward:
        actions.append(describe(downward, downward_action, "下降"))
    if not actions:
        return animation

    unique_entities = {item["entity"] for item in deltas}
    if len(unique_entities) == 1:
        subject = next(iter(unique_entities))
        verb = "上升" if upward else "下降"
        overall = f"从{state_label(first_key)}到{state_label(last_key)}，{subject}整体{verb}。"
    elif upward and downward:
        overall = f"从{state_label(first_key)}到{state_label(last_key)}，{metric}的变化方向不一致：部分实体上升、部分实体下降。"
    elif upward:
        overall = f"从{state_label(first_key)}到{state_label(last_key)}，{metric}整体上升。"
    else:
        overall = f"从{state_label(first_key)}到{state_label(last_key)}，{metric}整体下降。"

    corrected = {
        **animation,
        "is_target_chart_related": True,
        "overall_description": overall,
        "major_actions": actions,
        "reconciled_with_data": True,
        "data_state_keys": [first_key, last_key],
        "data_direction": "mixed" if upward and downward else ("increase" if upward else "decrease"),
        "data_delta_count": len(deltas),
        "reconciliation_note": "direction derived from recovered data first->last states",
    }
    return corrected


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
    max_frames = max(2, int(animation_cfg.get("max_frames", 6) or 6))
    if len(sampled_frames) > max_frames:
        picks = sorted(
            {
                round(index * (len(sampled_frames) - 1) / (max_frames - 1))
                for index in range(max_frames)
            }
        )
        sampled_frames = [sampled_frames[i] for i in picks]
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
