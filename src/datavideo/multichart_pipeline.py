from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

from datavideo.context import create_context_media
from .animation import detect_animation, reconcile_intent_with_data
from datavideo.chart_processors import SUPPORTED_PROCESSORS, detect_chart_type
from datavideo.cv_align import read_frame_title, read_series_label, run_cv_align, run_cv_align_line, reconcile_line_dynamic
from datavideo.cv_reconcile import reconcile_dynamic_data
from datavideo.cv_reconcile import write_dynamic_outputs
from datavideo.metadata import read_clip_rows
from datavideo.narration import transcribe_context_audio
from datavideo.semantic_render import metadata_from_dynamic, render_data_driven, render_data_driven_line, render_dynamic_states, prefer_frame_visible_title, frame_title_status, resolve_render_title
from datavideo.schemas import ensure_dir, read_json, write_json, write_jsonl

from .multichart_assets import (
    _keyframe_timestamp,
    build_semantic_state_svgs,
    recover_clip_data,
    select_keyframe,
)
from .multichart_qwen import MultichartQwenClient


def _clip_id(row: dict[str, Any]) -> str:
    return str(row.get("output_stem") or row.get("clip_id") or f"{row.get('chart_type') or 'chart'}_{row.get('chart_index') or 0}")


def _series_label_from_title(title: Any) -> str:
    text = str(title or "").strip().strip("\"'`.,")
    if not text:
        return ""
    for separator in (" in ", " for ", " from ", " over "):
        if separator in text:
            text = text.split(separator, 1)[0]
            break
    return text.strip().strip("'\"`").strip()[:60]


def _reference_clip_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "clip_id": _clip_id(row),
        "source_video": row.get("output_path"),
    }


def _load_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return read_clip_rows(cfg)


def _selected_keyframe_path(keyframes: dict[str, Any]) -> Path | None:
    assets = keyframes.get("assets") if isinstance(keyframes.get("assets"), dict) else {}
    selected = assets.get("selected")
    if selected:
        path = Path(selected)
        if path.exists():
            return path
    states = keyframes.get("states") if isinstance(keyframes.get("states"), list) else []
    for state in states:
        if isinstance(state, dict) and state.get("asset"):
            path = Path(state["asset"])
            if path.exists():
                return path
    return None


def _cv_align_enabled(row: dict[str, Any], cfg: dict[str, Any]) -> bool:
    align_cfg = cfg.get("cv_align") if isinstance(cfg.get("cv_align"), dict) else {}
    if align_cfg.get("enabled") is False:
        return False
    chart_type = str(row.get("chart_type") or "").lower()
    # combined multichart clips frequently contain bar charts, so keep CV
    # alignment enabled for them too.
    return "bar" in chart_type or "combined" in chart_type


def _write_candidate_report(
    clip_root: Path,
    row: dict[str, Any],
    media: dict[str, Any],
    intervals: dict[str, Any],
    asr_report: dict[str, Any],
    keyframes: dict[str, Any],
    animation: dict[str, Any],
    semantic: dict[str, Any],
    chart_data: dict[str, Any],
    semantic_state_svgs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    clip_payload = _reference_clip_metadata(row)
    clip_payload["animation_description"] = animation.get("overall_description")
    clip_payload["animation_action_count"] = len(animation.get("major_actions", [])) if isinstance(animation.get("major_actions"), list) else 0
    clip_payload["animation_confidence"] = animation.get("confidence")
    clip_payload["is_target_chart_related"] = animation.get("is_target_chart_related")
    clip_report = {
        "clip": clip_payload,
        "context": media,
        "intervals": intervals,
        "asr": asr_report,
        "clip_video": str(clip_root / "clip.mp4"),
        "keyframes": keyframes,
        "animation_detection": animation,
        "semantic": semantic,
        "semantic_state_svgs": semantic_state_svgs or {},
        "chart_data": chart_data,
    }
    write_json(clip_root / "clip_report.json", clip_report)
    return clip_report


def run_context_pipeline(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    rows = _load_rows(cfg)
    processed_root = ensure_dir(cfg.get("processed_root", "data/processed"))
    results = []
    failures = []
    for row in rows:
        clip_id = _clip_id(row)
        try:
            media = create_context_media({**cfg, "processed_root": str(processed_root)}, row, force=force)
            results.append({"clip_id": clip_id, **media})
        except Exception as exc:
            failure = {"clip_id": clip_id, "failure_reason": str(exc)}
            failures.append(failure)
            write_json(processed_root / clip_id / "context_failed.json", failure)
    report = {"clip_count": len(rows), "completed_count": len(results), "failure_count": len(failures), "clips": results, "failures": failures}
    write_json(processed_root / "multichart_v2_context_report.json", report)
    return report


def run_asr_pipeline(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    rows = _load_rows(cfg)
    processed_root = ensure_dir(cfg.get("processed_root", "data/processed"))
    results = []
    failures = []
    for row in rows:
        clip_id = _clip_id(row)
        processed_dir = ensure_dir(processed_root / clip_id)
        try:
            if not (processed_dir / "intervals.json").exists() or not (processed_dir / "context_audio_16k_mono.wav").exists():
                create_context_media({**cfg, "processed_root": str(processed_root)}, row, force=force)
            intervals = read_json(processed_dir / "intervals.json")
            report = transcribe_context_audio(
                cfg,
                clip_id,
                processed_dir / "context_audio_16k_mono.wav",
                intervals,
                processed_dir,
                force=force,
            )
            results.append(report)
        except Exception as exc:
            failure = {"clip_id": clip_id, "failure_reason": str(exc)}
            failures.append(failure)
            write_json(processed_dir / "narration" / "asr_failed.json", failure)
    report = {"clip_count": len(rows), "completed_count": len(results), "failure_count": len(failures), "clips": results, "failures": failures}
    write_json(processed_root / "multichart_v2_asr_report.json", report)
    return report


def _as_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _line_metadata_from_dynamic(
    dynamic: dict[str, Any],
    *,
    title: str,
    unit: str = "",
    x_labels: list[str] | None = None,
) -> dict[str, Any]:
    series_values: dict[str, list[float]] = {}
    series_x_labels: dict[str, list[str]] = {}
    for row_item in dynamic.get("states") or []:
        if not isinstance(row_item, dict):
            continue
        value = _as_number(row_item.get("value"))
        if value is None:
            continue
        entity = str(row_item.get("entity") or row_item.get("entity_id") or "series")
        series_values.setdefault(entity, []).append(value)
        series_x_labels.setdefault(entity, []).append(str(row_item.get("state_key") or row_item.get("x_label") or ""))
    return {
        "title": title,
        "unit": unit,
        "chart_type": "line",
        "x_labels": x_labels or [],
        "series": [
            {"name": name, "values": values, "x_labels": series_x_labels.get(name, [])}
            for name, values in series_values.items()
        ],
    }


def _safe_state_key(key: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "-", key.lower()).strip("-")
    return safe or "state"


def _state_groups(dynamic: dict[str, Any] | None) -> list[tuple[str, str, list[dict[str, Any]]]]:
    """Return ordered ``(state_key, state_label, rows)`` groups from dynamic data."""
    states = dynamic.get("states") if isinstance(dynamic, dict) else []
    if not isinstance(states, list):
        return []
    # Rows without an explicit state key are static chart marks recovered from
    # one visual state, not separate video states. Keep them as one flat sample.
    if states and not any(isinstance(row, dict) and row.get("state_key") not in (None, "") for row in states):
        return []
    chart_type = str(dynamic.get("chart_type") or "").lower() if isinstance(dynamic, dict) else ""
    if chart_type in {"line", "area", "scatter"}:
        visual_ranges = set()
        for row in states:
            if not isinstance(row, dict):
                continue
            if str(row.get("source_type") or "") not in {"visual", "visual_frame_align"}:
                continue
            visual_ranges.add((_as_number(row.get("state_start")), _as_number(row.get("state_end"))))
        if len(visual_ranges) <= 1:
            return []
    groups: dict[str, list[dict[str, Any]]] = {}
    order: dict[str, float] = {}
    labels: dict[str, str] = {}
    for row in states:
        if not isinstance(row, dict):
            continue
        key = str(row.get("state_key") or row.get("state_label") or row.get("state_id") or "")
        if not key:
            continue
        start = _as_number(row.get("state_start"))
        groups.setdefault(key, []).append(row)
        order[key] = min(order.get(key, start if start is not None else 0.0), start if start is not None else 0.0)
        labels[key] = str(row.get("state_label") or row.get("state_key") or labels.get(key, key))
    return [
        (key, labels.get(key, key), groups[key])
        for key in sorted(groups, key=lambda item: (order[item], item))
    ]


def _find_state_render_dir(
    clip_root: Path,
    state_key: str,
    rows: list[dict[str, Any]],
) -> Path | None:
    """Locate the data-driven semantic output dir for one state."""
    safe = _safe_state_key(state_key)
    candidates = [clip_root / "semantic_states" / safe]
    for row in rows:
        state_id = str(row.get("state_id") or "")
        if not state_id:
            continue
        label = str(row.get("state_label") or state_key)
        safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label).strip("_") or state_id
        candidates.append(clip_root / "semantic_states" / f"{state_id}_{safe_label}")
        candidates.append(clip_root / "semantic_states" / f"{state_id}_{safe}")
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "semantic.svg").is_file():
            return candidate
    return None


def _find_state_keyframe(clip_root: Path, state_key: str) -> Path | None:
    """Locate the extracted keyframe PNG for one state."""
    manifest_path = clip_root / "keyframes" / "keyframe_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        for state in manifest.get("states") or []:
            if isinstance(state, dict) and str(state.get("state_key") or "") == state_key:
                asset = state.get("asset")
                if asset and Path(asset).is_file():
                    return Path(asset)
    safe = _safe_state_key(state_key)
    state_dir = clip_root / "keyframes" / "states"
    if state_dir.is_dir():
        matches = sorted(state_dir.glob(f"state_*{safe}*.png"))
        if matches:
            return matches[0]
        matches = sorted(state_dir.glob("state_*.png"))
        if len(matches) == 1:
            return matches[0]
    return None


def _pick_primary_state(
    clip_report: dict[str, Any],
    groups: list[tuple[str, str, list[dict[str, Any]]]],
) -> str | None:
    """Pick the state that best matches the selected keyframe (CV-verified preferred)."""
    if not groups:
        return None
    keyframes = clip_report.get("keyframes") or {}
    selected_ts = _as_number((keyframes.get("timestamps") or {}).get("selected"))
    if selected_ts is not None:
        best_key: str | None = None
        best_delta: float | None = None
        for key, _, rows in groups:
            for row in rows:
                start = _as_number(row.get("state_start"))
                if start is None:
                    continue
                delta = abs(start - selected_ts)
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_key = key
        if best_key is not None and best_delta is not None and best_delta <= 1.0:
            return best_key
    for key, _, rows in groups:
        if any(str(row.get("source_type") or "") == "visual_frame_align" for row in rows):
            return key
    return groups[-1][0]


def _copy_if_exists(src: Path, dst: Path, written: dict[str, str], dst_name: str) -> None:
    if src.exists() and src.stat().st_size > 0:
        shutil.copy2(src, dst)
        written[dst_name] = str(dst)


def _write_state_table(
    clip_root: Path,
    rows: list[dict[str, Any]],
    out_path: Path,
) -> bool:
    csv_path = clip_root / "dynamic_data.csv"
    if csv_path.exists():
        wanted = {
            str(row.get("state_key") or row.get("state_label") or row.get("state_id"))
            for row in rows
        }
        state_ids = {str(row.get("state_id")) for row in rows if row.get("state_id")}
        kept = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            for record in csv.DictReader(f):
                if str(record.get("state_key") or record.get("state_label") or record.get("state_id")) in wanted:
                    kept.append(record)
                elif state_ids and str(record.get("state_id")) in state_ids:
                    kept.append(record)
        if kept:
            with out_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(kept[0].keys()))
                writer.writeheader()
                writer.writerows(kept)
            return True
    if not rows:
        return False
    fields = [
        "clip_id",
        "state_id",
        "state_key",
        "state_label",
        "entity",
        "entity_id",
        "metric",
        "value",
        "unit",
        "value_type",
        "source_type",
        "confidence",
        "review_status",
        "state_start",
        "state_end",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return True


def build_dataset_folder(
    clip_root: str | Path,
    clip_report: dict[str, Any],
) -> dict[str, Any]:
    """Assemble a self-contained ``dataset/`` folder for one clip.

    For dynamic clips (multiple recovered data states) the folder contains a
    top-level full dynamic data table + reconciled animation intent, and one
    ``states/<state_key>/`` subfolder per state with that state's semantic.svg,
    components svg, keyframe, data rows and a static intent. Static clips keep
    a single flat sample (semantic.svg + data_table.csv + intent.json).
    """
    clip_root = Path(clip_root)
    out = clip_root / "dataset"
    if out.exists():
        shutil.rmtree(out)
    out = ensure_dir(out)
    written: dict[str, str] = {}

    dynamic = None
    dynamic_path = clip_root / "dynamic_data.json"
    if dynamic_path.exists():
        try:
            dynamic = json.loads(dynamic_path.read_text(encoding="utf-8"))
        except Exception:
            dynamic = None
    groups = [
        (key, label, rows)
        for key, label, rows in _state_groups(dynamic)
        if any(
            str(row.get("source_type") or "") in {"visual", "visual_frame_align"}
            for row in rows
        )
    ]
    dynamic_packaging = len(groups) >= 2

    intent = None
    intent_path = clip_root / "animation_detection.json"
    if intent_path.exists():
        try:
            intent = json.loads(intent_path.read_text(encoding="utf-8"))
        except Exception:
            intent = None
    if intent is not None and dynamic is not None:
        intent = reconcile_intent_with_data(intent, dynamic)
    if intent is not None:
        write_json(out / "intent.json", intent)
        written["intent.json"] = str(out / "intent.json")

    if dynamic_packaging:
        table_src = clip_root / "dynamic_data.csv"
        if table_src.exists() and table_src.stat().st_size > 0:
            shutil.copy2(table_src, out / "data_table.csv")
            written["data_table.csv"] = str(out / "data_table.csv")
    if "data_table.csv" not in written:
        _copy_if_exists(clip_root / "final_data_table.csv", out / "data_table.csv", written, "data_table.csv")

    nar = None
    processed_dir = (clip_report.get("context") or {}).get("processed_dir")
    if processed_dir:
        nar = Path(processed_dir) / "narration" / "selected_full_sentences.jsonl"
    if nar is None:
        asr_path = (clip_report.get("asr") or {}).get("path")
        if asr_path:
            nar = Path(asr_path).parent / "selected_full_sentences.jsonl"
    if nar is not None and nar.exists() and nar.stat().st_size > 0:
        shutil.copy2(nar, out / "narration.jsonl")
        written["narration.jsonl"] = str(out / "narration.jsonl")

    keyframes = clip_report.get("keyframes") or {}
    assets = keyframes.get("assets") if isinstance(keyframes.get("assets"), dict) else {}
    selected = assets.get("selected")
    if selected and Path(selected).exists():
        shutil.copy2(Path(selected), out / "keyframe.png")
        written["keyframe.png"] = str(out / "keyframe.png")
    _copy_if_exists(clip_root / "aligned_overlay.png", out / "aligned_overlay.png", written, "aligned_overlay.png")

    primary_dir = None
    if dynamic_packaging:
        primary_key = _pick_primary_state(clip_report, groups)
        if primary_key is not None:
            primary_rows = next((rows for key, _, rows in groups if key == primary_key), [])
            primary_dir = _find_state_render_dir(clip_root, primary_key, primary_rows)
    if primary_dir is not None:
        _copy_if_exists(primary_dir / "semantic.svg", out / "semantic.svg", written, "semantic.svg")
        _copy_if_exists(primary_dir / "semantic_components.svg", out / "semantic_components.svg", written, "semantic_components.svg")
    if "semantic.svg" not in written:
        _copy_if_exists(clip_root / "semantic.svg", out / "semantic.svg", written, "semantic.svg")
    if "semantic_components.svg" not in written:
        _copy_if_exists(clip_root / "semantic_components.svg", out / "semantic_components.svg", written, "semantic_components.svg")

    clip = clip_report.get("clip") or {}
    chart_type = str(clip.get("chart_type") or "")
    clip_id = str(clip.get("clip_id") or "")
    state_entries = []
    if dynamic_packaging:
        for state_key, state_label, rows in groups:
            safe = _safe_state_key(state_key)
            state_out = ensure_dir(out / "states" / safe)
            state_files: dict[str, str] = {}
            render_dir = _find_state_render_dir(clip_root, state_key, rows)
            if render_dir is not None:
                _copy_if_exists(render_dir / "semantic.svg", state_out / "semantic.svg", state_files, "semantic.svg")
                _copy_if_exists(render_dir / "semantic_components.svg", state_out / "semantic_components.svg", state_files, "semantic_components.svg")
                _copy_if_exists(render_dir / "semantic_scene.json", state_out / "semantic_scene.json", state_files, "semantic_scene.json")
                _copy_if_exists(render_dir / "semantic_components.json", state_out / "semantic_components.json", state_files, "semantic_components.json")
            keyframe = _find_state_keyframe(clip_root, state_key)
            if keyframe is not None and keyframe.exists():
                shutil.copy2(keyframe, state_out / "keyframe.png")
                state_files["keyframe.png"] = str(state_out / "keyframe.png")
            if _write_state_table(clip_root, rows, state_out / "data_table.csv"):
                state_files["data_table.csv"] = str(state_out / "data_table.csv")
            metric = str(rows[0].get("metric") or "指标") if rows else "指标"
            static_intent = {
                "clip_id": clip_id,
                "state_key": state_key,
                "state_label": state_label,
                "chart_type": chart_type,
                "is_static": True,
                "static_description": f"渲染{state_label}年的{metric}图表（静态状态快照）。",
                "source": "static_state_snapshot",
            }
            write_json(state_out / "intent.json", static_intent)
            state_files["intent.json"] = str(state_out / "intent.json")
            state_entries.append(
                {
                    "state_key": state_key,
                    "state_label": state_label,
                    "dir": str(state_out),
                    "entity_count": len(rows),
                    "files": state_files,
                }
            )

    values = []
    for _, _, rows in groups:
        for row in rows:
            values.append(
                {
                    "state_key": row.get("state_key") or row.get("state_label"),
                    "entity": row.get("entity"),
                    "value": row.get("value"),
                    "type": row.get("value_type"),
                    "confidence": row.get("confidence"),
                }
            )
    if not values:
        table = clip_root / "final_data_table.csv"
        if table.exists():
            with table.open("r", encoding="utf-8-sig", newline="") as f:
                for record in csv.DictReader(f):
                    values.append(
                        {
                            "state_key": None,
                            "entity": record.get("entity"),
                            "value": record.get("value"),
                            "type": record.get("type"),
                            "confidence": record.get("confidence"),
                        }
                    )

    manifest = {
        "clip_id": clip_id,
        "title": clip.get("raw_video_title"),
        "chart_type": chart_type,
        "source_time_range": {
            "start": clip.get("start_seconds"),
            "end": clip.get("end_seconds"),
        },
        "needs_review": bool(keyframes.get("needs_review")),
        "boundary_reason": keyframes.get("boundary_reason"),
        "animation_description": (intent or {}).get("overall_description"),
        "intent_reconciled_with_data": bool((intent or {}).get("reconciled_with_data")),
        "data_state_count": len(groups),
        "states": state_entries,
        "values": values,
        "files": written,
    }
    write_json(out / "manifest.json", manifest)
    written["manifest.json"] = str(out / "manifest.json")
    return {"dataset_dir": str(out), "files": written, "state_count": len(groups)}


def run_pipeline(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    processed_root = ensure_dir(cfg.get("processed_root", "data/processed"))
    generated_root = ensure_dir(cfg.get("generated_root", "data/generated"))
    rows = _load_rows(cfg)
    client = MultichartQwenClient(cfg)

    clip_reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for row in rows:
        clip_id = _clip_id(row)
        processed_dir = ensure_dir(processed_root / clip_id)
        clip_root = ensure_dir(generated_root / clip_id)

        try:
            media = create_context_media({**cfg, "processed_root": str(processed_root)}, row, force=force)
            intervals = media["intervals"]
            visual_clip = Path(media["visual_clip"])
            if not visual_clip.exists():
                raise RuntimeError("missing_visual_clip: run `context` first")
            media = {
                **media,
                "visual_clip": str(visual_clip),
                "intervals": intervals,
            }
            asr_report_path = processed_dir / "narration" / "asr_report.json"
            asr_report = read_json(asr_report_path) if asr_report_path.exists() else {"status": "missing", "path": str(asr_report_path)}
            prior_report_path = clip_root / "clip_report.json"
            prior_visual = read_json(prior_report_path).get("clip", {}).get("visual_clip_path") if prior_report_path.exists() else None
            asset_force = force or prior_visual != str(visual_clip)
            candidate_clip = clip_root / "clip.mp4"
            if asset_force or not candidate_clip.exists():
                import shutil

                shutil.copy2(visual_clip, candidate_clip)

            keyframes = select_keyframe(
                candidate_clip,
                _reference_clip_metadata(row),
                clip_root / "keyframes",
                {**cfg, "processed_root": str(processed_root)},
                client=client,
                force=asset_force,
                context_video=media.get("context_video"),
                context_visual_end=(media.get("intervals") or {}).get("visual_clip_context", {}).get("end"),
            )
            animation = detect_animation(
                {**cfg, "processed_root": str(processed_root)},
                _reference_clip_metadata(row),
                keyframes,
                clip_root,
                client=client,
                force=asset_force,
            )
            chart_data = recover_clip_data(
                cfg,
                keyframes,
                _reference_clip_metadata(row),
                clip_root,
                client=client,
                force=asset_force,
            )
            dynamic = chart_data.get("dynamic_data") or {}
            recovered_type = (chart_data.get("metadata") or {}).get("chart_type") or row.get("chart_type")
            processor, declared_type, type_consistent = detect_chart_type(row.get("chart_type"), recovered_type)
            render_metadata = metadata_from_dynamic(
                dynamic,
                visible_text=(chart_data.get("metadata") or {}).get("visible_text"),
            )
            if render_metadata:
                original_title = (chart_data.get("metadata") or {}).get("title")
                if original_title:
                    render_metadata["title"] = resolve_render_title(
                        original_title,
                        render_metadata.get("title"),
                    )
                write_json(clip_root / "chart_metadata.json", render_metadata)
                chart_data = {**chart_data, "metadata": render_metadata}
            else:
                render_metadata = chart_data.get("metadata") or {}
            if processor == "line":
                selected_keyframe = _selected_keyframe_path(keyframes)
                if selected_keyframe is not None:
                    line_report = run_cv_align_line(_clip_id(row), selected_keyframe, clip_root, cfg=cfg)
                    visible_text = (chart_data.get("metadata") or {}).get("visible_text") or []
                    original_title = (chart_data.get("metadata") or {}).get("title")
                    resolved_title = prefer_frame_visible_title(resolve_render_title(original_title, ""), visible_text)
                    try:
                        frame_title = read_frame_title(selected_keyframe, cfg)
                        if frame_title and len(frame_title) >= 3:
                            resolved_title = frame_title
                    except Exception:
                        pass
                    series_label = None
                    qwen_series = (chart_data.get("metadata") or {}).get("series")
                    if isinstance(qwen_series, list) and qwen_series and isinstance(qwen_series[0], dict):
                        candidate = str(qwen_series[0].get("name") or "").strip()
                        if candidate and candidate.lower() not in {"series", "value", "metric", "unknown"}:
                            series_label = candidate
                    if not series_label:
                        try:
                            series_label = read_series_label(selected_keyframe, cfg)
                        except Exception:
                            series_label = None
                    if not series_label:
                        series_label = _series_label_from_title(resolved_title)
                    cv_dynamic = reconcile_line_dynamic(
                        line_report.get("lines") or [],
                        clip_id=_clip_id(row),
                        image_path=selected_keyframe,
                        keyframe_timestamp=_keyframe_timestamp(keyframes),
                        unit=line_report.get("tick_unit") or "",
                        series_label=series_label,
                    )
                    cv_has_values = bool(cv_dynamic.get("states"))
                    if cv_has_values:
                        dynamic = cv_dynamic
                        chart_data = {**chart_data, "dynamic_data": dynamic}
                    line_metadata = _line_metadata_from_dynamic(
                        dynamic,
                        title=resolved_title,
                        unit=line_report.get("tick_unit") or str((chart_data.get("metadata") or {}).get("unit") or ""),
                        x_labels=line_report.get("x_axis_labels") or [],
                    )
                    semantic = render_data_driven_line(_clip_id(row), line_metadata, clip_root)
                    semantic["cv_align"] = line_report
                    semantic["reconciled"] = {
                        "line_count": line_report.get("line_count", 0),
                        "point_count": line_report.get("point_count", 0),
                        "used_cv_values": cv_has_values,
                        "fallback_source": None if cv_has_values else "qwen_dynamic_data",
                    }
                    write_dynamic_outputs(clip_root, dynamic)
                    write_json(clip_root / "chart_metadata.json", line_metadata)
            else:
                semantic = render_data_driven(_clip_id(row), render_metadata, clip_root)
            if processor != "line":
                semantic["state_renders"] = render_dynamic_states(
                    _clip_id(row),
                    dynamic,
                    clip_root,
                    visible_text=(chart_data.get("metadata") or {}).get("visible_text"),
                )
            else:
                semantic["state_renders"] = []
            if processor == "bar" and _cv_align_enabled(row, cfg):
                selected_keyframe = _selected_keyframe_path(keyframes)
                entities: list[dict[str, Any]] = []
                seen: set[str] = set()
                for state_row in (dynamic.get("states") or []) if isinstance(dynamic, dict) else []:
                    if not isinstance(state_row, dict):
                        continue
                    eid = str(state_row.get("entity_id") or "")
                    if eid in ("", "unknown") or eid in seen:
                        continue
                    seen.add(eid)
                    entities.append({"id": eid, "label": str(state_row.get("entity") or eid)})
                if selected_keyframe is not None and entities:
                    cv_report = run_cv_align(
                        _clip_id(row),
                        selected_keyframe,
                        entities,
                        clip_root,
                        client=client,
                        cfg=cfg,
                    )
                    semantic["cv_align"] = cv_report
                    implausible = cv_report.get("implausible_bars") or []
                    reconciled = None
                    if not implausible:
                        reconciled = reconcile_dynamic_data(
                            dynamic,
                            cv_report,
                            clip_id=_clip_id(row),
                            keyframe_timestamp=_keyframe_timestamp(keyframes),
                            image_path=selected_keyframe,
                            out_dir=clip_root,
                        )
                    else:
                        semantic["reconciled"] = {
                            "updated_bar_count": 0,
                            "skipped_bar_count": len(implausible),
                            "skipped_bars": implausible,
                            "reason": "frame values failed plausibility; kept recovered data table",
                        }
                    if reconciled:
                        dynamic = reconciled["dynamic"]
                        chart_data = {**chart_data, "dynamic_data": dynamic}
                        corrected_metadata = metadata_from_dynamic(
                            dynamic,
                            visible_text=(chart_data.get("metadata") or {}).get("visible_text"),
                        )
                        if corrected_metadata:
                            original_title = (chart_data.get("metadata") or {}).get("title")
                            if original_title:
                                corrected_metadata["title"] = resolve_render_title(
                                    original_title,
                                    corrected_metadata.get("title"),
                                )
                            cv_orientation = (cv_report or {}).get("orientation")
                            if cv_orientation:
                                corrected_metadata["orientation"] = cv_orientation
                            write_json(clip_root / "chart_metadata.json", corrected_metadata)
                            chart_data = {**chart_data, "metadata": corrected_metadata}
                            semantic = render_data_driven(_clip_id(row), corrected_metadata, clip_root)
                        semantic["state_renders"] = render_dynamic_states(
                            _clip_id(row),
                            dynamic,
                            clip_root,
                            visible_text=(chart_data.get("metadata") or {}).get("visible_text"),
                        )
                        semantic["cv_align"] = cv_report
                        semantic["reconciled"] = {
                            "updated_bar_count": reconciled["updated_bar_count"],
                            "skipped_bar_count": reconciled["skipped_bar_count"],
                            "state_key": reconciled["state_key"],
                            "state_id": reconciled["state_id"],
                        }
            animation = reconcile_intent_with_data(animation, dynamic)
            write_json(clip_root / "animation_detection.json", animation)
            semantic_state_svgs = build_semantic_state_svgs(
                chart_data.get("semantic_state_inputs"),
                clip_root,
                cfg,
                force=asset_force,
            )

            clip_report = _write_candidate_report(
                clip_root,
                row,
                media,
                intervals,
                asr_report,
                keyframes,
                animation,
                semantic,
                semantic_state_svgs,
                chart_data,
            )
            clip_report["visual_boundary_source"] = "web_reference_interval"
            clip_report["deprecated_clip_boundary_review_ignored"] = True
            clip_report["asset_status"] = "fresh"
            clip_report["clip"]["visual_clip_path"] = str(visual_clip)
            clip_report["clip"]["visual_clip_source"] = "reference_source"
            write_json(clip_root / "clip_report.json", clip_report)
            clip_report["dataset"] = build_dataset_folder(clip_root, clip_report)
            write_json(clip_root / "clip_report.json", clip_report)
            clip_reports.append(clip_report)
            failed_path = clip_root / "clip_report_failed.json"
            if failed_path.exists():
                failed_path.unlink()
        except Exception as exc:
            failure = {"clip_id": clip_id, "clip": row, "failure_reason": str(exc)}
            write_json(clip_root / "clip_report_failed.json", failure)
            failures.append(failure)

    write_jsonl(generated_root / "multichart_v2_clips.jsonl", [report["clip"] for report in clip_reports])
    run_report = {
        "sample_id": cfg["sample_id"],
        "source": str(cfg.get("clip_metadata_csv") or cfg.get("raw_clips_jsonl", "data/raw/datavideo_clips.jsonl")),
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
    return run_report
