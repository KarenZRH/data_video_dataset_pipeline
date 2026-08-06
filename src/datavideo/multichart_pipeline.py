from __future__ import annotations

from pathlib import Path
from typing import Any

from datavideo.context import create_context_media
from .animation import detect_animation
from datavideo.cv_align import run_cv_align
from datavideo.cv_reconcile import reconcile_dynamic_data
from datavideo.metadata import read_clip_rows
from datavideo.narration import transcribe_context_audio
from datavideo.semantic import build_semantic_svg
from datavideo.semantic_render import (
    metadata_from_dynamic,
    render_data_driven,
    render_dynamic_states,
    resolve_render_title,
)
from datavideo.schemas import ensure_dir, read_json, write_json, write_jsonl

from .multichart_assets import (
    _keyframe_asset,
    _keyframe_timestamp,
    build_semantic_state_svgs,
    recover_clip_data,
    select_keyframe,
)
from .multichart_qwen import MultichartQwenClient


def _clip_id(row: dict[str, Any]) -> str:
    return str(row.get("output_stem") or f"{row['chart_type']}_{row['chart_index']}")


def _reference_clip_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "clip_id": _clip_id(row),
        "source_video": row.get("output_path"),
    }


def _load_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return read_clip_rows(cfg)


def _semantic_input_path(keyframes: dict[str, Any], state_inputs: Any) -> str:
    if isinstance(state_inputs, dict):
        inputs = state_inputs.get("semantic_inputs")
        if isinstance(inputs, list) and inputs:
            asset = inputs[0].get("semantic_input") if isinstance(inputs[0], dict) else None
            if asset:
                return str(asset)
    states = keyframes.get("states") if isinstance(keyframes.get("states"), list) else []
    for state in states:
        if isinstance(state, dict) and state.get("asset"):
            return str(state["asset"])
    assets = keyframes.get("assets") if isinstance(keyframes.get("assets"), dict) else {}
    if assets.get("selected"):
        return str(assets["selected"])
    raise RuntimeError("No semantic input frame available")


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
            state_inputs = chart_data.get("semantic_state_inputs") if isinstance(chart_data, dict) else None
            try:
                semantic_input = _semantic_input_path(keyframes, state_inputs)
            except Exception:
                semantic_input = _keyframe_asset(keyframes)
            qwen_semantic = None
            if semantic_input and Path(semantic_input).exists():
                qwen_semantic = build_semantic_svg(
                    Path(semantic_input),
                    ensure_dir(clip_root / "qwen_semantic"),
                    cfg,
                    force=asset_force,
                )
            semantic = render_data_driven(_clip_id(row), chart_data.get("metadata") or {}, clip_root)
            semantic["qwen_semantic_svg"] = (qwen_semantic or {}).get("semantic_svg")
            semantic["state_renders"] = render_dynamic_states(
                _clip_id(row),
                chart_data.get("dynamic_data") or {},
                clip_root,
            )
            dynamic = chart_data.get("dynamic_data") or {}
            entities: list[dict[str, Any]] = []
            seen: set[str] = set()
            for state_row in (dynamic.get("states") or []) if isinstance(dynamic, dict) else []:
                if not isinstance(state_row, dict):
                    continue
                eid = str(state_row.get("entity_id") or "")
                if eid in ("", "unknown") or eid in seen:
                    continue
                seen.add(eid)
                entities.append(
                    {"id": eid, "label": str(state_row.get("entity") or eid)}
                )
            semantic["cv_align"] = run_cv_align(
                _clip_id(row),
                _keyframe_asset(keyframes),
                entities,
                clip_root,
                client=client,
                cfg=cfg,
            )
            cv_report = semantic["cv_align"] or {}
            implausible = cv_report.get("implausible_bars") or []
            reconciled = None
            if not implausible:
                reconciled = reconcile_dynamic_data(
                    dynamic,
                    cv_report,
                    clip_id=_clip_id(row),
                    keyframe_timestamp=_keyframe_timestamp(keyframes),
                    image_path=_keyframe_asset(keyframes),
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
                    "state_key": reconciled["state_key"],
                    "state_id": reconciled["state_id"],
                }
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
