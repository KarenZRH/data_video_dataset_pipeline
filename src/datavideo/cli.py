from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .bar_assets import run_bar_assets_pipeline
from .bar_dominant import run_bar_dominant_pipeline
from .chart_data import recover_chart_data
from .clip_detect import merge_positive_frames, refine_clip_with_fine_results
from .detect import detect_data_video_clips
from .frames import extract_frames, write_contact_sheet, write_frame_manifest
from .keyframes import extract_scored_keyframe
from .manifest import load_config, write_video_manifest
from .media import extract_clip, ffprobe, normalize_video
from .merge_review import review_merged_clips
from .model_client import make_qwen_client
from .review_db import init_db
from .schemas import ensure_dir, read_jsonl, write_json, write_jsonl
from .svg_trace import trace_svg
from datavideo_multichart.assets import run_pipeline as run_multichart_assets_pipeline
from datavideo_multichart_v2.pipeline import (
    run_asr_pipeline as run_multichart_asr_v2_pipeline,
    run_chart_boundary_pipeline as run_multichart_chart_boundary_v2_pipeline,
    run_context_pipeline as run_multichart_context_v2_pipeline,
    run_pipeline as run_multichart_assets_v2_pipeline,
    run_proposal_pipeline as run_multichart_proposal_v2_pipeline,
)
from datavideo_multichart_v2.reviewed_outputs import apply_latest_reviews as apply_multichart_v2_reviews


def _duration(video: str | Path) -> float:
    return float(ffprobe(video)["format"]["duration"])


def _classify_groups(cfg: dict[str, Any], frames: list[dict[str, Any]], out_path: Path, force: bool) -> list[dict[str, Any]]:
    if out_path.exists() and not force:
        return read_jsonl(out_path)
    client = make_qwen_client(cfg)
    results = []
    max_n = int(cfg["model"].get("max_frames_per_call", 3))
    for i in range(0, len(frames), max_n):
        group = frames[i : i + max_n]
        response = client.classify_frames([row["path"] for row in group])
        for row in group:
            results.append(
                {
                    "sample_id": cfg["sample_id"],
                    "frame_id": row["frame_id"],
                    "timestamp": row["timestamp"],
                    "image_path": row["path"],
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
    write_jsonl(out_path, results)
    return results


def _classify_clip_groups(
    cfg: dict[str, Any],
    clip_id: str,
    frames: list[dict[str, Any]],
    out_path: Path,
    force: bool,
) -> list[dict[str, Any]]:
    rows = _classify_groups(cfg, frames, out_path, force)
    for row in rows:
        row["clip_id"] = clip_id
    write_jsonl(out_path, rows)
    return rows


def stage0(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    ensure_dir(cfg["processed_dir"])
    ensure_dir(cfg["generated_dir"])
    ensure_dir(cfg["reviewed_dir"])
    init_db(cfg["review_db"])
    manifest = write_video_manifest(cfg)
    media = normalize_video(cfg, force=force)
    return {"manifest": manifest, "media": media}


def stage1(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    processed = ensure_dir(cfg["processed_dir"])
    generated = ensure_dir(cfg["generated_dir"])
    normalized = processed / "normalized.mp4"
    if not normalized.exists():
        stage0(cfg, force=force)
    duration = _duration(normalized)

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
    coarse_results = _classify_groups(cfg, coarse_frames, generated / "qwen_frame_results.jsonl", force)
    coarse_scores = {row["frame_id"]: row["result"]["confidence"] for row in coarse_results}
    write_contact_sheet(coarse_frames[:32], generated / "contact_sheet.jpg", scores=coarse_scores)

    candidates = merge_positive_frames(
        coarse_frames,
        coarse_results,
        expand=cfg["sampling"]["clip_expand_seconds"],
        min_duration=cfg["sampling"]["min_clip_seconds"],
        duration=duration,
        out_path=generated / "candidate_clips.jsonl",
        min_positive_frames=int(cfg.get("detection", {}).get("min_positive_frames", 2)),
        min_confidence=float(cfg.get("detection", {}).get("min_confidence", 0.5)),
        max_gap_seconds=cfg.get("detection", {}).get("max_gap_seconds"),
    )
    if not candidates:
        raise RuntimeError("No chart candidate clips found")

    refined_clips = []
    clip_reports = []
    for clip in candidates:
        clip_root = ensure_dir(generated / "clips" / clip["clip_id"])
        fine_dir = processed / "frames" / "fine_8fps" / clip["clip_id"]
        fine_frames = extract_frames(
            normalized,
            fine_dir,
            cfg["sampling"]["fine_fps"],
            cfg["sampling"]["short_side"],
            "fine",
            force=force,
            start=clip["start"],
            end=clip["end"],
        )
        for row in fine_frames:
            row["clip_id"] = clip["clip_id"]
        fine_results = _classify_clip_groups(cfg, clip["clip_id"], fine_frames, clip_root / "qwen_fine_frame_results.jsonl", force)
        refined = refine_clip_with_fine_results(
            clip,
            fine_frames,
            fine_results,
            min_confidence=float(cfg.get("detection", {}).get("min_confidence", 0.5)),
        )
        refined_clips.append(refined)
        clip_video = extract_clip(normalized, refined["start"], refined["end"], clip_root / "clip.mp4", force=force)

        keyframe_dir = clip_root / "keyframes"
        keyframe_manifest = extract_scored_keyframe(normalized, refined, keyframe_dir, cfg, force=force)
        initial = keyframe_manifest["assets"]["initial"]
        trace_report = trace_svg(initial, clip_root, cfg, force=force)
        data_report = recover_chart_data(cfg, initial, clip_root)

        clip_reports.append(
            {
                "clip": refined,
                "clip_video": str(clip_video),
                "keyframes": keyframe_manifest,
                "trace": trace_report,
                "chart_data": data_report,
            }
        )
    write_jsonl(generated / "refined_clips.jsonl", refined_clips)

    first_report = clip_reports[0]

    reviewed = ensure_dir(cfg["reviewed_dir"])
    if not (reviewed / "README.md").exists():
        (reviewed / "README.md").write_text("Human-reviewed outputs for this sample belong here.\n", encoding="utf-8")

    run_report = {
        "sample_id": cfg["sample_id"],
        "normalized_video": str(normalized),
        "audio_wav": str(processed / "audio_16k_mono.wav"),
        "coarse_frame_count": len(coarse_frames),
        "candidate_clip_count": len(candidates),
        "refined_clip_count": len(refined_clips),
        "clips": clip_reports,
        "selected_clip": first_report["clip"],
        "clip_video": first_report["clip_video"],
        "keyframes": first_report["keyframes"],
        "trace": first_report["trace"],
        "chart_data": first_report["chart_data"],
        "model_path": os.environ.get(cfg["model"]["env_var"]),
        "prompt_version": cfg["model"]["prompt_version"],
        "config_hash": cfg["config_hash"],
    }
    write_json(generated / "run_report.json", run_report)
    return run_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Data-video dataset pipeline")
    parser.add_argument(
        "command",
        choices=[
            "stage0",
            "stage1",
            "run",
            "detect",
            "merge-review",
            "bar-dominant",
            "bar-assets",
            "multichart-assets",
            "multichart-assets-v2",
            "multichart-context-v2",
            "multichart-chart-boundary-v2",
            "multichart-asr-v2",
            "multichart-propose-v2",
            "multichart-review-v2",
        ],
    )
    parser.add_argument("--config", default="configs/stage1_bar.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--clip-id", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.clip_id:
        cfg["clip_id"] = args.clip_id
    if args.command == "stage0":
        report = stage0(cfg, force=args.force)
    elif args.command == "stage1":
        report = stage1(cfg, force=args.force)
    elif args.command == "detect":
        report = detect_data_video_clips(cfg, force=args.force)
    elif args.command == "merge-review":
        report = review_merged_clips(cfg, force=args.force)
    elif args.command == "bar-dominant":
        report = run_bar_dominant_pipeline(cfg, force=args.force)
    elif args.command == "bar-assets":
        report = run_bar_assets_pipeline(cfg, force=args.force)
    elif args.command == "multichart-assets":
        report = run_multichart_assets_pipeline(cfg, force=args.force)
    elif args.command == "multichart-assets-v2":
        report = run_multichart_assets_v2_pipeline(cfg, force=args.force)
    elif args.command == "multichart-context-v2":
        report = run_multichart_context_v2_pipeline(cfg, force=args.force)
    elif args.command == "multichart-chart-boundary-v2":
        report = run_multichart_chart_boundary_v2_pipeline(cfg, force=args.force)
    elif args.command == "multichart-asr-v2":
        report = run_multichart_asr_v2_pipeline(cfg, force=args.force)
    elif args.command == "multichart-propose-v2":
        report = run_multichart_proposal_v2_pipeline(cfg, force=args.force)
    elif args.command == "multichart-review-v2":
        report = apply_multichart_v2_reviews(cfg)
    else:
        stage0(cfg, force=args.force)
        report = stage1(cfg, force=args.force)
    print(report)


if __name__ == "__main__":
    main()
