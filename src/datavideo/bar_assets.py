from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .chart_data import recover_chart_data
from .keyframes import extract_scored_keyframe
from .media import extract_clip
from .model_client import make_qwen_client
from .schemas import ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from .svg_trace import trace_svg


def run_bar_assets_pipeline(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    generated = ensure_dir(cfg["generated_dir"])
    processed = Path(cfg["processed_dir"])
    normalized = processed / "normalized.mp4"
    clips = read_jsonl(generated / "final_bar_clips.jsonl")
    client = make_qwen_client(cfg)

    clip_reports: list[dict[str, Any]] = []
    for clip in clips:
        clip_id = clip["clip_id"]
        clip_root = ensure_dir(generated / "clips" / clip_id)
        clip_report_path = clip_root / "clip_report.json"
        if clip_report_path.exists() and not force:
            cached_report = read_json(clip_report_path)
            cached_clip = cached_report.get("clip", {})
            if (
                cached_clip.get("clip_id") == clip_id
                and cached_clip.get("start") == clip.get("start")
                and cached_clip.get("end") == clip.get("end")
            ):
                clip_reports.append(cached_report)
                continue
        keyframe_dir = ensure_dir(clip_root / "keyframes")

        clip_video = clip_root / "clip.mp4"
        if force or not clip_video.exists():
            source_clip_value = clip.get("clip_mp4")
            source_clip = Path(source_clip_value) if source_clip_value else None
            if source_clip is not None and source_clip.is_file() and not force:
                shutil.copy2(source_clip, clip_video)
            else:
                extract_clip(normalized, float(clip["start"]), float(clip["end"]), clip_video, force=True)

        keyframes = extract_scored_keyframe(normalized, clip, keyframe_dir, cfg, client=client, force=force)
        initial = keyframes["assets"]["initial"]
        trace = trace_svg(initial, clip_root, cfg, force=force)
        chart_data = recover_chart_data(cfg, initial, clip_root, client=client)

        report = {
            "clip": clip,
            "clip_video": str(clip_video),
            "keyframes": keyframes,
            "trace": trace,
            "chart_data": chart_data,
        }
        write_json(clip_report_path, report)
        clip_reports.append(report)

    run_report = {
        "sample_id": cfg["sample_id"],
        "source": str(generated / "final_bar_clips.jsonl"),
        "clip_count": len(clips),
        "clips": clip_reports,
        "config_hash": cfg["config_hash"],
    }
    write_json(generated / "run_report.json", run_report)
    write_jsonl(generated / "refined_clips.jsonl", [row["clip"] for row in clip_reports])
    return run_report
