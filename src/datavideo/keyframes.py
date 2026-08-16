from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageStat

from .frames import extract_frames
from .model_client import make_qwen_client
from .schemas import ensure_dir, read_json, write_json, write_jsonl


def extract_still(video: str | Path, timestamp: float, out: str | Path, force: bool = False) -> Path:
    out = Path(out)
    ensure_dir(out.parent)
    def _extract(ts: float, target: Path) -> None:
        cmd = [
            "ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", str(video),
            "-frames:v", "1", "-q:v", "2",
        ]
        if target.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            cmd += ["-f", "image2"]
        cmd.append(str(target))
        subprocess.run(cmd, check=True)

    if not (out.exists() and not force):
        _extract(timestamp, out)
        # Seeking to the exact end of a video frequently yields no frame
        # (e.g. clip.mp4 of 6.0s at ts=6.0). Retry slightly earlier so a
        # boundary state still gets a valid still instead of an empty file.
        if not out.exists() or out.stat().st_size == 0:
            try:
                from datavideo.media import ffprobe
                duration = float(ffprobe(video)["format"]["duration"])
                retry = max(0.0, min(timestamp, duration - 0.25))
                if retry != timestamp:
                    _extract(retry, out)
            except Exception:
                pass
    # ffmpeg may honour the container's codec over the requested extension
    # (e.g. a JPEG payload written to a .png path). Verify the payload
    # matches the extension and re-encode when it does not.
    if out.suffix.lower() == ".png" and out.exists():
        header = out.read_bytes()[:8]
        if header != b"\x89PNG\r\n\x1a\n":
            fixed = out.with_suffix(".png.tmp")
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(out),
                    "-frames:v", "1", "-pix_fmt", "rgba", "-f", "image2", str(fixed),
                ],
                check=True,
            )
            fixed.replace(out)
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"Failed to extract still at {timestamp:.3f}s from {video}")
    return out


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _sample_fps(cfg: dict[str, Any]) -> float:
    requested = cfg.get("keyframes", {}).get("sample_fps", cfg.get("sampling", {}).get("fine_fps", 6))
    try:
        return max(4.0, min(8.0, float(requested)))
    except (TypeError, ValueError):
        return 6.0


def _chart_identity(clip: dict[str, Any]) -> str:
    fields = ["chart_identities", "chart_titles", "axis_labels", "category_labels", "chart_types", "mark_types"]
    parts = []
    for field in fields:
        value = clip.get(field)
        if isinstance(value, list):
            text = ",".join(sorted(str(item) for item in value if item))
        else:
            text = str(value or "")
        if text:
            parts.append(f"{field}={text}")
    if not parts:
        candidates = clip.get("candidates") if isinstance(clip.get("candidates"), list) else []
        for cand in candidates[:1]:
            for field in fields:
                value = cand.get(field)
                if isinstance(value, list):
                    text = ",".join(sorted(str(item) for item in value if item))
                else:
                    text = str(value or "")
                if text:
                    parts.append(f"{field}={text}")
    return "; ".join(parts) or "unknown bar-chart identity"


def _in_candidate_source(frame: dict[str, Any], clip: dict[str, Any]) -> bool:
    timestamp = float(frame["timestamp"])
    candidates = clip.get("candidates") if isinstance(clip.get("candidates"), list) else []
    for cand in candidates:
        start = float(cand.get("source_start", cand.get("start", clip["start"])))
        end = float(cand.get("source_end", cand.get("end", clip["end"])))
        if start <= timestamp <= end:
            return True
    return not candidates and float(clip["start"]) <= timestamp <= float(clip["end"])


def _image_motion_scores(frames: list[dict[str, Any]]) -> dict[str, float]:
    images = []
    for frame in frames:
        image = Image.open(frame["path"]).convert("L").resize((96, 96))
        images.append(image)
    pairwise = []
    for prev, cur in zip(images, images[1:]):
        diff = ImageChops.difference(prev, cur)
        pairwise.append(_clamp01(ImageStat.Stat(diff).mean[0] / 255.0))
    scores = {}
    for idx, frame in enumerate(frames):
        neighbors = []
        if idx > 0:
            neighbors.append(pairwise[idx - 1])
        if idx < len(pairwise):
            neighbors.append(pairwise[idx])
        scores[frame["frame_id"]] = sum(neighbors) / len(neighbors) if neighbors else 0.0
    return scores


def _heuristic_keyframe_score(frame: dict[str, Any], clip: dict[str, Any], motion_score: float) -> dict[str, Any]:
    same_chart = _in_candidate_source(frame, clip)
    staticness = 1.0 - motion_score
    return {
        "same_chart": same_chart,
        "scene_change": not same_chart,
        "pre_change": same_chart,
        "post_change_state": False,
        "complete_initial_chart": same_chart,
        "all_target_categories_visible": same_chart,
        "completeness": 0.65 if same_chart else 0.0,
        "staticness": staticness,
        "chart_identity_consistency": 0.75 if same_chart else 0.0,
        "initial_state_representative": 0.65 if same_chart else 0.0,
        "motion_score": motion_score,
        "reason": "heuristic score from sampled frame membership and image motion",
    }


def _combined_score(score: dict[str, Any]) -> float:
    return (
        1.5 * _clamp01(score.get("initial_state_representative"))
        + 1.2 * _clamp01(score.get("completeness"))
        + 1.0 * _clamp01(score.get("chart_identity_consistency"))
        + 0.5 * _clamp01(score.get("staticness"))
        + (0.5 if score.get("pre_change") else -2.0)
        + (0.5 if score.get("all_target_categories_visible") else -2.0)
        - 1.0 * _clamp01(score.get("motion_score"), default=1.0)
        - (2.0 if score.get("post_change_state") else 0.0)
        - (2.0 if score.get("scene_change") else 0.0)
    )


def _selection_rank(row: dict[str, Any]) -> tuple[bool, bool, bool, bool, bool, bool, float, float, float, float]:
    score = row["score"]
    motion_score = _clamp01(score.get("motion_score"), default=1.0)
    return (
        bool(score.get("same_chart")),
        not bool(score.get("scene_change")),
        bool(score.get("pre_change")),
        not bool(score.get("post_change_state")),
        bool(score.get("complete_initial_chart")),
        bool(score.get("all_target_categories_visible")),
        _clamp01(score.get("initial_state_representative")),
        float(row["combined_score"]),
        -motion_score,
        -float(row["timestamp"]),
    )


def extract_scored_keyframe(
    video: str | Path,
    clip: dict[str, Any],
    out_dir: str | Path,
    cfg: dict[str, Any],
    *,
    client: Any | None = None,
    force: bool = False,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    manifest_path = out_dir / "keyframe_manifest.json"
    if manifest_path.exists() and not force:
        cached = read_json(manifest_path)
        selected = Path(cached.get("assets", {}).get("selected", ""))
        if cached.get("clip_id") == clip["clip_id"] and selected.exists():
            return cached
    frame_dir = ensure_dir(out_dir / "candidate_frames")
    fps = _sample_fps(cfg)
    frames = extract_frames(
        video,
        frame_dir,
        fps,
        int(cfg.get("sampling", {}).get("short_side", 768)),
        "keyframe_candidate",
        force=force,
        start=float(clip["start"]),
        end=float(clip["end"]),
    )
    if not frames:
        raise RuntimeError(f"No keyframe candidates sampled for {clip['clip_id']}")

    motion_scores = _image_motion_scores(frames)
    scorer = client or make_qwen_client(cfg)
    identity = _chart_identity(clip)
    scored_rows = []
    for frame in frames:
        motion_score = motion_scores.get(frame["frame_id"], 0.0)
        model_score = scorer.score_keyframe_candidate(frame["path"], identity)
        score = model_score["result"]
        if model_score["model_status"] != "qwen":
            score = _heuristic_keyframe_score(frame, clip, motion_score)
        else:
            score["motion_score"] = max(_clamp01(score.get("motion_score"), default=1.0), motion_score)
        combined = _combined_score(score)
        scored_rows.append(
            {
                **frame,
                "score": score,
                "combined_score": round(combined, 4),
                "raw_response": model_score["raw_response"],
                "model_status": model_score["model_status"],
                "failure_reason": model_score["failure_reason"],
            }
        )

    selected = max(scored_rows, key=_selection_rank)
    timestamp = float(selected["timestamp"])
    asset = str(extract_still(video, timestamp, out_dir / "selected.png", force=force))
    manifest = {
        "clip_id": clip["clip_id"],
        "timestamps": {"selected": timestamp},
        "assets": {"selected": asset},
        "selection_method": "sampled_frame_priority_selected_keyframe",
        "source_frame_id": selected["frame_id"],
        "sample_fps": fps,
        "chart_identity": identity,
        "selected_score": selected["score"],
        "combined_score": selected["combined_score"],
        "score_manifest": str(out_dir / "keyframe_scores.jsonl"),
        "requirements": {
            "same_chart": True,
            "scene_change": False,
            "pre_change": True,
            "post_change_state": False,
            "complete_initial_chart": True,
            "all_target_categories_visible": True,
            "description": "Select a representative clip keyframe before the main data-change animation; motion is only an auxiliary tie-breaker after completeness.",
        },
    }
    write_jsonl(out_dir / "keyframe_scores.jsonl", scored_rows)
    write_json(manifest_path, manifest)
    return manifest
