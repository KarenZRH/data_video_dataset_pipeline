from __future__ import annotations

from pathlib import Path
from typing import Any

from .schemas import write_jsonl


def _is_positive_result(res: dict[str, Any], min_confidence: float) -> bool:
    if not res.get("is_chart"):
        return False
    confidence = float(res.get("confidence", 0.0) or 0.0)
    if confidence >= min_confidence:
        return True
    # Qwen sometimes copies the schema's 0.0 confidence while the rest of the JSON is a clear positive.
    return (
        confidence == 0.0
        and res.get("chart_visible", True)
        and not res.get("occlusion", False)
        and res.get("scene_state") in {"chart_entering", "stable_chart", "chart_animating", "chart_leaving"}
        and bool(res.get("chart_types"))
    )


def _effective_confidence(res: dict[str, Any]) -> float:
    confidence = float(res.get("confidence", 0.0) or 0.0)
    if confidence > 0:
        return confidence
    if res.get("is_chart") and res.get("chart_types"):
        return 0.6
    return 0.0


def merge_positive_frames(
    frame_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    *,
    expand: float,
    min_duration: float,
    duration: float,
    out_path: str | Path,
    min_positive_frames: int = 2,
    min_confidence: float = 0.5,
    max_gap_seconds: float | None = None,
) -> list[dict[str, Any]]:
    by_id = {row["frame_id"]: row for row in result_rows}
    positives = []
    for frame in frame_rows:
        res = by_id.get(frame["frame_id"], {}).get("result", {})
        if _is_positive_result(res, min_confidence):
            positives.append({**frame, "confidence": _effective_confidence(res)})
    clips = []
    if positives:
        start = positives[0]["timestamp"]
        end = positives[0]["timestamp"]
        confs = [positives[0]["confidence"]]
        last = positives[0]["timestamp"]
        step = 1.0 / max(0.001, positives[0].get("fps", 2))
        for row in positives[1:]:
            allowed_gap = max_gap_seconds if max_gap_seconds is not None else step * 1.75
            if row["timestamp"] - last <= allowed_gap:
                end = row["timestamp"]
                confs.append(row["confidence"])
            else:
                if len(confs) >= min_positive_frames:
                    clips.append(_clip(start, end, confs, expand, min_duration, duration, len(clips)))
                start = end = row["timestamp"]
                confs = [row["confidence"]]
            last = row["timestamp"]
        if len(confs) >= min_positive_frames:
            clips.append(_clip(start, end, confs, expand, min_duration, duration, len(clips)))
    write_jsonl(out_path, clips)
    return clips


def _clip(start: float, end: float, confs: list[float], expand: float, min_duration: float, duration: float, idx: int) -> dict[str, Any]:
    s = max(0.0, start - expand)
    e = min(duration, end + expand)
    if e - s < min_duration:
        pad = (min_duration - (e - s)) / 2
        s = max(0.0, s - pad)
        e = min(duration, e + pad)
    return {
        "clip_id": f"clip_{idx:03d}",
        "start": round(s, 3),
        "end": round(e, 3),
        "confidence": round(sum(confs) / len(confs), 4),
        "source": "coarse_qwen_merge",
        "failure_reason": None,
    }


def refine_clip_with_fine_results(
    clip: dict[str, Any],
    fine_frames: list[dict[str, Any]],
    fine_results: list[dict[str, Any]],
    *,
    min_confidence: float = 0.5,
) -> dict[str, Any]:
    by_id = {row["frame_id"]: row for row in fine_results}
    positives = [
        {**frame, "confidence": by_id.get(frame["frame_id"], {}).get("result", {}).get("confidence", 0.0)}
        for frame in fine_frames
        if _is_positive_result(by_id.get(frame["frame_id"], {}).get("result", {}), min_confidence)
    ]
    if positives:
        return {
            **clip,
            "start": round(max(0.0, positives[0]["timestamp"] - 0.25), 3),
            "end": round(positives[-1]["timestamp"] + 0.25, 3),
            "confidence": round(sum(p["confidence"] for p in positives) / len(positives), 4),
            "source": "fine_qwen_refine",
        }
    return {**clip, "source": "fine_no_positive_keep_coarse", "failure_reason": "no fine positive frames"}


def refine_with_fine_results(
    clip: dict[str, Any],
    fine_frames: list[dict[str, Any]],
    fine_results: list[dict[str, Any]],
    out_path: str | Path,
) -> list[dict[str, Any]]:
    refined = refine_clip_with_fine_results(clip, fine_frames, fine_results)
    write_jsonl(out_path, [refined])
    return [refined]
