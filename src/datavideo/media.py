from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .schemas import ensure_dir, write_json


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def ffprobe(path: str | Path) -> dict[str, Any]:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    return json.loads(out)


def normalize_video(cfg: dict[str, Any], force: bool = False) -> dict[str, str]:
    processed = ensure_dir(cfg["processed_dir"])
    src = Path(cfg["video_path"])
    video_out = processed / "normalized.mp4"
    wav_out = processed / "audio_16k_mono.wav"
    return standardize_media(
        src,
        processed,
        cfg["video_standardization"],
        video_out=video_out,
        wav_out=wav_out,
        force=force,
    )


def standardize_media(
    src: str | Path,
    out_dir: str | Path,
    settings: dict[str, Any],
    *,
    video_out: str | Path,
    wav_out: str | Path,
    report_name: str = "standardization_report.json",
    force: bool = False,
) -> dict[str, str]:
    out_dir = ensure_dir(out_dir)
    src = Path(src)
    video_out = Path(video_out)
    wav_out = Path(wav_out)
    if force or not video_out.exists():
        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-c:v",
                settings["video_codec"],
                "-pix_fmt",
                settings["pixel_format"],
                "-r",
                str(settings["fps"]),
                "-fps_mode",
                "cfr",
                "-crf",
                str(settings["crf"]),
                "-preset",
                "medium",
                "-c:a",
                settings["audio_codec"],
                "-ar",
                str(settings["audio_rate"]),
                "-movflags",
                "+faststart",
                str(video_out),
            ]
        )
    if force or not wav_out.exists():
        run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_out),
                "-vn",
                "-ac",
                str(settings["wav_channels"]),
                "-ar",
                str(settings["wav_rate"]),
                str(wav_out),
            ]
        )
    report = {
        "video": str(video_out),
        "wav": str(wav_out),
        "normalized_video": str(video_out),
        "audio_wav": str(wav_out),
        "probe": ffprobe(video_out),
    }
    write_json(out_dir / report_name, report)
    return {"video": str(video_out), "wav": str(wav_out)}


def extract_clip(video: str | Path, start: float, end: float, out: str | Path, force: bool = False) -> Path:
    out = Path(out)
    ensure_dir(out.parent)
    if out.exists() and not force:
        return out
    duration = max(0.1, end - start)
    run(["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{duration:.3f}", "-c", "copy", str(out)])
    return out


def extract_clip_accurate(
    video: str | Path,
    start: float,
    end: float,
    out: str | Path,
    settings: dict[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    out = Path(out)
    ensure_dir(out.parent)
    expected_duration = max(0.1, float(end) - float(start))
    if force or not out.exists():
        run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{float(start):.3f}",
                "-i",
                str(video),
                "-t",
                f"{expected_duration:.3f}",
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-c:v",
                settings["video_codec"],
                "-pix_fmt",
                settings["pixel_format"],
                "-r",
                str(settings["fps"]),
                "-fps_mode",
                "cfr",
                "-crf",
                str(settings["crf"]),
                "-preset",
                "medium",
                "-c:a",
                settings["audio_codec"],
                "-ar",
                str(settings["audio_rate"]),
                "-movflags",
                "+faststart",
                str(out),
            ]
        )
    probe = ffprobe(out)
    actual_duration = float(probe["format"]["duration"])
    return {
        "video": str(out),
        "expected_duration_seconds": round(expected_duration, 3),
        "actual_duration_seconds": round(actual_duration, 3),
        "duration_error_seconds": round(actual_duration - expected_duration, 3),
        "probe": probe,
    }
