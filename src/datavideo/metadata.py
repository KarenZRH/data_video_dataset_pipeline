from __future__ import annotations

import csv
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .schemas import read_jsonl


def _parse_time(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.isdigit():
        return float(text)
    parts = text.split(":")
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return None
    if len(numbers) == 2:
        minutes, seconds = numbers
        return minutes * 60 + seconds
    if len(numbers) == 3:
        hours, minutes, seconds = numbers
        return hours * 3600 + minutes * 60 + seconds
    return None


def _format_time(seconds: float | None) -> str:
    if seconds is None:
        return ""
    whole = int(round(seconds))
    return f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}"


def _video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.endswith("youtu.be"):
        return parsed.path.strip("/")
    query_id = parse_qs(parsed.query).get("v", [""])[0]
    return query_id or parsed.path.rstrip("/").split("/")[-1]


def _clean_chart_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text or "unknown"


def read_clip_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    csv_path = cfg.get("clip_metadata_csv")
    if csv_path:
        rows = read_clip_csv(
            csv_path,
            require_time_bounds=bool(cfg.get("require_time_bounds", True)),
        )
    else:
        rows = read_jsonl(Path(cfg.get("raw_clips_jsonl", "data/raw/datavideo_clips.jsonl")))

    clip_id = cfg.get("clip_id")
    if clip_id:
        rows = [row for row in rows if str(row.get("output_stem") or row.get("clip_id")) == str(clip_id)]

    max_clips = cfg.get("max_clips")
    if max_clips is not None and not clip_id:
        rows = rows[: int(max_clips)]
    return rows


def read_clip_csv(path: str | Path, *, require_time_bounds: bool = True) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        source_rows = list(csv.DictReader(f))

    chart_counts: dict[str, int] = {}
    seen: set[tuple[str, str, str, str]] = set()
    rows: list[dict[str, Any]] = []
    for idx, source in enumerate(source_rows, start=1):
        url = str(source.get("Link") or source.get("youtube_url") or "").strip()
        if not url:
            continue
        start_seconds = _parse_time(source.get("Start"))
        end_seconds = _parse_time(source.get("End"))
        if require_time_bounds and (start_seconds is None or end_seconds is None or end_seconds <= start_seconds):
            continue

        chart_type = _clean_chart_type(source.get("ChartType"))
        title = str(source.get("Title") or "").strip()
        source_key = (_video_id(url), chart_type, _format_time(start_seconds), _format_time(end_seconds))
        if source_key in seen:
            continue
        seen.add(source_key)
        chart_counts[chart_type] = chart_counts.get(chart_type, 0) + 1
        chart_index = chart_counts[chart_type]
        output_stem = f"{chart_type}_{chart_index}"
        video_id = source_key[0]
        duration = None if start_seconds is None or end_seconds is None else max(0.0, end_seconds - start_seconds)
        rows.append(
            {
                "chart_type": chart_type,
                "chart_index": chart_index,
                "raw_video_title": title,
                "video_id": video_id,
                "youtube_url": url,
                "start_time": _format_time(start_seconds),
                "end_time": _format_time(end_seconds),
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "duration_seconds": duration,
                "channel": str(source.get("Author") or "").strip(),
                "year": int(source["Year"]) if str(source.get("Year") or "").isdigit() else None,
                "source_website": str(source.get("SourceWebsite") or "").strip(),
                "source_row_id": str(source.get("ID") or source.get("\ufeffID") or idx),
                "output_stem": output_stem,
                "output_path": f"data/raw/videos/{output_stem}.mp4",
            }
        )
    return rows
