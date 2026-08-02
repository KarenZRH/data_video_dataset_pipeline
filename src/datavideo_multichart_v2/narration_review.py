from __future__ import annotations

from pathlib import Path
from typing import Any

from datavideo.schemas import read_json, read_jsonl


def load_narration_for_review(processed_root: str | Path, clip_id: str | None = None) -> dict[str, Any]:
    """Load machine ASR narration in the sentence-first shape used by review."""
    processed_root = Path(processed_root)
    narration_dir = processed_root / "narration"
    provenance = _read_json_if_exists(narration_dir / "transcript_provenance.json")
    status = str(provenance.get("narration_status") or ("provisional" if narration_dir.exists() else "missing"))

    selected = _sentences_from_selected(narration_dir / "selected_full_sentences.jsonl")
    if selected:
        source = narration_dir / "selected_full_sentences.jsonl"
        sentences = selected
    else:
        segments = _sentences_from_segments(narration_dir / "context_segments.jsonl")
        if segments:
            source = narration_dir / "context_segments.jsonl"
            sentences = segments
        else:
            source = narration_dir / "context_transcript_raw.json"
            sentences = _sentences_from_raw(source)

    return {
        "clip_id": clip_id or processed_root.name,
        "status": status,
        "sentences": sentences,
        "full_text": " ".join(str(row.get("text", "")).strip() for row in sentences if row.get("text")).strip(),
        "machine_source": str(source) if source.exists() else "",
        "provenance": provenance,
    }


def clean_narration_sentences(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text", "") or "").strip()
        if not text:
            continue
        clean_rows.append(
            {
                "start": _optional_float(row.get("start")),
                "end": _optional_float(row.get("end")),
                "text": text,
            }
        )
    return clean_rows


def narration_full_text(sentences: list[dict[str, Any]]) -> str:
    return " ".join(str(row.get("text", "")).strip() for row in sentences if row.get("text")).strip()


def _sentences_from_selected(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for row in read_jsonl(path):
        rows.append(
            {
                "start": _optional_float(row.get("start_context", row.get("start"))),
                "end": _optional_float(row.get("end_context", row.get("end"))),
                "text": str(row.get("text", "") or "").strip(),
                "confidence": row.get("confidence"),
                "needs_review": row.get("needs_review"),
                "keep_in_reviewed": True,
                "source_start": row.get("start_source"),
                "source_end": row.get("end_source"),
            }
        )
    return [row for row in rows if row["text"]]


def _sentences_from_segments(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for row in read_jsonl(path):
        rows.append(
            {
                "start": _optional_float(row.get("start")),
                "end": _optional_float(row.get("end")),
                "text": str(row.get("text", "") or "").strip(),
                "confidence": row.get("avg_logprob"),
                "needs_review": None,
                "keep_in_reviewed": True,
                "source_start": row.get("start_source_seconds"),
                "source_end": row.get("end_source_seconds"),
            }
        )
    return [row for row in rows if row["text"]]


def _sentences_from_raw(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = read_json(path)
    text = str(raw.get("text", "") or "").strip()
    if text:
        return [{"start": None, "end": None, "text": text, "keep_in_reviewed": True}]
    segments = raw.get("segments", []) if isinstance(raw.get("segments"), list) else []
    return [
        {
            "start": _optional_float(segment.get("start")),
            "end": _optional_float(segment.get("end")),
            "text": str(segment.get("text", "") or "").strip(),
            "keep_in_reviewed": True,
        }
        for segment in segments
        if isinstance(segment, dict) and str(segment.get("text", "") or "").strip()
    ]


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def filter_reviewed_narration_sentences(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        keep = row.get("keep_in_reviewed")
        if keep is False:
            continue
        text = str(row.get("text", "") or "").strip()
        if not text:
            continue
        kept.append(
            {
                "start": _optional_float(row.get("start")),
                "end": _optional_float(row.get("end")),
                "text": text,
            }
        )
    return kept
