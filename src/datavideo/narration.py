from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from .schemas import ensure_dir, read_json, write_json, write_jsonl


def _seconds(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _text_join(words: Iterable[str]) -> str:
    text = " ".join(word.strip() for word in words if str(word).strip())
    return " ".join(text.split())


@lru_cache(maxsize=4)
def _load_whisper_model(model_path: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_path, device=device, compute_type=compute_type)


def _asr_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("asr", {})


def _sentence_pause_threshold(cfg: dict[str, Any]) -> float:
    return _seconds(_asr_cfg(cfg).get("pause_threshold_seconds", 0.45))


def _sentence_margin(cfg: dict[str, Any]) -> float:
    return _seconds(_asr_cfg(cfg).get("sentence_safety_margin_seconds", 0.2))


def _max_boundary_expansion(cfg: dict[str, Any]) -> float:
    return _seconds(_asr_cfg(cfg).get("max_boundary_expansion_seconds", 5.0))


def _to_serializable_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for seg in segments:
        rows.append(
            {
                "id": seg.get("id"),
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": seg.get("text"),
                "avg_logprob": seg.get("avg_logprob"),
                "no_speech_prob": seg.get("no_speech_prob"),
                "words": seg.get("words", []),
            }
        )
    return rows


def _to_serializable_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "segment_id": row.get("segment_id"),
            "word": row.get("word"),
            "start": row.get("start"),
            "end": row.get("end"),
            "probability": row.get("probability"),
        }
        for row in words
    ]


def _model_info_dict(info: Any) -> dict[str, Any]:
    fields = ["language", "language_probability", "duration", "duration_after_vad", "all_language_probs"]
    return {field: getattr(info, field) for field in fields if hasattr(info, field)}


def _word_ends_sentence(word: str) -> bool:
    word = word.strip()
    return bool(word) and word[-1] in {".", "?", "!", "。", "？", "！"}


def _word_boundary_reason(current: dict[str, Any], next_word: dict[str, Any] | None, pause_threshold: float) -> str | None:
    if _word_ends_sentence(str(current.get("word", ""))):
        return "punctuation"
    if next_word is None:
        return "segment_end"
    gap = _seconds(next_word.get("start")) - _seconds(current.get("end"))
    if gap >= pause_threshold:
        return "pause"
    return None


def build_sentence_boundaries(
    segments: list[dict[str, Any]],
    words: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    context_duration_seconds: float,
) -> list[dict[str, Any]]:
    if not words:
        return []

    pause_threshold = _sentence_pause_threshold(cfg)
    max_expand = _max_boundary_expansion(cfg)
    rows: list[dict[str, Any]] = []
    start_index = 0
    sentence_index = 0

    for index, word in enumerate(words):
        next_word = words[index + 1] if index + 1 < len(words) else None
        reason = _word_boundary_reason(word, next_word, pause_threshold)
        if reason is None:
            continue
        sent_words = words[start_index : index + 1]
        if not sent_words:
            continue
        start = _seconds(sent_words[0].get("start"))
        end = _seconds(sent_words[-1].get("end"))
        avg_prob = sum(_seconds(item.get("probability")) for item in sent_words) / len(sent_words)
        needs_review = avg_prob < 0.55 or (end - start) > max_expand or start <= 0.0 or end >= context_duration_seconds
        rows.append(
            {
                "sentence_index": sentence_index,
                "start_context_seconds": round(start, 3),
                "end_context_seconds": round(end, 3),
                "start_source_seconds": None,
                "end_source_seconds": None,
                "text": _text_join(item.get("word", "") for item in sent_words),
                "confidence": round(avg_prob, 4),
                "boundary_reason": reason,
                "needs_review": bool(needs_review),
            }
        )
        sentence_index += 1
        start_index = index + 1

    if start_index < len(words):
        sent_words = words[start_index:]
        start = _seconds(sent_words[0].get("start"))
        end = _seconds(sent_words[-1].get("end"))
        avg_prob = sum(_seconds(item.get("probability")) for item in sent_words) / len(sent_words)
        rows.append(
            {
                "sentence_index": sentence_index,
                "start_context_seconds": round(start, 3),
                "end_context_seconds": round(end, 3),
                "start_source_seconds": None,
                "end_source_seconds": None,
                "text": _text_join(item.get("word", "") for item in sent_words),
                "confidence": round(avg_prob, 4),
                "boundary_reason": "fallback",
                "needs_review": True,
            }
        )

    for row in rows:
        row["needs_review"] = bool(row["needs_review"]) or row["confidence"] < 0.6
    return rows


def transcribe_context_audio(
    cfg: dict[str, Any],
    clip_id: str,
    audio_path: str | Path,
    intervals: dict[str, Any],
    processed_dir: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    processed_dir = ensure_dir(processed_dir)
    narration_dir = ensure_dir(processed_dir / "narration")
    raw_path = narration_dir / "context_transcript_raw.json"
    segments_path = narration_dir / "context_segments.jsonl"
    words_path = narration_dir / "context_words.jsonl"
    boundaries_path = narration_dir / "sentence_boundaries.jsonl"
    selected_path = narration_dir / "selected_full_sentences.jsonl"
    overlap_path = narration_dir / "overlap_words.jsonl"
    report_path = narration_dir / "asr_report.json"

    if raw_path.exists() and segments_path.exists() and words_path.exists() and boundaries_path.exists() and selected_path.exists() and overlap_path.exists() and report_path.exists() and not force:
        return read_json(report_path)
    if raw_path.exists() and segments_path.exists() and words_path.exists() and boundaries_path.exists() and not force:
        raw = read_json(raw_path)
        derived = write_reference_narration_outputs(narration_dir, raw, intervals, model_path=raw.get("model_path"))
        report = read_json(report_path) if report_path.exists() else {"clip_id": clip_id}
        report.update(derived)
        write_json(report_path, report)
        return report

    asr_cfg = _asr_cfg(cfg)
    model_env = asr_cfg.get("model_path_env", "WHISPER_MODEL_PATH")
    model_path = os.environ.get(model_env)
    if not model_path:
        raise RuntimeError(f"Missing ASR model path in environment variable {model_env}")

    model = _load_whisper_model(
        model_path,
        str(asr_cfg.get("device", "cuda")),
        str(asr_cfg.get("compute_type", "float16")),
    )
    segments_iter, info = model.transcribe(
        str(audio_path),
        word_timestamps=bool(asr_cfg.get("word_timestamps", True)),
        vad_filter=bool(asr_cfg.get("vad_filter", True)),
    )

    context_start = _seconds(intervals.get("context_source", {}).get("start"))
    segments: list[dict[str, Any]] = []
    words: list[dict[str, Any]] = []
    for idx, segment in enumerate(segments_iter):
        seg_words = []
        for word in getattr(segment, "words", []) or []:
            word_row = {
                "segment_id": idx,
                "word": word.word,
                "start": round(_seconds(word.start), 3),
                "end": round(_seconds(word.end), 3),
                "probability": round(_seconds(getattr(word, "probability", None)), 4) if getattr(word, "probability", None) is not None else None,
            }
            seg_words.append(word_row)
            words.append({"segment_id": idx, **word_row})
        segments.append(
            {
                "id": idx,
                "start": round(_seconds(segment.start), 3),
                "end": round(_seconds(segment.end), 3),
                "text": segment.text,
                "avg_logprob": getattr(segment, "avg_logprob", None),
                "no_speech_prob": getattr(segment, "no_speech_prob", None),
                "words": seg_words,
            }
        )

    context_duration_seconds = float(intervals.get("context_duration_seconds") or 0.0)
    boundaries = build_sentence_boundaries(segments, words, cfg, context_duration_seconds=context_duration_seconds)
    for row in boundaries:
        row["start_source_seconds"] = round(context_start + _seconds(row["start_context_seconds"]), 3)
        row["end_source_seconds"] = round(context_start + _seconds(row["end_context_seconds"]), 3)

    raw = {
        "clip_id": clip_id,
        "audio_path": str(audio_path),
        "model_path": model_path,
        "model_info": _model_info_dict(info),
        "intervals": intervals,
        "segments": _to_serializable_segments(segments),
        "words": _to_serializable_words(words),
        "sentence_boundaries": boundaries,
    }
    write_json(raw_path, raw)
    write_jsonl(segments_path, [
        {
            **row,
            "start_source_seconds": round(context_start + _seconds(row["start"]), 3),
            "end_source_seconds": round(context_start + _seconds(row["end"]), 3),
        }
        for row in segments
    ])
    write_jsonl(words_path, [
        {
            **row,
            "start_source_seconds": round(context_start + _seconds(row["start"]), 3),
            "end_source_seconds": round(context_start + _seconds(row["end"]), 3),
        }
        for row in words
    ])
    write_jsonl(boundaries_path, boundaries)
    derived = write_reference_narration_outputs(narration_dir, raw, intervals, model_path=model_path)
    report = {
        "clip_id": clip_id,
        "model_path": model_path,
        "model_info": _model_info_dict(info),
        "segment_count": len(segments),
        "word_count": len(words),
        "sentence_count": len(boundaries),
        "raw_path": str(raw_path),
        "segments_path": str(segments_path),
        "words_path": str(words_path),
        "boundaries_path": str(boundaries_path),
        "cache_enabled": bool(asr_cfg.get("cache", True)),
        **derived,
    }
    write_json(report_path, report)
    return report


def _sentence_for_time(boundaries: list[dict[str, Any]], timestamp: float) -> dict[str, Any] | None:
    for row in boundaries:
        if _seconds(row.get("start_context_seconds")) <= timestamp <= _seconds(row.get("end_context_seconds")):
            return row
    return None


def select_full_sentences_for_reference(
    boundaries: list[dict[str, Any]],
    words: list[dict[str, Any]],
    intervals: dict[str, Any],
) -> dict[str, Any]:
    reference = intervals.get("reference_source", {})
    context_source = intervals.get("context_source", {})
    visual = intervals.get("visual_clip_context", {})
    ref_start_source = _seconds(reference.get("start"))
    ref_end_source = _seconds(reference.get("end"))
    context_start_source = _seconds(context_source.get("start"))
    visual_start_context = _seconds(visual.get("start"))
    visual_end_context = _seconds(visual.get("end"))
    clip_duration = max(0.0, visual_end_context - visual_start_context)
    incomplete_context = bool(intervals.get("requires_context_redownload"))

    selected = []
    for row in boundaries:
        start_source = _seconds(row.get("start_source_seconds"))
        end_source = _seconds(row.get("end_source_seconds"))
        overlap_start = max(start_source, ref_start_source)
        overlap_end = min(end_source, ref_end_source)
        if overlap_end <= overlap_start:
            continue
        sentence_words = [
            word
            for word in words
            if _seconds(word.get("start")) >= _seconds(row.get("start_context_seconds")) - 1e-3
            and _seconds(word.get("end")) <= _seconds(row.get("end_context_seconds")) + 1e-3
        ]
        if sentence_words and not any(
            min(context_start_source + _seconds(word.get("end")), ref_end_source)
            > max(context_start_source + _seconds(word.get("start")), ref_start_source)
            for word in sentence_words
        ):
            continue
        sentence_index = row.get("sentence_index", len(selected))
        selected.append(
            {
                "sentence_id": f"sent_{int(sentence_index):03d}",
                "sentence_index": sentence_index,
                "text": row.get("text", ""),
                "start_source": round(start_source, 3),
                "end_source": round(end_source, 3),
                "start_context": row.get("start_context_seconds"),
                "end_context": row.get("end_context_seconds"),
                "overlap_source": {"start": round(overlap_start, 3), "end": round(overlap_end, 3)},
                "starts_before_visual_clip": start_source < ref_start_source,
                "ends_after_visual_clip": end_source > ref_end_source,
                "confidence": row.get("confidence"),
                "needs_review": bool(row.get("needs_review")) or incomplete_context,
                "incomplete_context": incomplete_context,
                "boundary_reason": row.get("boundary_reason"),
            }
        )

    overlap_words = []
    for word in words:
        start_context = _seconds(word.get("start"))
        end_context = _seconds(word.get("end"))
        start_source = context_start_source + start_context
        end_source = context_start_source + end_context
        overlap_start = max(start_source, ref_start_source)
        overlap_end = min(end_source, ref_end_source)
        if overlap_end <= overlap_start:
            continue
        overlap_words.append(
            {
                **word,
                "start_source": round(start_source, 3),
                "end_source": round(end_source, 3),
                "start_context": round(start_context, 3),
                "end_context": round(end_context, 3),
                "start_visual": round(max(0.0, min(clip_duration, overlap_start - ref_start_source)), 3),
                "end_visual": round(max(0.0, min(clip_duration, overlap_end - ref_start_source)), 3),
            }
        )

    status = "incomplete_context" if incomplete_context else ("needs_review" if any(row["needs_review"] for row in selected) else "provisional")
    return {
        "selected_full_sentences": selected,
        "overlap_words": overlap_words,
        "visual_subtitle_segments": _words_to_subtitle_segments(overlap_words, clip_duration),
        "status": status,
    }


def _words_to_subtitle_segments(words: list[dict[str, Any]], clip_duration: float) -> list[dict[str, Any]]:
    if not words:
        return []
    return [
        {
            "id": 0,
            "start": round(max(0.0, min(clip_duration, _seconds(words[0].get("start_visual")))), 3),
            "end": round(max(0.0, min(clip_duration, _seconds(words[-1].get("end_visual")))), 3),
            "text": _text_join(row.get("word", "") for row in words),
        }
    ]


def write_reference_narration_outputs(
    narration_dir: str | Path,
    raw_transcript: dict[str, Any],
    intervals: dict[str, Any],
    *,
    model_path: str | None = None,
) -> dict[str, Any]:
    narration_dir = ensure_dir(narration_dir)
    selected = select_full_sentences_for_reference(
        raw_transcript.get("sentence_boundaries", []),
        raw_transcript.get("words", []),
        intervals,
    )
    selected_path = narration_dir / "selected_full_sentences.jsonl"
    overlap_path = narration_dir / "overlap_words.jsonl"
    srt_path = narration_dir / "subtitles.srt"
    provenance_path = narration_dir / "transcript_provenance.json"
    write_jsonl(selected_path, selected["selected_full_sentences"])
    write_jsonl(overlap_path, selected["overlap_words"])
    srt_path.write_text(_build_srt(selected["visual_subtitle_segments"]), encoding="utf-8")
    provenance = {
        "text_source": "context_transcript_raw.json",
        "context_source": intervals.get("context_source"),
        "reference_source": intervals.get("reference_source"),
        "visual_clip_context": intervals.get("visual_clip_context"),
        "model_path": model_path or raw_transcript.get("model_path"),
        "narration_status": selected["status"],
        "rule": "full sentences use source/context time and may extend outside visual clip; subtitles use only visual-overlap words",
    }
    write_json(provenance_path, provenance)
    return {
        "selected_full_sentences_path": str(selected_path),
        "overlap_words_path": str(overlap_path),
        "subtitles_path": str(srt_path),
        "transcript_provenance_path": str(provenance_path),
        "narration_status": selected["status"],
        "selected_sentence_count": len(selected["selected_full_sentences"]),
        "overlap_word_count": len(selected["overlap_words"]),
    }


def expand_chart_interval_to_complete_speech(
    intervals: dict[str, Any],
    boundaries: list[dict[str, Any]],
    *,
    context_duration_seconds: float,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    chart = intervals.get("chart_context") or {}
    chart_start = _seconds(chart.get("start"))
    chart_end = _seconds(chart.get("end"))
    context_start = 0.0
    context_end = context_duration_seconds
    margin = _sentence_margin(cfg)
    max_expand = _max_boundary_expansion(cfg)
    needs_review = bool(intervals.get("requires_context_redownload"))
    reasons: list[str] = []

    start_sentence = _sentence_for_time(boundaries, chart_start)
    end_sentence = _sentence_for_time(boundaries, chart_end)

    proposed_start = chart_start
    proposed_end = chart_end

    if start_sentence and chart_start > _seconds(start_sentence["start_context_seconds"]) + 1e-3:
        proposed_start = max(context_start, _seconds(start_sentence["start_context_seconds"]) - margin)
        reasons.append("chart start falls inside a narration sentence")
    if end_sentence and chart_end < _seconds(end_sentence["end_context_seconds"]) - 1e-3:
        proposed_end = min(context_end, _seconds(end_sentence["end_context_seconds"]) + margin)
        reasons.append("chart end falls inside a narration sentence")

    min_start = max(context_start, chart_start - max_expand)
    max_end = min(context_end, chart_end + max_expand)
    if proposed_start < min_start:
        proposed_start = min_start
        needs_review = True
        reasons.append("start expansion reached max boundary expansion")
    if proposed_end > max_end:
        proposed_end = max_end
        needs_review = True
        reasons.append("end expansion reached max boundary expansion")

    if start_sentence and start_sentence.get("confidence", 0.0) < 0.6:
        needs_review = True
    if end_sentence and end_sentence.get("confidence", 0.0) < 0.6:
        needs_review = True

    if proposed_start <= context_start or proposed_end >= context_end:
        needs_review = True
    if not reasons:
        reasons.append("chart interval already lands on sentence boundaries or silence")

    proposed = {
        "start": round(max(context_start, proposed_start), 3),
        "end": round(min(context_end, proposed_end), 3),
    }
    source_offset = _seconds(intervals.get("context_source", {}).get("start"))
    proposed_source = {
        "start": round(source_offset + proposed["start"], 3),
        "end": round(source_offset + proposed["end"], 3),
    }
    intervals = {**intervals}
    intervals["proposed_clip_context"] = proposed
    intervals["proposed_clip_source"] = proposed_source
    intervals["boundary_reason"] = "; ".join(reasons)
    intervals["needs_review"] = bool(needs_review)
    return intervals


def slice_transcript(
    raw_transcript: dict[str, Any],
    reviewed_start_context: float,
    reviewed_end_context: float,
) -> dict[str, Any]:
    segments = raw_transcript.get("segments", [])
    words = raw_transcript.get("words", [])
    clipped_segments = []
    clipped_words = []

    for seg in segments:
        start = _seconds(seg.get("start"))
        end = _seconds(seg.get("end"))
        if end <= reviewed_start_context or start >= reviewed_end_context:
            continue
        clipped_segments.append(
            {
                **seg,
                "start": round(max(0.0, start - reviewed_start_context), 3),
                "end": round(max(0.0, min(reviewed_end_context, end) - reviewed_start_context), 3),
            }
        )

    for word in words:
        start = _seconds(word.get("start"))
        end = _seconds(word.get("end"))
        if end <= reviewed_start_context or start >= reviewed_end_context:
            continue
        clipped_words.append(
            {
                **word,
                "start": round(max(0.0, start - reviewed_start_context), 3),
                "end": round(max(0.0, min(reviewed_end_context, end) - reviewed_start_context), 3),
            }
        )

    text = _text_join(row.get("word", "") for row in clipped_words)
    return {
        "segments": clipped_segments,
        "words": clipped_words,
        "text": text,
    }


def _format_timestamp(seconds: float, sep: str) -> str:
    seconds = max(0.0, seconds)
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    sec = total_s % 60
    total_m = total_s // 60
    minute = total_m % 60
    hour = total_m // 60
    if sep == ",":
        return f"{hour:02d}:{minute:02d}:{sec:02d},{ms:03d}"
    return f"{hour:02d}:{minute:02d}:{sec:02d}.{ms:03d}"


def _build_srt(segments: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, seg in enumerate(segments, start=1):
        lines.append(str(idx))
        lines.append(f"{_format_timestamp(_seconds(seg.get('start')), ',')} --> {_format_timestamp(_seconds(seg.get('end')), ',')}")
        lines.append(str(seg.get("text", "")).strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_vtt(segments: list[dict[str, Any]]) -> str:
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_format_timestamp(_seconds(seg.get('start')), '.')} --> {_format_timestamp(_seconds(seg.get('end')), '.')}")
        lines.append(str(seg.get("text", "")).strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def write_transcript_bundle(
    clip_root: str | Path,
    raw_transcript: dict[str, Any],
    reviewed_start_context: float,
    reviewed_end_context: float,
    *,
    intervals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    clip_root = ensure_dir(clip_root)
    narration_dir = ensure_dir(clip_root / "narration")
    sliced = slice_transcript(raw_transcript, reviewed_start_context, reviewed_end_context)
    segments_path = narration_dir / "transcript_segments.jsonl"
    words_path = narration_dir / "transcript_words.jsonl"
    txt_path = narration_dir / "transcript.txt"
    srt_path = narration_dir / "subtitles.srt"
    vtt_path = narration_dir / "subtitles.vtt"

    source_offset = _seconds(intervals.get("context_source", {}).get("start")) if intervals else 0.0
    write_jsonl(
        segments_path,
        [
            {
                **row,
                "start_context_seconds": round(reviewed_start_context + _seconds(row.get("start")), 3),
                "end_context_seconds": round(reviewed_start_context + _seconds(row.get("end")), 3),
                "start_source_seconds": round(source_offset + reviewed_start_context + _seconds(row.get("start")), 3),
                "end_source_seconds": round(source_offset + reviewed_start_context + _seconds(row.get("end")), 3),
            }
            for row in sliced["segments"]
        ],
    )
    write_jsonl(
        words_path,
        [
            {
                **row,
                "start_context_seconds": round(reviewed_start_context + _seconds(row.get("start")), 3),
                "end_context_seconds": round(reviewed_start_context + _seconds(row.get("end")), 3),
                "start_source_seconds": round(source_offset + reviewed_start_context + _seconds(row.get("start")), 3),
                "end_source_seconds": round(source_offset + reviewed_start_context + _seconds(row.get("end")), 3),
            }
            for row in sliced["words"]
        ],
    )
    txt_path.write_text((sliced["text"] or "").strip() + "\n", encoding="utf-8")
    srt_path.write_text(_build_srt(sliced["segments"]), encoding="utf-8")
    vtt_path.write_text(_build_vtt(sliced["segments"]), encoding="utf-8")
    return {
        "segments_path": str(segments_path),
        "words_path": str(words_path),
        "txt_path": str(txt_path),
        "srt_path": str(srt_path),
        "vtt_path": str(vtt_path),
        "text": sliced["text"],
    }
