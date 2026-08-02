from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datavideo.manifest import load_config
from datavideo.review_db import latest_reviews_by_clip, save_review
from datavideo.schemas import read_json
from datavideo_multichart_v2.reviewed_outputs import CLIP_REVIEW_STAGE


st.set_page_config(page_title="Multichart V2 Clip Boundary Review", layout="wide")
st.warning(
    "Deprecated for current web-annotated multichart v2 workflow: webpage reference time is the visual clip boundary. "
    "This page is retained only for old records or future non-annotated experiments."
)


DECISION_LABELS = {
    "approved": "通过",
    "needs_revision": "需要修改",
    "excluded": "排除",
    "saved": "保存",
}


def _abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _load_run_report(processed_root: Path, generated: Path) -> dict[str, Any]:
    proposal_path = processed_root / "multichart_v2_proposal_report.json"
    if proposal_path.exists():
        return read_json(proposal_path)
    legacy_proposal_path = generated / "multichart_v2_proposal_report.json"
    if legacy_proposal_path.exists():
        return read_json(legacy_proposal_path)
    run_path = generated / "multichart_v2_run_report.json"
    if run_path.exists():
        return read_json(run_path)
    st.warning("Run multichart-propose-v2 first.")
    st.stop()


def _interval(value: dict[str, Any] | None) -> tuple[float, float] | None:
    if not value:
        return None
    return (float(value.get("start", 0.0)), float(value.get("end", 0.0)))


def _interval_row(name: str, intervals: dict[str, Any], key: str) -> dict[str, Any]:
    value = intervals.get(key)
    if not value:
        return {"name": name, "start": None, "end": None, "duration": None}
    start = float(value.get("start", 0.0))
    end = float(value.get("end", 0.0))
    return {"name": name, "start": start, "end": end, "duration": round(end - start, 3)}


def _load_sentences(processed_dir: Path) -> list[dict[str, Any]]:
    path = processed_dir / "narration" / "sentence_boundaries.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _sentence_text(sentences: list[dict[str, Any]], start: float, end: float) -> str:
    overlapping = [
        row
        for row in sentences
        if float(row.get("end_context_seconds", 0.0)) > start and float(row.get("start_context_seconds", 0.0)) < end
    ]
    return "\n".join(
        f"[{row.get('start_context_seconds')} - {row.get('end_context_seconds')}] {row.get('text', '')}"
        for row in overlapping
    )


cfg_path = st.sidebar.text_input("Config", os.environ.get("DATAVIDEO_REVIEW_CONFIG", "configs/multichart_assets_v2.yaml"))
cfg = load_config(ROOT / cfg_path)
generated = _abs(cfg.get("generated_root", cfg.get("generated_dir", "data/generated_v2")))
processed_root = _abs(cfg.get("processed_root", "data/processed"))
review_db = _abs(cfg["review_db"])
run_report = _load_run_report(processed_root, generated)

st.title(f"Clip Boundary Review: {cfg['sample_id']}")

reports = [report for report in run_report.get("clips", []) if isinstance(report, dict)]
clip_ids = [
    report.get("clip_id") or report.get("clip", {}).get("clip_id")
    for report in reports
    if report.get("clip_id") or report.get("clip", {}).get("clip_id")
]
selected_clip_id = st.sidebar.selectbox("Clip", clip_ids)
report = next(report for report in reports if (report.get("clip_id") or report.get("clip", {}).get("clip_id")) == selected_clip_id)
clip = report.get("clip", {"clip_id": selected_clip_id})
processed_dir = processed_root / selected_clip_id
intervals_path = processed_dir / "intervals.json"

if not intervals_path.exists():
    st.error(f"Missing intervals.json for {selected_clip_id}")
    st.stop()

intervals = read_json(intervals_path)
duration = float(intervals.get("context_duration_seconds", 0.0) or 0.0)
context_source_start = float(intervals.get("context_source", {}).get("start", 0.0) or 0.0)
proposed = _interval(intervals.get("proposed_clip_context")) or _interval(intervals.get("chart_context")) or (0.0, duration)
latest_clip_review = latest_reviews_by_clip(review_db, cfg["sample_id"], stage=CLIP_REVIEW_STAGE).get(selected_clip_id)
latest_review_value = latest_clip_review.get("reviewed_value", {}).get("clip", {}) if latest_clip_review else {}
reviewed = _interval(latest_review_value.get("reviewed_clip_context")) or proposed

if intervals.get("requires_context_redownload"):
    st.warning("This sample lacks true boundary context and is marked requires_context_redownload.")
if intervals.get("needs_review"):
    st.info("ASR sentence boundary proposal needs review.")

left, right = st.columns([1.15, 0.85])
with left:
    st.subheader("Context Video")
    context_video = processed_dir / "context.mp4"
    if context_video.exists():
        st.video(str(context_video))
    else:
        st.warning("context.mp4 is missing.")

    start_context = st.number_input(
        "Reviewed clip start in context seconds",
        value=float(reviewed[0]),
        min_value=0.0,
        max_value=max(duration, 0.001),
        step=0.1,
        format="%.3f",
    )
    end_context = st.number_input(
        "Reviewed clip end in context seconds",
        value=float(reviewed[1]),
        min_value=0.0,
        max_value=max(duration, 0.001),
        step=0.1,
        format="%.3f",
    )
    st.caption(
        f"Source absolute time: {context_source_start + start_context:.3f}s - "
        f"{context_source_start + end_context:.3f}s"
    )

with right:
    st.subheader("Intervals")
    interval_rows = [
        _interval_row("reference source", intervals, "reference_source"),
        _interval_row("context source", intervals, "context_source"),
        _interval_row("chart context", intervals, "chart_context"),
        _interval_row("proposed context", intervals, "proposed_clip_context"),
        {
            "name": "reviewed context",
            "start": reviewed[0],
            "end": reviewed[1],
            "duration": round(reviewed[1] - reviewed[0], 3),
        },
    ]
    st.dataframe(pd.DataFrame(interval_rows), use_container_width=True, hide_index=True)
    st.write(f"Boundary reason: {intervals.get('boundary_reason') or 'none'}")

sentences = _load_sentences(processed_dir)
with st.expander("ASR Sentence Boundaries", expanded=True):
    st.text(_sentence_text(sentences, start_context, end_context) or "No ASR sentence boundaries available.")
    if sentences:
        st.dataframe(pd.DataFrame(sentences), use_container_width=True, hide_index=True)

st.subheader("Audit")
decision_label = st.radio("Decision", list(DECISION_LABELS.values()), horizontal=True)
decision = next(key for key, label in DECISION_LABELS.items() if label == decision_label)
reviewer = st.text_input("Reviewer", "local")
notes = st.text_area("Notes")

if st.button("Submit Clip Boundary Review", type="primary"):
    if end_context <= start_context:
        st.error("Clip end must be greater than clip start.")
        st.stop()
    if decision == "approved" and intervals.get("requires_context_redownload"):
        st.error("This sample cannot be approved until true context audio/video is redownloaded. Save as draft/revision or exclude it.")
        st.stop()

    reviewed_clip_context = {"start": round(start_context, 3), "end": round(end_context, 3)}
    reviewed_clip_source = {
        "start": round(context_source_start + start_context, 3),
        "end": round(context_source_start + end_context, 3),
    }
    reviewed_value = {
        "clip_id": selected_clip_id,
        "clip": {
            "clip_id": selected_clip_id,
            "reviewed_clip_context": reviewed_clip_context,
            "reviewed_clip_source": reviewed_clip_source,
            "proposed_clip_context": intervals.get("proposed_clip_context"),
            "proposed_clip_source": intervals.get("proposed_clip_source"),
            "chart_context": intervals.get("chart_context"),
            "chart_source": intervals.get("chart_source"),
            "reference_source": intervals.get("reference_source"),
            "context_source": intervals.get("context_source"),
            "coordinate_system": "context seconds for *_context; source-video absolute seconds for *_source",
        },
        "narration": {
            "boundary_reason": intervals.get("boundary_reason"),
            "needs_review": intervals.get("needs_review"),
            "reviewed_sentence_text": _sentence_text(sentences, start_context, end_context),
        },
    }
    original_value = {"clip": clip, "intervals": intervals}
    save_review(
        review_db,
        {
            "sample_id": cfg["sample_id"],
            "stage": CLIP_REVIEW_STAGE,
            "decision": decision,
            "original_value": original_value,
            "reviewed_value": reviewed_value,
            "reviewer": reviewer,
            "notes": notes,
            "model_version": Path(os.environ.get(cfg.get("model", {}).get("env_var", "MODEL_PATH"), "")).name,
            "config_hash": cfg.get("config_hash"),
        },
    )
    st.success("Saved clip boundary review to review.db. Final reviewed outputs are written by multichart-review-v2 after all approvals.")
