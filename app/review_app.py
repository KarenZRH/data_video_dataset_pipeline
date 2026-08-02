from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datavideo.manifest import load_config
from datavideo.review_db import save_review
from datavideo.reviewed_outputs import apply_latest_reviews
from datavideo.schemas import read_json, read_jsonl


st.set_page_config(page_title="Data Video Review", layout="wide")

cfg_path = st.sidebar.text_input("Config", os.environ.get("DATAVIDEO_REVIEW_CONFIG", "configs/stage1_bar.yaml"))
cfg = load_config(ROOT / cfg_path)
generated = ROOT / cfg["generated_dir"]

st.title(f"Review: {cfg['sample_id']}")

run_report_path = generated / "run_report.json"
if not run_report_path.exists():
    st.warning("Run the pipeline first: python -m datavideo.cli bar-assets --config configs/stage1_bar.yaml")
    st.stop()

run_report = read_json(run_report_path)
final_clips_path = generated / "final_bar_clips.jsonl"
clips_path = generated / "refined_clips.jsonl"
clips = read_jsonl(final_clips_path) if final_clips_path.exists() else read_jsonl(clips_path)
clip = clips[0] if clips else run_report.get("selected_clip", {})
clip_ids = [row.get("clip_id", f"clip_{idx:03d}") for idx, row in enumerate(clips)] or ["clip_000"]
selected_clip_id = st.sidebar.selectbox("Clip", clip_ids)
clip = next((row for row in clips if row.get("clip_id") == selected_clip_id), clip)
clip_root = generated / "clips" / selected_clip_id

left, right = st.columns([1, 1])
with left:
    st.subheader("Clip")
    clip_path = clip_root / "clip.mp4"
    if not clip_path.exists():
        clip_path = generated / "clip.mp4"
    if clip_path.exists():
        st.video(str(clip_path))
    start = st.number_input("Start", value=float(clip.get("start", 0.0)), step=0.1, format="%.3f")
    end = st.number_input("End", value=float(clip.get("end", 0.0)), step=0.1, format="%.3f")

with right:
    st.subheader("Trace Report")
    svg_report_path = clip_root / "svg_report.json"
    svg_report = read_json(svg_report_path) if svg_report_path.exists() else {}
    st.json(svg_report)

st.subheader("Initial Frame")
initial_path = clip_root / "keyframes" / "initial.png"
if initial_path.exists():
    st.image(str(initial_path), caption="initial.png", use_container_width=True)

st.subheader("Initial vs Trace Preview")
cols = st.columns(2)
with cols[0]:
    if initial_path.exists():
        st.image(str(initial_path), caption="initial.png", use_container_width=True)
with cols[1]:
    preview = clip_root / "trace_preview.png"
    if not preview.exists():
        preview = generated / "trace_preview.png"
    if preview.exists():
        st.image(str(preview), caption="trace_preview.png", use_container_width=True)

st.subheader("Recovered Data")
csv_path = clip_root / "chart_data.csv"
if not csv_path.exists():
    csv_path = generated / "chart_data.csv"
if csv_path.exists():
    df = pd.read_csv(csv_path)
else:
    df = pd.DataFrame([{"index": 0, "label": None, "value": None}])
edited = st.data_editor(df, num_rows="dynamic", use_container_width=True)

metadata_path = clip_root / "chart_metadata.json"
validation_path = clip_root / "chart_data_validation.json"
if not metadata_path.exists():
    metadata_path = generated / "chart_metadata.json"
if not validation_path.exists():
    validation_path = generated / "chart_data_validation.json"
metadata = read_json(metadata_path) if metadata_path.exists() else {}
validation = read_json(validation_path) if validation_path.exists() else {}
with st.expander("Metadata and Validation", expanded=False):
    st.json({"metadata": metadata, "validation": validation})

st.subheader("Audit")
decision = st.radio("Decision", ["通过", "需要修改", "排除", "保存"], horizontal=True)
reviewer = st.text_input("Reviewer", "local")
notes = st.text_area("Notes")

if st.button("Submit Review", type="primary"):
    reviewed_value = {
        "clip_id": selected_clip_id,
        "clip": {"start": start, "end": end},
        "chart_data": edited.to_dict(orient="records"),
        "keyframe": "initial.png",
    }
    original_value = {
        "clip": clip,
        "chart_data": df.to_dict(orient="records"),
    }
    save_review(
        ROOT / cfg["review_db"],
        {
            "sample_id": cfg["sample_id"],
            "stage": "stage1_review",
            "decision": decision,
            "original_value": original_value,
            "reviewed_value": reviewed_value,
            "reviewer": reviewer,
            "notes": notes,
            "model_version": Path(run_report.get("model_path") or "").name,
            "config_hash": cfg["config_hash"],
        },
    )
    reviewed_dir = ROOT / cfg["reviewed_dir"]
    reviewed_dir.mkdir(parents=True, exist_ok=True)
    (reviewed_dir / "latest_review.json").write_text(
        json.dumps(
            {"clip_id": selected_clip_id, "decision": decision, "value": reviewed_value, "notes": notes},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = apply_latest_reviews(cfg)
    st.success(f"Saved and rebuilt reviewed set: {report['accepted_count']} accepted, {report['excluded_count']} excluded")

if st.button("Rebuild Reviewed From Latest Reviews"):
    report = apply_latest_reviews(cfg)
    st.success(f"Rebuilt reviewed set: {report['accepted_count']} accepted, {report['excluded_count']} excluded")
