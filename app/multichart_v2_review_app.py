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
from datavideo.review_db import save_review
from datavideo.schemas import read_json
from datavideo_multichart_v2.narration_review import (
    filter_reviewed_narration_sentences,
    load_narration_for_review,
    narration_full_text,
)
from datavideo_multichart_v2.reviewed_outputs import REVIEW_STAGE, apply_latest_reviews


st.set_page_config(page_title="Multichart V2 Review", layout="wide")

DECISION_LABELS = {
    "approved": "通过",
    "needs_revision": "需要修改",
    "excluded": "排除",
    "saved": "保存",
}

ANIMATION_TYPES = [
    "no_clear_animation",
    "bar_grow",
    "bar_shrink",
    "line_draw_upward",
    "line_draw_downward",
    "pie_or_donut_segments_appear",
    "map_region_highlight",
    "chart_type_transition",
    "element_appear",
    "element_disappear",
    "element_highlight",
    "other",
]


def _abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _load_run_report(generated: Path) -> dict[str, Any]:
    path = generated / "multichart_v2_run_report.json"
    if not path.exists():
        st.warning("Run `assets` first.")
        st.stop()
    return read_json(path)


def _clip_duration(report: dict[str, Any]) -> float:
    value = report.get("keyframes", {}).get("clip_context", {}).get("clip_duration_seconds")
    if value is not None:
        return float(value)
    intervals = report.get("intervals", {})
    visual = intervals.get("visual_clip_relative") or {}
    if visual:
        return float(visual.get("end", 0.0)) - float(visual.get("start", 0.0))
    return float(report.get("media", {}).get("duration", 0.0) or 0.0)


def _keyframe_options(clip_root: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = report.get("keyframes", {})
    options = []
    initial = manifest.get("assets", {}).get("initial") or clip_root / "keyframes" / "initial.png"
    options.append(
        {
            "name": "initial",
            "asset": str(initial),
            "timestamp": manifest.get("timestamps", {}).get("initial"),
            "source_frame_id": manifest.get("source_frame_id"),
        }
    )
    for state in manifest.get("states", []) or []:
        if isinstance(state, dict):
            options.append(
                {
                    "name": state.get("name") or Path(str(state.get("asset", "state"))).stem,
                    "asset": state.get("asset"),
                    "timestamp": state.get("timestamp"),
                    "source_frame_id": state.get("source_frame_id"),
                }
            )
    states_dir = clip_root / "keyframes" / "states"
    known = {str(_abs(option["asset"]).resolve()) for option in options if option.get("asset")}
    if states_dir.exists():
        for path in sorted(states_dir.glob("state_*.png")):
            if str(path.resolve()) not in known:
                options.append({"name": path.stem, "asset": str(path), "timestamp": None, "source_frame_id": None})
    return [option for option in options if option.get("asset") and _abs(option["asset"]).exists()]


def _load_chart_data(clip_root: Path) -> pd.DataFrame:
    csv_path = clip_root / "chart_data.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raw_path = clip_root / "chart_data_clip_raw.json"
    if raw_path.exists():
        raw = read_json(raw_path)
        rows = raw.get("response", {}).get("data", {}).get("rows", [])
        if rows:
            return pd.DataFrame(rows)
    return pd.DataFrame(columns=["label", "series", "value", "unit", "raw_text", "evidence_text", "source_frame", "time_seconds"])


def _load_animation(clip_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    path = clip_root / "animation_detection.json"
    if path.exists():
        return read_json(path)
    return report.get("animation_detection", {}) if isinstance(report.get("animation_detection"), dict) else {}


def _animation_actions_dataframe(animation_report: dict[str, Any]) -> pd.DataFrame:
    actions = animation_report.get("major_actions", []) if isinstance(animation_report.get("major_actions"), list) else []
    rows = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        timestamps = action.get("evidence_timestamps", [])
        rows.append(
            {
                "action": action.get("action"),
                "description": action.get("description"),
                "evidence_timestamps": ", ".join(str(value) for value in timestamps) if isinstance(timestamps, list) else "",
            }
        )
    return pd.DataFrame(rows, columns=["action", "description", "evidence_timestamps"])


def _clean_animation_actions(df: pd.DataFrame) -> list[dict[str, Any]]:
    actions = []
    for row in _clean_records(df):
        timestamps = []
        for value in str(row.get("evidence_timestamps", "") or "").split(","):
            try:
                timestamps.append(float(value.strip()))
            except ValueError:
                continue
        actions.append(
            {
                "action": row.get("action") or "other",
                "description": row.get("description") or "",
                "evidence_timestamps": timestamps,
            }
        )
    return actions


def _narration_dataframe(narration: dict[str, Any]) -> pd.DataFrame:
    rows = narration.get("sentences", []) if isinstance(narration.get("sentences"), list) else []
    return pd.DataFrame(rows, columns=["start", "end", "text", "confidence", "needs_review", "keep_in_reviewed", "source_start", "source_end"])


def _clean_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    df = df.where(pd.notnull(df), None)
    records = df.to_dict(orient="records")
    return [dict(row) for row in records if any(value not in (None, "") for value in row.values())]


cfg_path = st.sidebar.text_input("Config", os.environ.get("DATAVIDEO_REVIEW_CONFIG", "configs/multichart_assets_v2.yaml"))
cfg = load_config(ROOT / cfg_path)
generated = _abs(cfg.get("generated_root", cfg.get("generated_dir", "data/generated_v2")))
run_report = _load_run_report(generated)

st.title(f"Multichart V2 Review: {cfg['sample_id']}")

reports = [report for report in run_report.get("clips", []) if isinstance(report, dict)]
clip_ids = [report.get("clip", {}).get("clip_id") for report in reports]
clip_ids = [clip_id for clip_id in clip_ids if clip_id]
selected_clip_id = st.sidebar.selectbox("Clip", clip_ids)
report = next(report for report in reports if report.get("clip", {}).get("clip_id") == selected_clip_id)
clip = report.get("clip", {})
clip_root = generated / selected_clip_id
processed_root = _abs(cfg.get("processed_root", "data/processed")) / selected_clip_id
duration = _clip_duration(report)

left, right = st.columns([1, 1])
with left:
    st.subheader("Clip")
    clip_path = clip_root / "clip.mp4"
    if clip_path.exists():
        st.video(str(clip_path))
    start_seconds = st.number_input("Clip start seconds", value=0.0, min_value=0.0, max_value=max(duration, 0.001), step=0.1, format="%.3f")
    end_seconds = st.number_input("Clip end seconds", value=duration, min_value=0.0, max_value=max(duration, 0.001), step=0.1, format="%.3f")
    st.caption(f"Source time range: {clip.get('start_time')} - {clip.get('end_time')} | duration {duration:.3f}s")

with right:
    st.subheader("Semantic Preview")
    preview = clip_root / "semantic_preview.png"
    if preview.exists():
        st.image(str(preview), caption="semantic_preview.png", use_container_width=True)
    svg_report = read_json(clip_root / "svg_report.json") if (clip_root / "svg_report.json").exists() else {}
    with st.expander("Semantic SVG report", expanded=False):
        st.json(svg_report)

st.subheader("Final Keyframe")
keyframe_options = _keyframe_options(clip_root, report)
labels = [f"{option['name']} | {option.get('timestamp')}s | {option.get('source_frame_id')}" for option in keyframe_options]
selected_label = st.radio("Select keyframe", labels, horizontal=False) if labels else None
selected_keyframe = keyframe_options[labels.index(selected_label)] if selected_label else {}

cols = st.columns(min(4, max(1, len(keyframe_options))))
for idx, option in enumerate(keyframe_options):
    with cols[idx % len(cols)]:
        st.image(str(_abs(option["asset"])), caption=f"{option['name']} @ {option.get('timestamp')}s", use_container_width=True)

st.subheader("Chart Data")
df = _load_chart_data(clip_root)
edited = st.data_editor(df, num_rows="dynamic", use_container_width=True)

metadata = read_json(clip_root / "chart_metadata.json") if (clip_root / "chart_metadata.json").exists() else {}
validation = read_json(clip_root / "chart_data_validation.json") if (clip_root / "chart_data_validation.json").exists() else {}
with st.expander("Metadata and validation", expanded=False):
    st.json({"metadata": metadata, "validation": validation})

st.subheader("Animation")
animation_report = _load_animation(clip_root, report)
st.caption(
    f"{animation_report.get('frame_count', 0)} frames @ {animation_report.get('sample_fps', 0)} FPS | "
    f"{animation_report.get('visual_start', 0)}s - {animation_report.get('visual_end', 0)}s"
)
animation_related = st.checkbox(
    "Target chart related",
    value=bool(animation_report.get("is_target_chart_related", False)),
)
animation_description = st.text_area(
    "Overall description",
    value=str(animation_report.get("overall_description", "") or ""),
)
animation_confidence = st.number_input(
    "Animation confidence",
    min_value=0.0,
    max_value=1.0,
    value=float(animation_report.get("confidence", 0.0) or 0.0),
    step=0.01,
)
animation_df = _animation_actions_dataframe(animation_report)
edited_animation = st.data_editor(
    animation_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "action": st.column_config.SelectboxColumn("action", options=ANIMATION_TYPES),
    },
)
with st.expander("Animation model report", expanded=False):
    st.json(animation_report)

st.subheader("Narration")
narration = load_narration_for_review(processed_root, selected_clip_id)
st.caption(f"{narration.get('status', 'missing')} | source: {narration.get('machine_source') or 'missing'}")
narration_df = _narration_dataframe(narration)
st.caption("取消勾选 `keep_in_reviewed`，这句旁白就不会进入最终 reviewed。")
edited_narration = st.data_editor(
    narration_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "start": st.column_config.NumberColumn("start", format="%.3f"),
        "end": st.column_config.NumberColumn("end", format="%.3f"),
        "text": st.column_config.TextColumn("text", width="large"),
        "keep_in_reviewed": st.column_config.CheckboxColumn("keep_in_reviewed"),
    },
)
narration_sentences_for_preview = filter_reviewed_narration_sentences(_clean_records(edited_narration))
st.text_area("Narration full text preview", value=narration_full_text(narration_sentences_for_preview), height=90, disabled=True)
with st.expander("Narration provenance", expanded=False):
    st.json(narration.get("provenance", {}))

st.subheader("Audit")
decision_label = st.radio("Decision", list(DECISION_LABELS.values()), horizontal=True)
decision = next(key for key, label in DECISION_LABELS.items() if label == decision_label)
reviewer = st.text_input("Reviewer", "local")
notes = st.text_area("Notes")

if st.button("Submit Review", type="primary"):
    if end_seconds <= start_seconds:
        st.error("Clip end must be greater than clip start.")
        st.stop()
    chart_rows = _clean_records(edited)
    narration_sentences = filter_reviewed_narration_sentences(_clean_records(edited_narration))
    reviewed_value = {
        "clip_id": selected_clip_id,
        "clip": {
            "clip_id": selected_clip_id,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "source_start_time": clip.get("start_time"),
            "source_end_time": clip.get("end_time"),
        },
        "keyframe": selected_keyframe,
        "chart_data": chart_rows,
        "animation": {
            "clip_id": selected_clip_id,
            "target_chart_type": animation_report.get("target_chart_type"),
            "sample_fps": animation_report.get("sample_fps"),
            "frame_count": animation_report.get("frame_count"),
            "visual_start": animation_report.get("visual_start"),
            "visual_end": animation_report.get("visual_end"),
            "is_target_chart_related": animation_related,
            "overall_description": animation_description,
            "major_actions": _clean_animation_actions(edited_animation),
            "confidence": animation_confidence,
            "machine_report_path": str(clip_root / "animation_detection.json"),
        },
        "narration": {
            "clip_id": selected_clip_id,
            "status": narration.get("status", "missing"),
            "sentences": narration_sentences,
            "full_text": narration_full_text(narration_sentences),
            "machine_source": narration.get("machine_source", ""),
        },
    }
    original_value = {
        "clip": clip,
        "keyframes": report.get("keyframes", {}),
        "chart_data": df.where(pd.notnull(df), None).to_dict(orient="records"),
        "animation_detection": animation_report,
        "narration": narration,
    }
    save_review(
        ROOT / cfg["review_db"],
        {
            "sample_id": cfg["sample_id"],
            "stage": REVIEW_STAGE,
            "decision": decision,
            "original_value": original_value,
            "reviewed_value": reviewed_value,
            "reviewer": reviewer,
            "notes": notes,
            "model_version": Path(os.environ.get(cfg.get("model", {}).get("env_var", "MODEL_PATH"), "")).name,
            "config_hash": cfg.get("config_hash"),
        },
    )
    reviewed_dir = ROOT / cfg["reviewed_dir"]
    reviewed_dir.mkdir(parents=True, exist_ok=True)
    (reviewed_dir / "latest_review.json").write_text(
        json.dumps({"clip_id": selected_clip_id, "decision": decision, "value": reviewed_value, "notes": notes}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rebuild_report = apply_latest_reviews(cfg)
    st.success(
        f"Saved and rebuilt reviewed set: {rebuild_report['accepted_count']} accepted, "
        f"{rebuild_report['excluded_count']} excluded, {rebuild_report['unreviewed_count']} unreviewed"
    )

if st.button("Rebuild Reviewed From Latest Reviews"):
    rebuild_report = apply_latest_reviews(cfg)
    st.success(
        f"Rebuilt reviewed set: {rebuild_report['accepted_count']} accepted, "
        f"{rebuild_report['excluded_count']} excluded, {rebuild_report['unreviewed_count']} unreviewed"
    )
