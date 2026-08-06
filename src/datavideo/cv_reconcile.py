"""Reconcile recovered data tables with CV-aligned frame observations.

The CV alignment step detects the real bars in the keyframe and reads the
printed values. When the recovered data table is missing a visible bar (for
example Sub-Saharan Africa was dropped during Qwen recovery) or contains a
value that differs from the number actually printed in the frame, this module
merges the frame-read values back into the dynamic data for the keyframe's
state and rewrites the dynamic outputs (dynamic_data.json/csv,
final_data_table.csv, data_change_events.csv, data_events.jsonl).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .dynamic_data import (
    build_data_change_events,
    build_final_data_table,
    write_dynamic_outputs,
)


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _state_covers(row: dict[str, Any], ts: float) -> bool:
    start = _as_float(row.get("state_start"))
    if start is None:
        return False
    end = _as_float(row.get("state_end"))
    if end is None:
        end = start
    return start - 0.01 <= ts <= end + 0.01


def _nearest_state(states: list[dict[str, Any]], ts: float | None) -> dict[str, Any] | None:
    if ts is None or not states:
        return states[0] if states else None
    covering = [row for row in states if _state_covers(row, ts)]
    if covering:
        return covering[0]
    return min(
        states,
        key=lambda row: abs((_as_float(row.get("state_start")) or 0.0) - ts),
    )


def reconcile_dynamic_data(
    dynamic: dict[str, Any],
    cv_report: dict[str, Any],
    *,
    clip_id: str,
    keyframe_timestamp: float | None,
    image_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any] | None:
    """Merge frame-read bars into the dynamic data; returns None when nothing
    changed (e.g. no bars or no frame-read values)."""
    bars = cv_report.get("bars") or []
    states = [dict(row) for row in (dynamic.get("states") or [])]
    if not bars or not states:
        return None
    ts = _as_float(keyframe_timestamp)
    template = _nearest_state(states, ts)
    if template is None:
        return None

    changed = False
    updated_bar_count = 0
    skipped_bars: list[dict[str, Any]] = []
    for bar in bars:
        eid = str(bar.get("entity_id") or "")
        value = bar.get("value")
        if not eid:
            continue
        if not isinstance(value, (int, float)) or bar.get("value_plausible") is False:
            skipped_bars.append(
                {
                    "entity_id": eid,
                    "label": bar.get("label"),
                    "value_text": bar.get("value_text"),
                    "reason": bar.get("plausibility_message") or "no numeric value",
                }
            )
            continue
        label = str(bar.get("label") or eid)
        value_text = bar.get("value_text")
        row = next(
            (
                candidate
                for candidate in states
                if candidate.get("entity_id") == eid and (ts is None or _state_covers(candidate, ts))
            ),
            None,
        )
        if row is None:
            row = dict(template)
            row.update(
                {
                    "clip_id": clip_id,
                    "entity_id": eid,
                    "entity": label,
                    "state_start": ts if ts is not None else template.get("state_start"),
                    "state_end": ts if ts is not None else template.get("state_end"),
                }
            )
            states.append(row)
        row["value"] = value
        row["value_type"] = "exact"
        row["source_type"] = "visual_frame_align"
        row["confidence"] = max(float(row.get("confidence") or 0.0), 0.85)
        row["review_status"] = "machine"
        row["raw_text"] = value_text
        row["evidence_text"] = value_text or f"{value:g}"
        row["evidence_frames"] = [
            {
                "frame_id": Path(image_path).stem,
                "time_seconds": ts,
                "path": str(image_path),
            }
        ]
        row["evidence_sentence_id"] = None
        changed = True
        updated_bar_count += 1

    if not changed:
        return None

    final_table = build_final_data_table(states)
    change_events = build_data_change_events(states)
    numeric_fact_count = sum(1 for row in states if row.get("value") is not None)
    dynamic = {
        **dynamic,
        "clip_id": clip_id,
        "states": states,
        "final_data_table": final_table,
        "data_change_events": change_events,
        "numeric_fact_count": numeric_fact_count,
        "data_completeness": (
            "complete"
            if states and numeric_fact_count == len(states)
            else "partial" if states else "none"
        ),
        "data_change_count": len(change_events),
        "include_in_dataset": True,
        "excluded": False,
        "exclude_reason": None,
    }
    written = write_dynamic_outputs(out_dir, dynamic)
    return {
        "dynamic": dynamic,
        "written": written,
        "updated_bar_count": updated_bar_count,
        "skipped_bar_count": len(skipped_bars),
        "skipped_bars": skipped_bars,
        "state_id": template.get("state_id"),
        "state_key": template.get("state_key"),
    }
