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

import re
from pathlib import Path
from typing import Any

from .dynamic_data import (
    axis_tick_keys,
    build_data_change_events,
    build_final_data_table,
    write_dynamic_outputs,
)


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_label(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _state_group_key(row: dict[str, Any]) -> str:
    return str(row.get("state_key") or row.get("state_label") or row.get("state_id") or "state")


def _row_rank(row: dict[str, Any], aligned_ids: set[str]) -> tuple[int, float]:
    """Higher is better: CV-aligned rows first, then CV-confirmed entities,
    then recovery confidence."""
    score = 0
    if str(row.get("source_type")) == "visual_frame_align":
        score += 2
    if str(row.get("entity_id") or "") in aligned_ids:
        score += 1
    return (score, float(row.get("confidence") or 0.0))


def clean_states(
    states: list[dict[str, Any]],
    aligned_bars: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove garbage rows from recovered dynamic states after CV alignment.

    The VLM recovery occasionally emits axis tick labels (0/50/100) as data
    values or hallucinated entities (e.g. "cycling" when the frame only has
    cyclists/drivers). CV-aligned bars are the ground truth for the keyframe
    state, so after alignment we deterministically drop:
      * values that every entity in a state shares (axis ticks);
      * duplicate labels inside one state (one entity, one value per state);
      * rows whose metric is another entity's label while the entity was not
        confirmed by CV alignment (metric/entity confusion).
    """
    if not states:
        return states
    aligned_ids = {
        str(b.get("entity_id") or "")
        for b in (aligned_bars or [])
        if b.get("entity_id")
    }
    # Axis ticks first, across the whole table: entities that share an
    # identical multi-value set at the same timestamp (e.g. every bar got
    # 0/50/100) are axis labels, even when the rows sit in different states.
    tick_keys = axis_tick_keys(states)
    if tick_keys:
        states = [
            r
            for r in states
            if (r.get("state_start"), str(r.get("entity_id") or "")) not in tick_keys
        ]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in states:
        groups.setdefault(_state_group_key(row), []).append(row)

    cleaned: list[dict[str, Any]] = []
    for rows in groups.values():
        entity_labels = {
            _normalize_label(r.get("entity") or r.get("entity_id"))
            for r in rows
            if r.get("entity") or r.get("entity_id")
        }

        # One value per entity label inside a state; prefer the CV-aligned row.
        best_by_label: dict[str, dict[str, Any]] = {}
        for r in rows:
            eid = str(r.get("entity_id") or "")
            label = _normalize_label(r.get("entity") or eid)
            if not label or label == "unknown":
                continue
            cur = best_by_label.get(label)
            if cur is None or _row_rank(r, aligned_ids) > _row_rank(cur, aligned_ids):
                best_by_label[label] = r
        rows = list(best_by_label.values())

        # Metric/entity confusion: a row whose metric is another entity's
        # label while the entity itself was not confirmed by CV is a
        # hallucination (e.g. entity "cycling" with metric "drivers").
        rows = [
            r
            for r in rows
            if _normalize_label(r.get("metric")) not in entity_labels
            or str(r.get("entity_id") or "") in aligned_ids
        ]
        cleaned.extend(rows)
    return cleaned


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
        row["value_type"] = str(bar.get("value_type") or "exact")
        row["source_type"] = "visual_frame_align"
        base_confidence = 0.7 if bar.get("value_estimated") else 0.85
        row["confidence"] = max(float(row.get("confidence") or 0.0), base_confidence)
        row["review_status"] = "machine"
        if bar.get("value_estimated"):
            row["needs_review"] = True
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

    states = clean_states(states, bars)
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
