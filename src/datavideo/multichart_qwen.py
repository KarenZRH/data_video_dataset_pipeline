from __future__ import annotations

import json
import re
from typing import Any

from datavideo.qwen_vl import QwenVLClient


KEYFRAME_SCORE_PROMPT = """Score this single video frame as a static representative keyframe for SVG tracing.

Clip context:
__CLIP_CONTEXT__

The target chart type is fixed by the context. Do not give a high score to a different chart type.

Choose frames that are useful as a static visual asset, not necessarily the first frame.
A good keyframe:
1. Matches the target chart type and target visual scene.
2. Shows the most complete/final chart state available in the clip.
3. Shows all visible data marks and printed text clearly.
4. Is not a transition, scene cut, cropped close-up, title card, decorative illustration, or wrong chart type.
5. Does not cut off important data marks/text at the frame edges.

Type-specific guidance:
- bar: require clear bar marks and category labels. If more bars/categories appear later, early partial states are incomplete.
- line/area/scatter: require the plotted marks/series and visible labels. If values/labels reveal over time, prefer the final complete state.
- pie/donut: require a circular pie/donut structure, segments/arcs, and printed percentages/labels when present.
- treemap: require multiple rectangular regions and their main labels; partial reveal states are incomplete.
- map: prefer frames where geographic regions, encoded colors/symbols, legend, and labels are readable.
- timeline: require an ordered path/line/arrow with multiple event/time labels. A map or title card is not a timeline.
- sankey: require visible nodes, flows, and labels. Penalize flows or nodes cut off by frame edges.
- pictograph: prefer frames with visible icon units, categories, labels, and legend.
- combined: prefer the frame where the combined chart encodings are simultaneously most readable.

Return strict JSON only:
{
  "target_chart_type_match": boolean,
  "scene_change_or_title_card": boolean,
  "structure_complete": boolean,
  "final_or_most_complete_state": boolean,
  "data_marks_readable": boolean,
  "printed_text_readable": boolean,
  "edge_crop_or_occlusion": boolean,
  "has_directly_printed_values": boolean,
  "completeness": number,
  "state_finality": number,
  "edge_integrity": number,
  "data_text_visibility": number,
  "motion_score": number,
  "state_summary": string,
  "reason": string
}
"""

CLIP_DATA_PROMPT = """Recover chart data from this sequence of video frames sampled from one data-video clip.

Chart context:
__CHART_CONTEXT__

Frame context:
__FRAME_CONTEXT__

Rules:
- Return only strict JSON.
- Recover data from the whole clip sequence, not only one frame.
- Extract only values directly printed in the frames: numbers, percentages, dates, years, ranks, counts, currency, labels, and units.
- HARD CONSTRAINT: do not infer or estimate numeric values. Never convert bar length, line height, area size, pie/donut angle, map color, pictograph icon count, treemap area, or geometry into numbers.
- A row is valid only when its numeric value is visible as printed text in at least one frame. Put that exact visible text in raw_text and evidence_text.
- Only include multiple rows for the same label/series when the printed data value itself changes. Do not create rows for bar entrance, line drawing, zooming, highlighting, or other pure animation interpolation if the printed value is unchanged.
- If the same printed data value appears in several consecutive frames, keep at most representative rows with source_frame/time_seconds evidence; downstream code will merge consecutive identical states.
- If labels are visible but no numeric values are printed, set has_extractable_data=false. Do not create approximate relative percentages such as 20/50/100.
- If a number is printed in a label/callout/legend/node/segment/bar/category, include it.
- If a printed value appears in multiple frames, deduplicate it.
- If the clip shows changing years/states, keep the year/state in every row when visible. For bar charts with a changing time state, extract every distinct printed year/state that is visible enough to read, including at least the first and last states.
- For each distinct printed year/state, return a complete set of rows for all visible entities in that state. Do not split one year/state into several states just because bars appear one after another.
- If the same chart changes from one printed year/state to another, preserve rows for both years/states even when labels are identical.
- If labels are visible but values require visual estimation or counting, set needs_manual_data=true and create manual_stub_rows.
- Preserve units exactly as shown when visible.
- For pictographs, read numbers in titles, annotations, legends, icon labels, and explanatory text. Do not count icons unless a printed unit-per-icon rule and count are both visible.
- For treemaps, read printed numbers/percentages inside rectangles, labels, legends, or callouts. Do not estimate rectangle areas.

Schema:
{
  "has_extractable_data": boolean,
  "needs_manual_data": boolean,
  "chart_type": string,
  "title": null,
  "unit": null,
  "x_axis": null,
  "y_axis": null,
  "series": [],
  "temporal_change": boolean,
  "states": [
    {
      "state": null,
      "year": null,
      "source_frame": null,
      "time_seconds": null,
      "rows": []
    }
  ],
  "rows": [
    {
      "state": null,
      "year": null,
      "label": null,
      "series": null,
      "x": null,
      "y": null,
      "value": null,
      "unit": null,
      "raw_text": null,
      "evidence_text": null,
      "source_frame": null,
      "time_seconds": null,
      "confidence": null
    }
  ],
  "manual_stub_rows": [
    {
      "label": null,
      "series": null,
      "x": null,
      "unit": null,
      "reason": null,
      "source_frame": null,
      "time_seconds": null
    }
  ],
  "visible_text": [],
  "uncertain_fields": [],
  "skip_reason": null,
  "notes": ""
}
"""

ANIMATION_PROMPT = """Compare the complete ordered frame sequence from one video clip and describe changes related to its target chart type.

Clip context:
__CLIP_CONTEXT__

Ordered frames:
__FRAME_CONTEXT__

Focus on the target_chart_type in Clip context. These are sampled still frames, so a difference between ordered frames is animation evidence; motion blur is not required.

Before deciding is_target_chart_related, answer the three concrete visual-change questions at the start of the JSON. Directly compare the same target data marks across the sequence:
- First separate chart identities using title, category labels, axes, and overall layout. Never compare bars or lines from different chart identities as if one grew or shrank into the other. An isolated boundary frame from a previous or next chart must not hide animation within the main repeated chart scene.
- For bars, compare their absolute length or height, not only their rank. Read printed values and year/time/state labels when visible. Unchanged category order or ranking does not mean the bars are static. A longer/taller bar is bar_grow and a shorter/lower bar is bar_shrink.
- For lines, check whether a target line is progressively drawn, extended, shortened, or moved.
- Also check whether target marks, axes, labels, legends, or annotations appear, disappear, or become highlighted.

Set is_target_chart_related=true whenever any target chart component changes. Set it to false only if the target chart is absent or all of its components remain visually unchanged after comparison. Ignore subtitles, logos, people, backgrounds, camera motion, decorative motion, and unrelated chart types. Unusual illustrations around data marks do not make the data-mark changes irrelevant.

The fields must be consistent: if target_mark_dimensions_change, printed_values_or_time_states_change, or target_components_appear_or_disappear is true, then is_target_chart_related must also be true.

Write overall_description and every action description in Chinese. overall_description must be exactly one concise sentence. major_actions may be empty or contain any number of genuinely distinct actions; do not split actions to fill a quota. Merge bars that change together in the same direction into one action. Use an action from allowed_animation_types when applicable. Use at most three timestamps from Ordered frames as evidence for each action.

Before returning, verify action-direction consistency: bar_grow is valid only when bar length/height or its printed value increases; bar_shrink is valid only when it decreases. A description containing decrease, shrink, shorter, lower, 减少, 缩短, 下降, or 降低 must not use bar_grow. When several bars decrease together, return one collective bar_shrink action rather than separate actions for each category.

Return strict JSON only:
{
  "target_marks_visible": boolean,
  "target_mark_dimensions_change": boolean,
  "printed_values_or_time_states_change": boolean,
  "target_components_appear_or_disappear": boolean,
  "is_target_chart_related": boolean,
  "overall_description": string,
  "major_actions": [
    {
      "action": string,
      "description": string,
      "evidence_timestamps": [number]
    }
  ],
  "confidence": number
}
"""


def _json_from_text(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.S)
    payload = match.group(0) if match else text
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        repaired = _repair_json_payload(payload)
        return json.loads(repaired)


def _repair_json_payload(payload: str) -> str:
    repaired = payload.strip()
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(r"}\s*\n\s*{", "},\n{", repaired)
    repaired = re.sub(r"]\s*\n\s*\"", "],\n\"", repaired)
    repaired = re.sub(r"}\s*\n\s*\"", "},\n\"", repaired)
    return repaired


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _normalize_keyframe_score(result: dict[str, Any]) -> dict[str, Any]:
    complete_chart = _as_bool(result.get("structure_complete", result.get("complete_chart", False)))
    target_match = _as_bool(result.get("target_chart_type_match", result.get("same_chart", False)))
    crop = _as_bool(result.get("edge_crop_or_occlusion", False))
    return {
        "target_chart_type_match": target_match,
        "same_chart": target_match,
        "scene_change_or_title_card": _as_bool(result.get("scene_change_or_title_card", result.get("scene_change", False))),
        "scene_change": _as_bool(result.get("scene_change_or_title_card", result.get("scene_change", False))),
        "structure_complete": complete_chart,
        "complete_chart": complete_chart,
        "final_or_most_complete_state": _as_bool(result.get("final_or_most_complete_state", False)),
        "data_marks_readable": _as_bool(result.get("data_marks_readable", complete_chart)),
        "printed_text_readable": _as_bool(result.get("printed_text_readable", result.get("labels_readable", False))),
        "labels_readable": _as_bool(result.get("printed_text_readable", result.get("labels_readable", False))),
        "edge_crop_or_occlusion": crop,
        "has_directly_printed_values": _as_bool(result.get("has_directly_printed_values", False)),
        "completeness": _clamp01(result.get("completeness", 0.0)),
        "state_finality": _clamp01(result.get("state_finality", 0.0)),
        "edge_integrity": _clamp01(result.get("edge_integrity", 0.0 if crop else 1.0)),
        "data_text_visibility": _clamp01(result.get("data_text_visibility", 0.0)),
        "chart_identity_consistency": _clamp01(result.get("chart_identity_consistency", 1.0 if target_match else 0.0)),
        "data_extraction_suitability": _clamp01(result.get("data_text_visibility", result.get("data_extraction_suitability", 0.0))),
        "staticness": _clamp01(result.get("staticness", 1.0)),
        "motion_score": _clamp01(result.get("motion_score", 1.0), default=1.0),
        "state_summary": str(result.get("state_summary", "") or ""),
        "reason": str(result.get("reason", "") or ""),
    }


def _flatten_chart_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    direct_rows = result.get("rows") if isinstance(result.get("rows"), list) else []
    rows.extend(row for row in direct_rows if isinstance(row, dict))
    grouped_states = result.get("states") if isinstance(result.get("states"), list) else []
    for state in grouped_states:
        if not isinstance(state, dict):
            continue
        state_rows = state.get("rows") if isinstance(state.get("rows"), list) else []
        for row in state_rows:
            if not isinstance(row, dict):
                continue
            rows.append(
                {
                    **row,
                    "state": row.get("state", state.get("state")),
                    "year": row.get("year", state.get("year")),
                    "source_frame": row.get("source_frame", state.get("source_frame")),
                    "time_seconds": row.get("time_seconds", state.get("time_seconds")),
                }
            )
    return rows


def _row_dedupe_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("year"),
        row.get("state"),
        row.get("label"),
        row.get("series"),
        row.get("x"),
        row.get("y"),
        row.get("value"),
        row.get("unit"),
        row.get("raw_text"),
        row.get("source_frame"),
        row.get("time_seconds"),
    )


def _looks_like_chart_title_entity(row: dict[str, Any], result: dict[str, Any]) -> bool:
    label = str(row.get("label") or row.get("series") or "").strip().lower()
    if not label:
        return False
    title = str(result.get("title") or "").strip().lower()
    y_axis = str(result.get("y_axis") or "").strip().lower()
    if title and label == title:
        return True
    if y_axis and re.fullmatch(re.escape(y_axis) + r"\s+\d{4}", label):
        return True
    return False


def _normalize_chart_data(result: dict[str, Any], chart_type: str) -> dict[str, Any]:
    rows = _flatten_chart_rows(result)
    visible_text = result.get("visible_text") if isinstance(result.get("visible_text"), list) else []
    normalized_rows = []
    seen_rows = set()
    for row in rows:
        if isinstance(row, dict):
            if _looks_like_chart_title_entity(row, result):
                continue
            candidate = (
                {
                    "state": row.get("state"),
                    "year": row.get("year"),
                    "label": row.get("label"),
                    "series": row.get("series"),
                    "x": row.get("x"),
                    "y": row.get("y"),
                    "value": row.get("value"),
                    "unit": row.get("unit"),
                    "raw_text": row.get("raw_text"),
                    "evidence_text": row.get("evidence_text", row.get("raw_text")),
                    "source_frame": row.get("source_frame"),
                    "time_seconds": row.get("time_seconds"),
                    "confidence": row.get("confidence"),
                }
            )
            if _has_direct_numeric_evidence(candidate, visible_text):
                candidate["evidence_text"] = _best_evidence_text(candidate, visible_text)
                dedupe_key = _row_dedupe_key(candidate)
                if dedupe_key in seen_rows:
                    continue
                seen_rows.add(dedupe_key)
                normalized_rows.append(candidate)
    has_extractable = _as_bool(result.get("has_extractable_data", bool(normalized_rows)))
    if not normalized_rows:
        has_extractable = False
    manual_rows = result.get("manual_stub_rows") if isinstance(result.get("manual_stub_rows"), list) else []
    return {
        "has_extractable_data": has_extractable,
        "needs_manual_data": _as_bool(result.get("needs_manual_data", bool(manual_rows))),
        "chart_type": str(result.get("chart_type", chart_type) or chart_type),
        "title": result.get("title"),
        "unit": result.get("unit"),
        "x_axis": result.get("x_axis"),
        "y_axis": result.get("y_axis"),
        "series": result.get("series") if isinstance(result.get("series"), list) else [],
        "temporal_change": _as_bool(result.get("temporal_change", False)),
        "rows": normalized_rows if has_extractable else [],
        "manual_stub_rows": manual_rows,
        "visible_text": visible_text,
        "uncertain_fields": result.get("uncertain_fields") if isinstance(result.get("uncertain_fields"), list) else [],
        "skip_reason": result.get("skip_reason") if not has_extractable else None,
        "notes": str(result.get("notes", "") or ""),
    }


ANIMATION_TYPES = {
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
}


def _normalize_animation(result: dict[str, Any]) -> dict[str, Any]:
    target_marks_visible = _as_bool(result.get("target_marks_visible", False))
    dimensions_change = _as_bool(result.get("target_mark_dimensions_change", False))
    values_or_states_change = _as_bool(result.get("printed_values_or_time_states_change", False))
    components_change = _as_bool(result.get("target_components_appear_or_disappear", False))
    related = _as_bool(result.get("is_target_chart_related", False)) or any(
        (dimensions_change, values_or_states_change, components_change)
    )
    raw_actions = result.get("major_actions") if isinstance(result.get("major_actions"), list) else []
    actions = []
    for item in raw_actions:
        if not isinstance(item, dict):
            continue
        timestamps = item.get("evidence_timestamps") if isinstance(item.get("evidence_timestamps"), list) else []
        clean_timestamps = []
        for value in timestamps:
            try:
                timestamp = round(float(value), 3)
            except (TypeError, ValueError):
                continue
            if timestamp not in clean_timestamps:
                clean_timestamps.append(timestamp)
        action = str(item.get("action", "other") or "other")
        if action not in ANIMATION_TYPES or action == "no_clear_animation":
            action = "other"
        actions.append(
            {
                "action": action,
                "description": str(item.get("description", "") or ""),
                "evidence_timestamps": clean_timestamps,
            }
        )
    return {
        "target_marks_visible": target_marks_visible,
        "target_mark_dimensions_change": dimensions_change,
        "printed_values_or_time_states_change": values_or_states_change,
        "target_components_appear_or_disappear": components_change,
        "is_target_chart_related": related,
        "overall_description": str(result.get("overall_description", "") or ""),
        "major_actions": actions if related else [],
        "confidence": _clamp01(result.get("confidence", 0.0)),
    }


def _has_direct_numeric_evidence(row: dict[str, Any], visible_text: list[Any] | None = None) -> bool:
    numeric_values = [row.get(key) for key in ("value", "x", "y") if row.get(key) not in (None, "")]
    if not numeric_values:
        return False
    evidence = " ".join(
        str(row.get(key) or "")
        for key in ("raw_text", "evidence_text", "label", "series", "unit")
    )
    visible_evidence = " ".join(str(item or "") for item in (visible_text or []))
    evidence = f"{evidence} {visible_evidence}"
    has_printed_number = bool(
        re.search(
            r"(?<![A-Za-z])[$€£¥]?\s*[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*(?:%|percent|million|billion|thousand|k|m|bn|km|mi|kg|g|x)?",
            evidence,
            re.I,
        )
    )
    if not has_printed_number:
        return False
    value_text = str(row.get("value") or "").replace(",", "").strip()
    label_text = str(row.get("label") or "").replace(",", "").strip()
    x_text = str(row.get("x") or "").replace(",", "").strip()
    unit_text = str(row.get("unit") or "").lower()
    raw_text = f"{row.get('raw_text') or ''} {row.get('evidence_text') or ''}".lower()
    if value_text and value_text in {label_text, x_text} and re.fullmatch(r"\d{4}", value_text):
        return False
    if unit_text in {"$", "usd", "dollar", "dollars"} and value_text:
        currency_pattern = rf"(?:[$€£¥]\s*{re.escape(value_text)}|{re.escape(value_text)}\s*(?:dollars?|usd)\b)"
        if not re.search(currency_pattern, raw_text.replace(",", ""), re.I):
            return False
    normalized_evidence = evidence.replace(",", "")
    for value in numeric_values:
        normalized_value = str(value).replace(",", "").strip()
        if normalized_value and re.search(re.escape(normalized_value), normalized_evidence):
            return True
    return False


def _best_evidence_text(row: dict[str, Any], visible_text: list[Any]) -> str | None:
    numeric_values = [row.get(key) for key in ("value", "x", "y") if row.get(key) not in (None, "")]
    candidates = [row.get("evidence_text"), row.get("raw_text"), *visible_text]
    for value in numeric_values:
        normalized_value = str(value).replace(",", "").strip()
        for candidate in candidates:
            text = str(candidate or "")
            if normalized_value and re.search(re.escape(normalized_value), text.replace(",", "")):
                return text
    return row.get("evidence_text") or row.get("raw_text")


class MultichartQwenClient:
    def __init__(self, cfg: dict[str, Any]):
        self.base = QwenVLClient(cfg)
        self.cfg = cfg

    @property
    def model_path(self) -> str | None:
        return self.base.model_path

    def score_keyframe_candidate(self, image_path: str, clip_context: dict[str, Any]) -> dict[str, Any]:
        if self.base.load():
            try:
                prompt = KEYFRAME_SCORE_PROMPT.replace(
                    "__CLIP_CONTEXT__", json.dumps(clip_context, ensure_ascii=False, indent=2)
                )
                raw = self.base._generate([image_path], prompt, max_new_tokens=224)
                result = _normalize_keyframe_score(_json_from_text(raw))
                return {"result": result, "raw_response": raw, "model_status": "qwen", "failure_reason": None}
            except Exception as exc:
                return self._unavailable_keyframe_score(f"qwen inference failed: {exc}")
        return self._unavailable_keyframe_score(self.base.load_error or "qwen unavailable")

    def recover_clip_data(self, image_paths: list[str], chart_context: dict[str, Any], frame_context: list[dict[str, Any]]) -> dict[str, Any]:
        chart_type = str(chart_context.get("chart_type", "unknown") or "unknown")
        if self.base.load():
            raw = None
            try:
                prompt = CLIP_DATA_PROMPT.replace(
                    "__CHART_CONTEXT__", json.dumps(chart_context, ensure_ascii=False, indent=2)
                ).replace("__FRAME_CONTEXT__", json.dumps(frame_context, ensure_ascii=False, indent=2))
                max_tokens = int(self.cfg.get("clip_data", {}).get("max_new_tokens", 4096))
                raw = self.base._generate(image_paths, prompt, max_new_tokens=max_tokens)
                data = _normalize_chart_data(_json_from_text(raw), chart_type)
                return {"data": data, "raw_response": raw, "model_status": "qwen", "failure_reason": None}
            except Exception as exc:
                return self._unknown_data(chart_type, f"qwen inference failed: {exc}", raw_response=raw)
        return self._unknown_data(chart_type, self.base.load_error or "qwen unavailable")

    def recover_chart_data(self, image_path: str, chart_context: dict[str, Any]) -> dict[str, Any]:
        frame_context = [{"image_index": 1, "source_frame": "keyframe", "time_seconds": None}]
        return self.recover_clip_data([image_path], chart_context, frame_context)

    def describe_animation(
        self,
        image_paths: list[str],
        clip_context: dict[str, Any],
        frame_context: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.base.load():
            raw = None
            try:
                prompt = ANIMATION_PROMPT.replace(
                    "__CLIP_CONTEXT__", json.dumps(clip_context, ensure_ascii=False, indent=2)
                ).replace("__FRAME_CONTEXT__", json.dumps(frame_context, ensure_ascii=False, indent=2))
                raw = self.base._generate(image_paths, prompt, max_new_tokens=512)
                result = _normalize_animation(_json_from_text(raw))
                return {"result": result, "raw_response": raw, "model_status": "qwen", "failure_reason": None}
            except Exception as exc:
                return self._unknown_animation(f"qwen inference failed: {exc}", raw_response=raw)
        return self._unknown_animation(self.base.load_error or "qwen unavailable")

    def _unavailable_keyframe_score(self, reason: str) -> dict[str, Any]:
        result = {
            "same_chart": False,
            "scene_change": True,
            "complete_chart": False,
            "data_marks_readable": False,
            "labels_readable": False,
            "legend_or_axes_readable": False,
            "staticness": 0.0,
            "completeness": 0.0,
            "chart_identity_consistency": 0.0,
            "data_extraction_suitability": 0.0,
            "motion_score": 1.0,
            "reason": f"Qwen multichart keyframe scoring unavailable: {reason}",
        }
        return {"result": result, "raw_response": None, "model_status": "qwen_unavailable", "failure_reason": reason}

    def _unknown_data(self, chart_type: str, reason: str, raw_response: str | None = None) -> dict[str, Any]:
        data = {
            "has_extractable_data": False,
            "needs_manual_data": False,
            "chart_type": chart_type,
            "title": None,
            "unit": None,
            "x_axis": None,
            "y_axis": None,
            "series": [],
            "rows": [],
            "manual_stub_rows": [],
            "visible_text": [],
            "uncertain_fields": [],
            "skip_reason": f"Could not reliably recover values: {reason}",
            "notes": "",
        }
        return {"data": data, "raw_response": raw_response, "model_status": "unavailable", "failure_reason": reason}

    def _unknown_animation(self, reason: str, raw_response: str | None = None) -> dict[str, Any]:
        result = {
            "is_target_chart_related": False,
            "overall_description": "无法可靠判断该片段中与目标图表相关的动画。",
            "major_actions": [],
            "confidence": 0.0,
        }
        return {"result": result, "raw_response": raw_response, "model_status": "unavailable", "failure_reason": reason}
