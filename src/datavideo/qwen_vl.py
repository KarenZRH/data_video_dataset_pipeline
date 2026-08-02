from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .schemas import chart_result


CHART_PROMPT = """You are detecting whether the provided consecutive video frame(s) are part of a data-video clip that visually contains a bar chart.
Positive means: a bar chart, bar-race chart, histogram-like bar chart, or grouped/stacked bar chart is clearly visible in the frame(s), with enough chart area to later extract keyframes and recover chart data.
Negative means: no chart, only a talking head, only decorative graphics, only plain text, a map/table/diagram without bars, or a transition where the chart is not yet readable.
If a bar chart is partly entering/leaving or animating but still readable, return is_chart=true and use the matching scene_state.
If uncertain, prefer is_chart=false with scene_state="uncertain" and a lower confidence.
Return only strict JSON with this exact schema:
{"is_chart": true, "chart_types": ["bar"], "chart_visible": true, "chart_completeness": 0.0, "occlusion": false, "scene_state": "stable_chart", "confidence": 0.0, "reason": "一句话说明依据"}
scene_state must be one of: non_chart, chart_entering, stable_chart, chart_animating, chart_leaving, transition, uncertain.
"""

DATA_VIDEO_CLIP_PROMPT = """目标：
判断输入的 1-3 张连续视频帧是否可能属于 data-video clip candidate。目标是高召回，宁可多保留候选，也不要漏掉图表出现、增长、强调等动画过程。

核心定义：
本轮检测的是 data-video clip candidate，不只是稳定、完整、可读的静态图表。

Positive 包括：
- 柱状图、水平柱状图、bar-race、类似进度条但实际表达数据的条形 mark；
- 图表正在出现、进入、增长、缩短、强调、排序或离开；
- 即使坐标轴、标题或数值暂时不完整，只要明显是数据可视化动画的一部分，也应判为 candidate。

Negative 包括：
- 纯人物画面；
- 普通字幕；
- 装饰性图形；
- 不表达数据的进度条或 UI 元素；
- 普通地图、照片、插图；
- 完全不可判断的数据含义画面。

重要策略：
如果不确定但存在“数据 mark 动画”的证据，优先保留为 candidate，设置低 confidence，不要直接丢弃。

只返回严格 JSON，不要 markdown，不要解释，字段必须完整。必须根据输入画面填写真实判断，不要复制字段说明中的占位值。
字段格式：
{
  "is_data_video_clip_candidate": boolean,
  "contains_data_marks": boolean,
  "data_mark_types": string[],
  "chart_types": string[],
  "chart_readable": boolean,
  "chart_completeness": number,
  "scene_state": string,
  "animation_cue": string,
  "confidence": number,
  "reason": string
}

如果画面没有数据 mark，必须设置 is_data_video_clip_candidate=false, contains_data_marks=false, data_mark_types=[], chart_types=[], scene_state="non_chart", confidence<=0.2。
scene_state 只能使用：
non_chart, chart_entering, stable_chart, chart_animating, chart_leaving, transition, uncertain
"""

DATA_PROMPT = """Recover the data from the bar chart keyframe. Return only strict JSON.
Use null for values or labels that cannot be read reliably. Do not invent values.
Schema: {"title": null, "x_axis": null, "y_axis": null, "unit": null, "bars": [{"label": null, "value": null}], "uncertain_fields": [], "notes": ""}
"""

MERGED_CLIP_REVIEW_PROMPT = """You are reviewing a contact sheet from one merged video segment.

Question:
Does this merged segment represent a complete bar-type data-video clip with a coherent visual data message, rather than an isolated animation or decorative motion?

Keep only if there is data encoding evidence. Data encoding evidence means visible bars or bar-like marks whose length/position/ordering/label clearly encodes quantities, categories, ranks, distances, progress toward a measured target, or comparison. Decorative shapes, isolated circles, maps, photos, plain text, vehicles moving, UI progress with no data meaning, or one-off motion are not enough.

Return strict JSON only. Use this schema:
{
  "is_complete_data_video_clip": boolean,
  "has_data_encoding_evidence": boolean,
  "data_encoding_evidence": string[],
  "coherent_visual_data_message": boolean,
  "chart_types": string[],
  "mark_types": string[],
  "scene_continuity": string,
  "decision": string,
  "confidence": number,
  "reason": string
}

decision must be one of: keep, exclude, uncertain.
"""

BAR_DOMINANT_FRAME_PROMPT = """You are detecting whether 1-3 consecutive video frames belong to the target class: bar-chart-dominant clip.

Target:
Positive only when the main narrative unit is expressed by bar marks. Bar marks include vertical bars, horizontal bars, stacked/grouped bars, bar-race bars, or bar-like marks whose length/position/order encodes quantities or categories.

Negative:
- only a short or incidental bar-like shape;
- circle, bubble, distance line, map, icon, illustration, vehicle, photo, or decorative motion;
- UI/progress bars without clear data encoding;
- mixed scenes where the current frames are not mainly bar-mark driven.

If a video segment briefly has bars and then switches to other marks, only the bar-dominant subsegment should be positive.

Return strict JSON only:
{
  "is_bar_chart_dominant_candidate": boolean,
  "bar_marks_visible": boolean,
  "bar_marks_dominant": boolean,
  "has_data_encoding_evidence": boolean,
  "scene_state": string,
  "confidence": number,
  "reason": string
}

scene_state must be one of: non_chart, chart_entering, stable_chart, chart_animating, chart_leaving, transition, uncertain.
"""

BAR_IDENTITY_PROMPT = """Identify the chart identity for these nearby frames from one bar-chart candidate.

Return strict JSON only:
{
  "chart_identity": string,
  "chart_title": string,
  "axis_labels": string[],
  "category_labels": string[],
  "chart_types": string[],
  "mark_types": string[],
  "animation_cue": string,
  "confidence": number,
  "reason": string
}

Use short strings. If text is unreadable, use "" or [].
"""

BAR_DOMINANT_CLIP_REVIEW_PROMPT = """You are reviewing a contact sheet from one merged candidate segment.

Question:
Is this a complete bar-chart-dominant data-video clip: a coherent semantic process where the main visual data message is expressed by bar marks?

Keep only if:
- bar marks are the dominant visual encoding for the candidate, not incidental;
- bar length/position/order/labels encode data;
- the segment is not a mixed segment where bars appear briefly and then the main narrative switches to circle, bubble, distance line, map, icon, illustration, or decorative motion;
- the segment forms a coherent data narrative process.

Return strict JSON only:
{
  "is_complete_bar_dominant_clip": boolean,
  "bar_marks_dominant": boolean,
  "has_data_encoding_evidence": boolean,
  "coherent_visual_data_message": boolean,
  "mixed_with_non_bar_marks": boolean,
  "suggested_start": number,
  "suggested_end": number,
  "chart_types": string[],
  "mark_types": string[],
  "decision": string,
  "confidence": number,
  "reason": string
}

decision must be one of: keep, trim, exclude, uncertain.
If the segment is mixed but contains a clear bar-dominant subclip, use decision="trim" and set suggested_start/suggested_end relative to the original video seconds.
"""

KEYFRAME_SCORE_PROMPT = """You are scoring one video frame as a seed keyframe for a data-video clip.

The goal is NOT to choose the most visually stable frame overall.
The goal is to choose a good INITIAL seed keyframe: a frame before the main data-change animation, where the target chart is still complete and representative.

Target chart identity:
__CHART_IDENTITY__

A good seed keyframe must satisfy these conditions:
1. It belongs to the same target chart.
2. It is before the main data-change animation.
3. It shows the complete initial chart state.
4. All target categories, labels, bars/marks, axes, and title that define the chart are visible.
5. It is not a later stable state after some categories/bars have disappeared, changed, moved away, or been replaced.
6. It is not a scene cut, next chart, zoomed/cropped view, transition frame, or partial chart.

Important rule:
A later frame can be very stable and clear, but if it is after the main change, it is a bad seed keyframe.
A slightly less stable frame before the change is better than a perfectly stable frame after the change.

Reject or strongly penalize frames where:
- any target category is missing or cropped;
- some bars/marks have already changed to a new state;
- the chart is already in the result/final state rather than the initial state;
- the frame belongs to the next scene or a different chart;
- the chart identity has changed, including title, axis labels, category set, chart type, or main visual encoding.

Score this frame using the following fields.

Return strict JSON only:
{
  "same_chart": boolean,
  "scene_change": boolean,
  "pre_change": boolean,
  "post_change_state": boolean,
  "complete_initial_chart": boolean,
  "all_target_categories_visible": boolean,
  "completeness": number,
  "staticness": number,
  "chart_identity_consistency": number,
  "initial_state_representative": number,
  "motion_score": number,
  "reason": string
}
"""

def _json_from_text(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.S)
    payload = match.group(0) if match else text
    return json.loads(payload)


def _dtype():
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


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


def _normalize_data_video_result(result: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "is_data_video_clip_candidate": _as_bool(result.get("is_data_video_clip_candidate", False)),
        "contains_data_marks": _as_bool(result.get("contains_data_marks", False)),
        "data_mark_types": result.get("data_mark_types") if isinstance(result.get("data_mark_types"), list) else [],
        "chart_types": result.get("chart_types") if isinstance(result.get("chart_types"), list) else [],
        "chart_readable": _as_bool(result.get("chart_readable", False)),
        "chart_completeness": _clamp01(result.get("chart_completeness", 0.0)),
        "scene_state": result.get("scene_state", "uncertain"),
        "animation_cue": str(result.get("animation_cue", "unknown") or "unknown"),
        "confidence": _clamp01(result.get("confidence", 0.0)),
        "reason": str(result.get("reason", "") or ""),
    }
    if normalized["scene_state"] not in {
        "non_chart",
        "chart_entering",
        "stable_chart",
        "chart_animating",
        "chart_leaving",
        "transition",
        "uncertain",
    }:
        normalized["scene_state"] = "uncertain"

    reason = normalized["reason"].lower()
    negative_terms = [
        "没有明显的数据",
        "没有数据可视化",
        "没有明显的图表",
        "不符合数据可视化",
        "不属于数据可视化",
        "无数据",
        "非图表",
        "no data",
        "no chart",
        "not a data",
        "not data",
    ]
    if any(term in reason for term in negative_terms):
        normalized.update(
            {
                "is_data_video_clip_candidate": False,
                "contains_data_marks": False,
                "data_mark_types": [],
                "chart_types": [],
                "chart_readable": False,
                "chart_completeness": 0.0,
                "scene_state": "non_chart",
                "animation_cue": "none",
                "confidence": min(normalized["confidence"], 0.2),
                "sanitized": True,
                "sanitized_reason": "negative reason contradicted positive fields",
            }
        )
    return normalized


def _normalize_merged_clip_review(result: dict[str, Any]) -> dict[str, Any]:
    decision = str(result.get("decision", "uncertain") or "uncertain").lower()
    if decision not in {"keep", "exclude", "uncertain"}:
        decision = "uncertain"
    evidence = result.get("data_encoding_evidence")
    normalized = {
        "is_complete_data_video_clip": _as_bool(result.get("is_complete_data_video_clip", False)),
        "has_data_encoding_evidence": _as_bool(result.get("has_data_encoding_evidence", False)),
        "data_encoding_evidence": evidence if isinstance(evidence, list) else [],
        "coherent_visual_data_message": _as_bool(result.get("coherent_visual_data_message", False)),
        "chart_types": result.get("chart_types") if isinstance(result.get("chart_types"), list) else [],
        "mark_types": result.get("mark_types") if isinstance(result.get("mark_types"), list) else [],
        "scene_continuity": str(result.get("scene_continuity", "uncertain") or "uncertain"),
        "decision": decision,
        "confidence": _clamp01(result.get("confidence", 0.0)),
        "reason": str(result.get("reason", "") or ""),
    }
    reason = normalized["reason"].lower()
    negative_terms = ["decorative", "isolated", "no data", "not data", "没有数据", "装饰", "孤立", "无数据"]
    if not normalized["has_data_encoding_evidence"] or any(term in reason for term in negative_terms):
        normalized["decision"] = "exclude" if normalized["confidence"] >= 0.4 else "uncertain"
        normalized["is_complete_data_video_clip"] = False
    if normalized["decision"] == "keep" and not (
        normalized["has_data_encoding_evidence"] and normalized["coherent_visual_data_message"]
    ):
        normalized["decision"] = "uncertain"
    return normalized


def _normalize_bar_dominant_frame_result(result: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "is_bar_chart_dominant_candidate": _as_bool(result.get("is_bar_chart_dominant_candidate", False)),
        "bar_marks_visible": _as_bool(result.get("bar_marks_visible", False)),
        "bar_marks_dominant": _as_bool(result.get("bar_marks_dominant", False)),
        "has_data_encoding_evidence": _as_bool(result.get("has_data_encoding_evidence", False)),
        "chart_identity": "",
        "chart_title": "",
        "axis_labels": [],
        "category_labels": [],
        "chart_types": [],
        "mark_types": [],
        "scene_state": result.get("scene_state", "uncertain"),
        "animation_cue": str(result.get("animation_cue", "unknown") or "unknown"),
        "confidence": _clamp01(result.get("confidence", 0.0)),
        "reason": str(result.get("reason", "") or ""),
    }
    if normalized["scene_state"] not in {
        "non_chart",
        "chart_entering",
        "stable_chart",
        "chart_animating",
        "chart_leaving",
        "transition",
        "uncertain",
    }:
        normalized["scene_state"] = "uncertain"
    reason = normalized["reason"].lower()
    non_bar_terms = ["circle", "bubble", "map", "icon", "illustration", "vehicle", "distance line", "decorative", "not bar"]
    if any(term in reason for term in non_bar_terms) and not normalized["bar_marks_dominant"]:
        normalized["is_bar_chart_dominant_candidate"] = False
    if not (
        normalized["bar_marks_visible"]
        and normalized["bar_marks_dominant"]
        and normalized["has_data_encoding_evidence"]
    ):
        normalized["is_bar_chart_dominant_candidate"] = False
    return normalized


def _normalize_bar_identity_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "chart_identity": str(result.get("chart_identity", "") or ""),
        "chart_title": str(result.get("chart_title", "") or ""),
        "axis_labels": result.get("axis_labels") if isinstance(result.get("axis_labels"), list) else [],
        "category_labels": result.get("category_labels") if isinstance(result.get("category_labels"), list) else [],
        "chart_types": result.get("chart_types") if isinstance(result.get("chart_types"), list) else [],
        "mark_types": result.get("mark_types") if isinstance(result.get("mark_types"), list) else [],
        "animation_cue": str(result.get("animation_cue", "unknown") or "unknown"),
        "confidence": _clamp01(result.get("confidence", 0.0)),
        "reason": str(result.get("reason", "") or ""),
    }


def _normalize_bar_dominant_clip_review(result: dict[str, Any]) -> dict[str, Any]:
    decision = str(result.get("decision", "uncertain") or "uncertain").lower()
    if decision not in {"keep", "trim", "exclude", "uncertain"}:
        decision = "uncertain"
    normalized = {
        "is_complete_bar_dominant_clip": _as_bool(result.get("is_complete_bar_dominant_clip", False)),
        "bar_marks_dominant": _as_bool(result.get("bar_marks_dominant", False)),
        "has_data_encoding_evidence": _as_bool(result.get("has_data_encoding_evidence", False)),
        "coherent_visual_data_message": _as_bool(result.get("coherent_visual_data_message", False)),
        "mixed_with_non_bar_marks": _as_bool(result.get("mixed_with_non_bar_marks", False)),
        "suggested_start": result.get("suggested_start"),
        "suggested_end": result.get("suggested_end"),
        "chart_types": result.get("chart_types") if isinstance(result.get("chart_types"), list) else [],
        "mark_types": result.get("mark_types") if isinstance(result.get("mark_types"), list) else [],
        "decision": decision,
        "confidence": _clamp01(result.get("confidence", 0.0)),
        "reason": str(result.get("reason", "") or ""),
    }
    if not (
        normalized["bar_marks_dominant"]
        and normalized["has_data_encoding_evidence"]
        and normalized["coherent_visual_data_message"]
    ):
        normalized["decision"] = "exclude" if normalized["confidence"] >= 0.4 else "uncertain"
        normalized["is_complete_bar_dominant_clip"] = False
    if normalized["mixed_with_non_bar_marks"] and normalized["decision"] == "keep":
        normalized["decision"] = "trim"
    return normalized


def _normalize_keyframe_score(result: dict[str, Any]) -> dict[str, Any]:
    complete_initial_chart = _as_bool(
        result.get("complete_initial_chart", result.get("complete_static_chart_form", False))
    )
    all_target_categories_visible = _as_bool(
        result.get("all_target_categories_visible", complete_initial_chart)
    )
    return {
        "same_chart": _as_bool(result.get("same_chart", False)),
        "scene_change": _as_bool(result.get("scene_change", False)),
        "pre_change": _as_bool(result.get("pre_change", False)),
        "post_change_state": _as_bool(result.get("post_change_state", False)),
        "complete_initial_chart": complete_initial_chart,
        "all_target_categories_visible": all_target_categories_visible,
        "completeness": _clamp01(result.get("completeness", 0.0)),
        "staticness": _clamp01(result.get("staticness", 0.0)),
        "chart_identity_consistency": _clamp01(result.get("chart_identity_consistency", 0.0)),
        "initial_state_representative": _clamp01(result.get("initial_state_representative", 0.0)),
        "motion_score": _clamp01(result.get("motion_score", 1.0), default=1.0),
        "reason": str(result.get("reason", "") or ""),
    }


class QwenVLClient:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.model_path = os.environ.get(cfg["model"]["env_var"])
        self.model = None
        self.processor = None
        self.model_version = None
        self.load_error = None

    def available(self) -> bool:
        if os.environ.get("DATAVIDEO_SKIP_QWEN") == "1":
            self.load_error = "DATAVIDEO_SKIP_QWEN=1"
            return False
        return bool(self.model_path and Path(self.model_path).exists())

    def load(self) -> bool:
        if self.model is not None:
            return True
        if not self.available():
            if not self.load_error:
                self.load_error = f"{self.cfg['model']['env_var']} is empty or missing"
            return False
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                local_files_only=True,
                min_pixels=224 * 224,
                max_pixels=768 * 768,
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=_dtype(),
                device_map="auto",
                local_files_only=True,
            )
            self.model_version = Path(self.model_path).name
            return True
        except Exception as exc:
            self.load_error = str(exc)
            return False

    def _generate(self, image_paths: list[str], prompt: str, max_new_tokens: int = 256) -> str:
        from qwen_vl_utils import process_vision_info

        assert self.model is not None and self.processor is not None
        content = [{"type": "image", "image": str(path)} for path in image_paths]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    def classify_frames(self, image_paths: list[str]) -> dict[str, Any]:
        if self.load():
            try:
                raw = self._generate(image_paths, CHART_PROMPT, max_new_tokens=96)
                result = _json_from_text(raw)
                return {"result": result, "raw_response": raw, "model_status": "qwen", "failure_reason": None}
            except Exception as exc:
                return self._unavailable_chart(f"qwen inference failed: {exc}")
        return self._unavailable_chart(self.load_error or "qwen unavailable")

    def detect_data_video_clip_candidate(self, image_paths: list[str]) -> dict[str, Any]:
        if self.load():
            try:
                raw = self._generate(image_paths, DATA_VIDEO_CLIP_PROMPT, max_new_tokens=128)
                result = _normalize_data_video_result(_json_from_text(raw))
                return {"result": result, "raw_response": raw, "model_status": "qwen", "failure_reason": None}
            except Exception as exc:
                return self._unavailable_data_video_candidate(f"qwen inference failed: {exc}")
        return self._unavailable_data_video_candidate(self.load_error or "qwen unavailable")

    def review_merged_clip_contact_sheet(self, image_path: str) -> dict[str, Any]:
        if self.load():
            try:
                raw = self._generate([image_path], MERGED_CLIP_REVIEW_PROMPT, max_new_tokens=192)
                result = _normalize_merged_clip_review(_json_from_text(raw))
                return {"result": result, "raw_response": raw, "model_status": "qwen", "failure_reason": None}
            except Exception as exc:
                return self._unavailable_merged_review(f"qwen inference failed: {exc}")
        return self._unavailable_merged_review(self.load_error or "qwen unavailable")

    def detect_bar_dominant_frames(self, image_paths: list[str]) -> dict[str, Any]:
        if self.load():
            try:
                raw = self._generate(image_paths, BAR_DOMINANT_FRAME_PROMPT, max_new_tokens=160)
                result = _normalize_bar_dominant_frame_result(_json_from_text(raw))
                return {"result": result, "raw_response": raw, "model_status": "qwen", "failure_reason": None}
            except Exception as exc:
                return self._unavailable_bar_dominant_frame(f"qwen inference failed: {exc}")
        return self._unavailable_bar_dominant_frame(self.load_error or "qwen unavailable")

    def identify_bar_candidate_frames(self, image_paths: list[str]) -> dict[str, Any]:
        if self.load():
            try:
                raw = self._generate(image_paths, BAR_IDENTITY_PROMPT, max_new_tokens=192)
                result = _normalize_bar_identity_result(_json_from_text(raw))
                return {"result": result, "raw_response": raw, "model_status": "qwen", "failure_reason": None}
            except Exception as exc:
                return self._unavailable_bar_identity(f"qwen inference failed: {exc}")
        return self._unavailable_bar_identity(self.load_error or "qwen unavailable")

    def review_bar_dominant_clip_contact_sheet(self, image_path: str) -> dict[str, Any]:
        if self.load():
            try:
                raw = self._generate([image_path], BAR_DOMINANT_CLIP_REVIEW_PROMPT, max_new_tokens=224)
                result = _normalize_bar_dominant_clip_review(_json_from_text(raw))
                return {"result": result, "raw_response": raw, "model_status": "qwen", "failure_reason": None}
            except Exception as exc:
                return self._unavailable_bar_dominant_clip(f"qwen inference failed: {exc}")
        return self._unavailable_bar_dominant_clip(self.load_error or "qwen unavailable")

    def score_keyframe_candidate(self, image_path: str, chart_identity: str) -> dict[str, Any]:
        if self.load():
            try:
                prompt = KEYFRAME_SCORE_PROMPT.replace("__CHART_IDENTITY__", chart_identity)
                raw = self._generate([image_path], prompt, max_new_tokens=192)
                result = _normalize_keyframe_score(_json_from_text(raw))
                return {"result": result, "raw_response": raw, "model_status": "qwen", "failure_reason": None}
            except Exception as exc:
                return self._unavailable_keyframe_score(f"qwen inference failed: {exc}")
        return self._unavailable_keyframe_score(self.load_error or "qwen unavailable")

    def recover_chart_data(self, image_path: str) -> dict[str, Any]:
        if self.load():
            try:
                raw = self._generate([image_path], DATA_PROMPT, max_new_tokens=512)
                return {"data": _json_from_text(raw), "raw_response": raw, "model_status": "qwen", "failure_reason": None}
            except Exception as exc:
                return self._unknown_data(f"qwen inference failed: {exc}")
        return self._unknown_data(self.load_error or "qwen unavailable")

    def _unavailable_chart(self, reason: str) -> dict[str, Any]:
        result = chart_result(
            is_chart=False,
            chart_type=self.cfg["chart_type"],
            confidence=0.0,
            scene_state="uncertain",
            chart_visible=False,
            chart_completeness=0.0,
            reason=f"Qwen chart detection unavailable: {reason}",
        )
        return {"result": result, "raw_response": None, "model_status": "qwen_unavailable", "failure_reason": reason}

    def _unavailable_data_video_candidate(self, reason: str) -> dict[str, Any]:
        result = {
            "is_data_video_clip_candidate": False,
            "contains_data_marks": False,
            "data_mark_types": [],
            "chart_types": [],
            "chart_readable": False,
            "chart_completeness": 0.0,
            "scene_state": "uncertain",
            "animation_cue": "unknown",
            "confidence": 0.0,
            "reason": f"Qwen data-video detection unavailable: {reason}",
        }
        return {"result": result, "raw_response": None, "model_status": "qwen_unavailable", "failure_reason": reason}

    def _unavailable_merged_review(self, reason: str) -> dict[str, Any]:
        result = {
            "is_complete_data_video_clip": False,
            "has_data_encoding_evidence": False,
            "data_encoding_evidence": [],
            "coherent_visual_data_message": False,
            "chart_types": [],
            "mark_types": [],
            "scene_continuity": "uncertain",
            "decision": "uncertain",
            "confidence": 0.0,
            "reason": f"Qwen merged clip review unavailable: {reason}",
        }
        return {"result": result, "raw_response": None, "model_status": "qwen_unavailable", "failure_reason": reason}

    def _unavailable_bar_dominant_frame(self, reason: str) -> dict[str, Any]:
        result = {
            "is_bar_chart_dominant_candidate": False,
            "bar_marks_visible": False,
            "bar_marks_dominant": False,
            "has_data_encoding_evidence": False,
            "chart_identity": "",
            "chart_title": "",
            "axis_labels": [],
            "category_labels": [],
            "chart_types": [],
            "mark_types": [],
            "scene_state": "uncertain",
            "animation_cue": "unknown",
            "confidence": 0.0,
            "reason": f"Qwen bar-dominant frame detection unavailable: {reason}",
        }
        return {"result": result, "raw_response": None, "model_status": "qwen_unavailable", "failure_reason": reason}

    def _unavailable_bar_dominant_clip(self, reason: str) -> dict[str, Any]:
        result = {
            "is_complete_bar_dominant_clip": False,
            "bar_marks_dominant": False,
            "has_data_encoding_evidence": False,
            "coherent_visual_data_message": False,
            "mixed_with_non_bar_marks": False,
            "suggested_start": None,
            "suggested_end": None,
            "chart_types": [],
            "mark_types": [],
            "decision": "uncertain",
            "confidence": 0.0,
            "reason": f"Qwen bar-dominant clip review unavailable: {reason}",
        }
        return {"result": result, "raw_response": None, "model_status": "qwen_unavailable", "failure_reason": reason}

    def _unavailable_bar_identity(self, reason: str) -> dict[str, Any]:
        result = {
            "chart_identity": "",
            "chart_title": "",
            "axis_labels": [],
            "category_labels": [],
            "chart_types": [],
            "mark_types": [],
            "animation_cue": "unknown",
            "confidence": 0.0,
            "reason": f"Qwen bar identity detection unavailable: {reason}",
        }
        return {"result": result, "raw_response": None, "model_status": "qwen_unavailable", "failure_reason": reason}

    def _unavailable_keyframe_score(self, reason: str) -> dict[str, Any]:
        result = {
            "same_chart": False,
            "scene_change": True,
            "pre_change": False,
            "post_change_state": False,
            "complete_initial_chart": False,
            "all_target_categories_visible": False,
            "completeness": 0.0,
            "staticness": 0.0,
            "chart_identity_consistency": 0.0,
            "initial_state_representative": 0.0,
            "motion_score": 1.0,
            "reason": f"Qwen keyframe scoring unavailable: {reason}",
        }
        return {"result": result, "raw_response": None, "model_status": "qwen_unavailable", "failure_reason": reason}

    def _unknown_data(self, reason: str) -> dict[str, Any]:
        data = {
            "title": None,
            "x_axis": None,
            "y_axis": None,
            "unit": None,
            "bars": [{"label": None, "value": None}],
            "uncertain_fields": ["title", "x_axis", "y_axis", "unit", "bars"],
            "notes": f"Could not reliably recover values: {reason}",
        }
        return {"data": data, "raw_response": None, "model_status": "unavailable", "failure_reason": reason}
