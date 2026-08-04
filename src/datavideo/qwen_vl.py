from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


SEMANTIC_COMPONENT_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "chart_type": {
            "type": "STRING",
            "enum": ["horizontal_bar", "vertical_bar", "line", "mixed", "unknown"],
        },
        "needs_review": {"type": "BOOLEAN"},
        "objects": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "STRING"},
                    "type": {
                        "type": "STRING",
                        "enum": [
                            "bar",
                            "icon",
                            "category_label",
                            "value_label",
                            "series",
                            "axis",
                            "annotation",
                            "title",
                        ],
                    },
                    "entity_id": {"type": "STRING", "nullable": True},
                    "label": {"type": "STRING"},
                    "text": {"type": "STRING", "nullable": True},
                    "text_status": {
                        "type": "STRING",
                        "enum": ["readable", "unreadable", "not_applicable"],
                    },
                    "bbox_px": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                    "dominant_color": {"type": "STRING", "nullable": True},
                    "confidence": {"type": "NUMBER"},
                    "reason": {"type": "STRING"},
                    "animation_axis": {"type": "STRING", "nullable": True},
                    "anchor": {"type": "STRING", "nullable": True},
                },
                "required": ["id", "type", "label", "text_status", "bbox_px", "confidence", "reason"],
            },
        },
        "entity_groups": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "entity_id": {"type": "STRING"},
                    "label": {"type": "STRING"},
                    "component_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["entity_id", "label", "component_ids", "confidence"],
            },
        },
        "warnings": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["chart_type", "needs_review", "objects", "entity_groups", "warnings"],
}

SEMANTIC_COMPONENT_PROMPT = """Generate semantic_components.json for this single data-video keyframe.
Return only JSON that matches the provided schema.

Semantic goal:
- Identify entities and their roles.
- Give rough bounding boxes for visible semantic parts.
- Use readable text when visible; otherwise set text to null and text_status to "unreadable".
- Group parts that belong to the same entity.
- Do not output final component ids. The program will compile stable ids later.

Return strict JSON only.
"""

QUALITY_REVIEW_PROMPT_PREFIX = """Review these generated dataset artifacts as an independent quality-control model.
Return only strict JSON:
{
  "needs_review": boolean,
  "severity": "low|medium|high",
  "issue_codes": [],
  "evidence": [],
  "recommended_action": "pass|manual_review|rerun"
}
Do not silently approve if visual evidence, recovered data, semantic roles, or state transitions look inconsistent.
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


def _normalize_bbox(bbox: Any) -> list[int] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        return [int(round(float(value))) for value in bbox]
    except Exception:
        return None


def _normalize_semantic_components(result: dict[str, Any]) -> dict[str, Any]:
    allowed_roles = {"bar", "icon", "category_label", "value_label", "series", "axis", "annotation", "title"}
    chart_type = str(result.get("chart_type", "unknown") or "unknown")
    entities: list[dict[str, Any]] = []
    for index, item in enumerate(result.get("entities", []), start=1):
        if not isinstance(item, dict):
            continue
        entity_key = str(item.get("entity_key") or item.get("label") or f"entity-{index}").strip()
        label = str(item.get("label") or entity_key).strip()
        components: list[dict[str, Any]] = []
        for component in item.get("components", []):
            if not isinstance(component, dict):
                continue
            role = str(component.get("role") or "").strip()
            if role not in allowed_roles:
                continue
            bbox = _normalize_bbox(component.get("bbox_px"))
            if bbox is None:
                continue
            text = component.get("text")
            text_status = str(component.get("text_status") or ("readable" if text else "unreadable"))
            components.append(
                {
                    "role": role,
                    "bbox_px": bbox,
                    "label": str(component.get("label") or label or role).strip(),
                    "text": None if text is None else str(text),
                    "text_status": text_status,
                    "confidence": _clamp01(component.get("confidence", 0.0)),
                    "reason": str(component.get("reason", "") or ""),
                    "animation_axis": str(component.get("animation_axis") or "") or None,
                    "anchor": str(component.get("anchor") or "") or None,
                }
            )
        if components:
            entities.append(
                {
                    "entity_key": entity_key,
                    "label": label,
                    "confidence": _clamp01(item.get("confidence", 0.0)),
                    "components": components,
                }
            )

    standalone_components: list[dict[str, Any]] = []
    for component in result.get("standalone_components", []):
        if not isinstance(component, dict):
            continue
        role = str(component.get("role") or "").strip()
        if role not in allowed_roles:
            continue
        bbox = _normalize_bbox(component.get("bbox_px"))
        if bbox is None:
            continue
        text = component.get("text")
        text_status = str(component.get("text_status") or ("readable" if text else "unreadable"))
        standalone_components.append(
            {
                "role": role,
                "bbox_px": bbox,
                "label": str(component.get("label") or role).strip(),
                "text": None if text is None else str(text),
                "text_status": text_status,
                "confidence": _clamp01(component.get("confidence", 0.0)),
                "reason": str(component.get("reason", "") or ""),
                "animation_axis": str(component.get("animation_axis") or "") or None,
                "anchor": str(component.get("anchor") or "") or None,
            }
        )

    return {
        "chart_type": chart_type,
        "entities": entities,
        "standalone_components": standalone_components,
        "warnings": [str(value) for value in result.get("warnings", []) if value],
    }


def _semantic_layout_quality_error(components: dict[str, Any]) -> str | None:
    entities = components.get("entities") if isinstance(components.get("entities"), list) else []
    standalone = components.get("standalone_components") if isinstance(components.get("standalone_components"), list) else []
    all_components = [component for entity in entities for component in entity.get("components", []) if isinstance(component, dict)]
    all_components.extend(component for component in standalone if isinstance(component, dict))
    if not all_components:
        return "semantic layout contains no components"
    degenerate = [component for component in all_components if component["bbox_px"][2] <= component["bbox_px"][0] or component["bbox_px"][3] <= component["bbox_px"][1]]
    if len(degenerate) == len(all_components):
        return "semantic layout contains only degenerate bboxes"
    return None


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

    def identify_semantic_components(
        self,
        image_path: str,
        chart_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.load():
            raw: str | None = None
            try:
                prompt = (
                    SEMANTIC_COMPONENT_PROMPT
                    + "\nChart metadata, if available:\n"
                    + json.dumps(chart_metadata or {}, ensure_ascii=False, indent=2)
                )
                raw = self._generate([image_path], prompt, max_new_tokens=1024)
                result = _normalize_semantic_components(_json_from_text(raw))
                quality_error = _semantic_layout_quality_error(result)
                if quality_error:
                    raise ValueError(quality_error)
                return {"result": result, "raw_response": raw, "model_status": "qwen", "failure_reason": None}
            except Exception as exc:
                return self._unavailable_semantic_components(f"qwen inference failed: {exc}", raw_response=raw)
        return self._unavailable_semantic_components(self.load_error or "qwen unavailable")

    def review_quality(self, image_paths: list[str], prompt: str) -> dict[str, Any]:
        if self.load():
            raw: str | None = None
            try:
                raw = self._generate(image_paths, QUALITY_REVIEW_PROMPT_PREFIX + "\n\n" + prompt, max_new_tokens=768)
                result = _json_from_text(raw)
                return {
                    "result": {
                        "needs_review": bool(result.get("needs_review", False)),
                        "severity": str(result.get("severity") or "medium"),
                        "issue_codes": result.get("issue_codes") if isinstance(result.get("issue_codes"), list) else [],
                        "evidence": result.get("evidence") if isinstance(result.get("evidence"), list) else [],
                        "recommended_action": str(result.get("recommended_action") or "manual_review"),
                    },
                    "raw_response": raw,
                    "model_status": "qwen",
                    "failure_reason": None,
                }
            except Exception as exc:
                return {
                    "result": {
                        "needs_review": True,
                        "severity": "medium",
                        "issue_codes": ["qc_vlm_failed"],
                        "evidence": [str(exc)],
                        "recommended_action": "manual_review",
                    },
                    "raw_response": raw,
                    "model_status": "qwen_unavailable",
                    "failure_reason": f"qwen quality review failed: {exc}",
                }
        return {
            "result": {
                "needs_review": True,
                "severity": "medium",
                "issue_codes": ["qc_vlm_unavailable"],
                "evidence": [self.load_error or "qwen unavailable"],
                "recommended_action": "manual_review",
            },
            "raw_response": None,
            "model_status": "qwen_unavailable",
            "failure_reason": self.load_error or "qwen unavailable",
        }

    def _unavailable_semantic_components(self, reason: str, raw_response: str | None = None) -> dict[str, Any]:
        result = {
            "chart_type": "unknown",
            "entities": [],
            "standalone_components": [],
            "warnings": [f"Qwen semantic component identification unavailable: {reason}"],
        }
        return {
            "result": result,
            "raw_response": raw_response,
            "model_status": "qwen_unavailable",
            "failure_reason": reason,
        }
