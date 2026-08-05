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
- For bar charts, include one type "bar" object for each visible bar mark. Do not represent bars only through labels or value labels.

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
    def _candidate_payloads(value: str) -> list[str]:
        payloads: list[str] = []
        text = value.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
            text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            payloads.append(text[start : end + 1])
        if start >= 0:
            payloads.append(text[start:])
        payloads.append(text)
        seen: set[str] = set()
        ordered: list[str] = []
        for payload in payloads:
            payload = payload.strip()
            if payload and payload not in seen:
                ordered.append(payload)
                seen.add(payload)
        return ordered

    def _repair_payload(payload: str) -> str:
        repaired = re.sub(r",\s*([}\]])", r"\1", payload.strip())
        repaired = re.sub(r"}\s*\n\s*{", "},\n{", repaired)
        repaired = re.sub(r"]\s*\n\s*\"", "],\n\"", repaired)
        repaired = re.sub(r"}\s*\n\s*\"", "},\n\"", repaired)
        stack: list[str] = []
        in_string = False
        escape = False
        for char in repaired:
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif char in ("}", "]") and stack and stack[-1] == char:
                stack.pop()
        if stack:
            repaired += "".join(reversed(stack))
        return repaired

    last_error: Exception | None = None
    for payload in _candidate_payloads(text):
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            last_error = exc
            try:
                return json.loads(_repair_payload(payload))
            except json.JSONDecodeError as repair_exc:
                last_error = repair_exc
                continue
    if last_error:
        raise last_error
    return json.loads(text)


def _dtype():
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_config_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _torch_dtype_from_config(preference: str | None):
    import torch

    dtype = (preference or "auto").strip().lower()
    if dtype in {"auto", ""}:
        return _dtype()
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp16", "float16", "half"}:
        return torch.float16
    if dtype in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported Qwen dtype_preference: {preference}")


def _merge_model_variant(model_cfg: dict[str, Any]) -> dict[str, Any]:
    merged = dict(model_cfg)
    if model_cfg.get("variant"):
        merged["selected_variant"] = str(model_cfg["variant"])
    variant_env = str(model_cfg.get("variant_env") or "QWEN_MODEL_VARIANT")
    variant = os.environ.get(variant_env)
    variants = model_cfg.get("variants") if isinstance(model_cfg.get("variants"), dict) else {}
    if not variant:
        return merged
    variant_cfg = variants.get(variant)
    if not isinstance(variant_cfg, dict):
        return merged
    merged = {**merged, **variant_cfg}
    merged["selected_variant"] = variant
    return merged


def _resolve_model_path(model_cfg: dict[str, Any]) -> tuple[str | None, str | None]:
    direct_path = model_cfg.get("path")
    if direct_path:
        return str(Path(str(direct_path)).expanduser()), "path"
    env_names: list[str] = []
    if isinstance(model_cfg.get("env_vars"), list):
        env_names.extend(str(name) for name in model_cfg["env_vars"] if name)
    if model_cfg.get("env_var"):
        env_names.append(str(model_cfg["env_var"]))
    for env_name in dict.fromkeys(env_names):
        value = os.environ.get(env_name)
        if value:
            return value, env_name
    return None, env_names[0] if env_names else None


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
    _shared: dict[str, Any] | None = None

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.model_cfg = _merge_model_variant(cfg["model"])
        self.model_path, self.model_path_source = _resolve_model_path(self.model_cfg)
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
        shared = QwenVLClient._shared
        if shared and self.model_path and Path(self.model_path).resolve() == Path(shared["model_path"]).resolve():
            self.model = shared["model"]
            self.processor = shared["processor"]
            self.model_version = shared["model_version"]
            return True
        if not self.available():
            if not self.load_error:
                source = self.model_path_source or "model.path"
                self.load_error = f"{source} is empty or missing"
            return False
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            torch_dtype = _torch_dtype_from_config(self.model_cfg.get("dtype_preference"))
            load_kwargs: dict[str, Any] = {
                "device_map": self.model_cfg.get("device_map", "auto"),
                "local_files_only": _as_config_bool(self.model_cfg.get("local_files_only"), True),
            }
            load_in_4bit = _as_config_bool(
                self.model_cfg.get("quantize_4bit", self.model_cfg.get("load_in_4bit")),
                False,
            )
            load_in_4bit_env = self.model_cfg.get("load_in_4bit_env")
            if load_in_4bit_env and os.environ.get(str(load_in_4bit_env)) is not None:
                load_in_4bit = _as_config_bool(os.environ.get(str(load_in_4bit_env)), load_in_4bit)
            if os.environ.get("DATAVIDEO_QUANTIZE_4BIT") is not None:
                load_in_4bit = _as_config_bool(os.environ.get("DATAVIDEO_QUANTIZE_4BIT"), load_in_4bit)
            if load_in_4bit:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_quant_type=str(self.model_cfg.get("bnb_4bit_quant_type", "nf4")),
                    bnb_4bit_use_double_quant=_as_config_bool(self.model_cfg.get("bnb_4bit_use_double_quant"), True),
                )
            else:
                load_kwargs["torch_dtype"] = torch_dtype

            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                local_files_only=_as_config_bool(self.model_cfg.get("local_files_only"), True),
                min_pixels=_as_int(self.model_cfg.get("min_pixels"), 224 * 224),
                max_pixels=_as_int(self.model_cfg.get("max_pixels"), 768 * 768),
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                **load_kwargs,
            )
            variant = self.model_cfg.get("selected_variant")
            self.model_version = f"{variant}:{Path(self.model_path).name}" if variant else Path(self.model_path).name
            QwenVLClient._shared = {
                "model": self.model,
                "processor": self.processor,
                "model_version": self.model_version,
                "model_path": self.model_path,
            }
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
                    + "\nOutput JSON schema:\n"
                    + json.dumps(SEMANTIC_COMPONENT_SCHEMA, ensure_ascii=False, indent=2)
                    + "\nChart metadata, if available:\n"
                    + json.dumps(chart_metadata or {}, ensure_ascii=False, indent=2)
                )
                raw = self._generate(
                    [image_path],
                    prompt,
                    max_new_tokens=_as_int(self.model_cfg.get("semantic_max_new_tokens"), 2048),
                )
                result = _json_from_text(raw)
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
