from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
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
                    "bbox_px": {
                        "type": "ARRAY",
                        "items": {"type": "INTEGER"},
                    },
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
                    "component_ids": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["entity_id", "label", "component_ids", "confidence"],
            },
        },
        "warnings": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
    },
    "required": ["chart_type", "needs_review", "objects", "entity_groups", "warnings"],
}


SEMANTIC_COMPONENT_PROMPT = """Generate semantic_components.json for this single data-video keyframe.

Return only JSON that matches the provided schema.

Semantic goal:
- Identify the semantic roles visible in the chart/video frame.
- Split visual entities such as car, plane, spaceship, chart series, bars, labels, icons, axes, title, annotation.
- Group each entity's icon + label + data mark when they belong together.
- Do not recover or invent real data values from geometry.
- Preserve visual position and draw order using bbox_px and object order.
- Use stable lowercase kebab-case IDs, e.g. car-icon, car-label, car-bar.
- Use entity_id without the "entity-" prefix, e.g. car, boing-747, spaceship.

Bounding boxes:
- bbox_px is [left_px, top_px, right_px, bottom_px] in image pixel coordinates.
- Use the actual visual extent of the component.
- right_px must be greater than left_px; bottom_px must be greater than top_px.
- Do not use placeholder boxes.

Roles:
- icon: pictorial object or entity illustration.
- category_label: entity/category text label.
- value_label: printed numeric or value text only.
- bar: data-encoding rectangle, not a label background.
- series: one line/curve/path in a line chart.
- axis: visible axis, baseline, tick line, or scale line.
- title: chart heading.
- annotation: other meaningful semantic element.

Animation fields:
- For bars, set animation_axis to "x" or "y" and anchor to "left", "right", "top", or "bottom".
- For line series, set animation_axis to "path" and anchor to "path-start".
- For non-animated components, use null for animation_axis and anchor.

Text:
- If text is visible and readable, fill text and text_status="readable".
- If the component is a label but text is not reliably readable, set text=null and text_status="unreadable".
- For icons/bars/series/axes without printed text, use text=null and text_status="not_applicable".

Quality:
- If any major component is uncertain or missing, keep the best objects you can and set needs_review=true with warnings.
- Do not fail the whole response just because one label is unreadable.
"""


def _json_from_text(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


class GeminiFlashClient:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        model_cfg = cfg.get("model", {})
        self.model_name = model_cfg.get("model_name", "gemini-2.5-flash")
        self.api_key_env = model_cfg.get("api_key_env", model_cfg.get("env_var", "GEMINI_API_KEY"))
        self.endpoint_base = model_cfg.get("endpoint_base", "https://generativelanguage.googleapis.com/v1beta")
        self.timeout = int(model_cfg.get("timeout_seconds", 90))
        self.temperature = float(model_cfg.get("temperature", 0.0))

    def identify_semantic_components(
        self,
        image_path: str,
        chart_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        api_key = os.environ.get(self.api_key_env) or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return self._unavailable(f"missing API key env var: {self.api_key_env} or GOOGLE_API_KEY")

        image = Path(image_path)
        prompt = (
            SEMANTIC_COMPONENT_PROMPT
            + "\nChart metadata, if available:\n"
            + json.dumps(chart_metadata or {}, ensure_ascii=False, indent=2)
        )
        payload = self._payload(prompt, image)
        raw_response: str | None = None
        try:
            response_json = self._post(payload, api_key)
            raw_response = json.dumps(response_json, ensure_ascii=False)
            text = self._extract_text(response_json)
            result = _json_from_text(text)
            return {
                "result": result,
                "raw_response": raw_response,
                "model_status": "gemini",
                "failure_reason": None,
            }
        except Exception as first_exc:
            fallback_payload = self._payload(prompt, image, use_schema=False)
            try:
                response_json = self._post(fallback_payload, api_key)
                raw_response = json.dumps(response_json, ensure_ascii=False)
                text = self._extract_text(response_json)
                result = _json_from_text(text)
                return {
                    "result": result,
                    "raw_response": raw_response,
                    "model_status": "gemini",
                    "failure_reason": None,
                }
            except Exception as second_exc:
                return self._unavailable(
                    f"gemini inference failed: {first_exc}; fallback failed: {second_exc}",
                    raw_response=raw_response,
                )

    def _payload(self, prompt: str, image_path: Path, *, use_schema: bool = True) -> dict[str, Any]:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        generation_config: dict[str, Any] = {
            "temperature": self.temperature,
            "responseMimeType": "application/json",
        }
        if use_schema:
            generation_config["responseSchema"] = SEMANTIC_COMPONENT_SCHEMA
        return {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": encoded,
                            }
                        },
                    ],
                }
            ],
            "generationConfig": generation_config,
        }

    def _post(self, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        url = f"{self.endpoint_base}/models/{self.model_name}:generateContent?key={api_key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API HTTP {exc.code}: {body}") from exc

    def _extract_text(self, response: dict[str, Any]) -> str:
        candidates = response.get("candidates") or []
        if not candidates:
            raise RuntimeError("Gemini response contained no candidates")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
        text = "\n".join(value for value in texts if value)
        if not text:
            raise RuntimeError("Gemini response contained no text")
        return text

    def _unavailable(self, reason: str, *, raw_response: str | None = None) -> dict[str, Any]:
        return {
            "result": {
                "objects": [],
                "entity_groups": [],
                "warnings": [reason],
            },
            "raw_response": raw_response,
            "model_status": "gemini_unavailable",
            "failure_reason": reason,
        }
