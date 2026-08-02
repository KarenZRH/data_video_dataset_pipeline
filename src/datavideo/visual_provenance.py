from __future__ import annotations

from pathlib import Path


def assert_qwen_visual_inputs(image_paths: list[str | Path]) -> None:
    for image_path in image_paths:
        parts = set(Path(image_path).parts)
        if "context" in parts or "chart_boundary" in parts:
            raise RuntimeError(f"Qwen input must come from visual_clip-derived frames, got {image_path}")
        if "visual_frames" not in parts and "keyframes" not in parts:
            raise RuntimeError(f"Qwen input is missing visual frame provenance, got {image_path}")
