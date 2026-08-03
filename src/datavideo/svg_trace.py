from __future__ import annotations

from pathlib import Path
from typing import Any

from .semantic import build_semantic_svg


def trace_svg(
    image_path: str | Path,
    out_dir: str | Path,
    cfg: dict[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    """Backward-compatible wrapper for the renamed semantic SVG builder."""
    return build_semantic_svg(image_path, out_dir, cfg, force=force)
