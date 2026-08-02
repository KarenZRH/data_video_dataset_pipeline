from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCENE_STATES = {
    "non_chart",
    "chart_entering",
    "stable_chart",
    "chart_animating",
    "chart_leaving",
    "transition",
    "uncertain",
}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fieldnames = sorted({key for row in rows for key in row.keys()}) or ["field", "value"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def chart_result(
    *,
    is_chart: bool,
    confidence: float,
    reason: str,
    chart_type: str = "bar",
    scene_state: str = "stable_chart",
    chart_visible: bool | None = None,
    chart_completeness: float | None = None,
    occlusion: bool = False,
) -> dict[str, Any]:
    if scene_state not in SCENE_STATES:
        scene_state = "uncertain"
    visible = is_chart if chart_visible is None else chart_visible
    completeness = confidence if chart_completeness is None else chart_completeness
    return {
        "is_chart": bool(is_chart),
        "chart_types": [chart_type] if is_chart else [],
        "chart_visible": bool(visible),
        "chart_completeness": float(max(0.0, min(1.0, completeness))),
        "occlusion": bool(occlusion),
        "scene_state": scene_state,
        "confidence": float(max(0.0, min(1.0, confidence))),
        "reason": reason,
    }
