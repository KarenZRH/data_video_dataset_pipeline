from __future__ import annotations

from pathlib import Path
from typing import Any

from .model_client import make_qwen_client
from .schemas import ensure_dir, write_csv, write_json


def recover_chart_data(
    cfg: dict[str, Any],
    image_path: str | Path,
    out_dir: str | Path,
    *,
    client: Any | None = None,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    client = client or make_qwen_client(cfg)
    response = client.recover_chart_data(str(image_path))
    data = response["data"]
    write_json(out_dir / "chart_data_raw.json", {"response": response, "model_path": client.model_path, "prompt_version": cfg["model"]["prompt_version"]})
    bars = data.get("bars") if isinstance(data, dict) else []
    rows = []
    for idx, bar in enumerate(bars or []):
        rows.append({"index": idx, "label": bar.get("label"), "value": bar.get("value")})
    if not rows:
        rows = [{"index": 0, "label": None, "value": None}]
    write_csv(out_dir / "chart_data.csv", rows)
    uncertain_fields = list(data.get("uncertain_fields", []))
    for row in rows:
        if row.get("label") is None:
            uncertain_fields.append(f"bars[{row['index']}].label")
        if row.get("value") is None:
            uncertain_fields.append(f"bars[{row['index']}].value")
    metadata = {
        "title": data.get("title"),
        "x_axis": data.get("x_axis"),
        "y_axis": data.get("y_axis"),
        "unit": data.get("unit"),
        "model_status": response["model_status"],
        "failure_reason": response["failure_reason"],
    }
    write_json(out_dir / "chart_metadata.json", metadata)
    validation = {
        "valid_schema": isinstance(data, dict) and isinstance(rows, list),
        "uncertain_fields": sorted(set(uncertain_fields)),
        "value_count": len(rows),
        "do_not_use_for_training_without_review": bool(uncertain_fields),
    }
    write_json(out_dir / "chart_data_validation.json", validation)
    return {"data": data, "metadata": metadata, "validation": validation}
