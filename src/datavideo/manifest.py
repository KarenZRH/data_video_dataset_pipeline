from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

from .schemas import ensure_dir, file_sha256, object_hash, write_json


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["config_path"] = str(path)
    cfg["config_hash"] = object_hash(cfg)
    return cfg


def command_version(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, check=False, capture_output=True, text=True)
        return (out.stdout or out.stderr).splitlines()[0]
    except Exception as exc:
        return f"unavailable: {exc}"


def write_video_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    video_path = Path(cfg["video_path"])
    row = {
        "sample_id": cfg["sample_id"],
        "chart_type": cfg["chart_type"],
        "source_video": str(video_path),
        "source_exists": video_path.exists(),
        "source_sha256": file_sha256(video_path) if video_path.exists() else None,
        "model_path": os.environ.get(cfg["model"]["env_var"]),
        "prompt_version": cfg["model"]["prompt_version"],
        "config_hash": cfg["config_hash"],
        "ffmpeg_version": command_version(["ffmpeg", "-version"]),
    }
    out = ensure_dir(cfg["processed_dir"]) / "video_manifest.json"
    write_json(out, row)
    return row
