from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

from .schemas import ensure_dir, file_sha256, object_hash, write_json


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_file(path: Path, seen: set[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Recursive config extends detected at {resolved}")
    seen.add(resolved)
    with resolved.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    extends = cfg.pop("extends", None)
    base_cfg: dict[str, Any] = {}
    if extends:
        extends_items = [extends] if isinstance(extends, (str, Path)) else list(extends)
        for base in extends_items:
            base_path = Path(base)
            if not base_path.is_absolute():
                base_path = resolved.parent / base_path
            base_cfg = _deep_merge(base_cfg, _load_config_file(base_path, seen))
    seen.remove(resolved)
    return _deep_merge(base_cfg, cfg)


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = _load_config_file(Path(path), set())
    cfg["config_path"] = str(path)
    cfg["config_hash"] = object_hash(cfg)
    return cfg


def _effective_model_cfg(model_cfg: dict[str, Any]) -> dict[str, Any]:
    merged = dict(model_cfg)
    if model_cfg.get("variant"):
        merged["selected_variant"] = str(model_cfg["variant"])
    variant_env = str(model_cfg.get("variant_env") or "QWEN_MODEL_VARIANT")
    variant = os.environ.get(variant_env)
    variants = model_cfg.get("variants") if isinstance(model_cfg.get("variants"), dict) else {}
    if variant and isinstance(variants.get(variant), dict):
        merged = {**merged, **variants[variant]}
        merged["selected_variant"] = variant
        return merged
    return merged


def _model_path_from_cfg(model_cfg: dict[str, Any]) -> str | None:
    effective = _effective_model_cfg(model_cfg)
    if effective.get("path"):
        return str(effective["path"])
    env_names: list[str] = []
    if isinstance(effective.get("env_vars"), list):
        env_names.extend(str(name) for name in effective["env_vars"] if name)
    if effective.get("env_var"):
        env_names.append(str(effective["env_var"]))
    for env_name in dict.fromkeys(env_names):
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def _model_variant_from_cfg(model_cfg: dict[str, Any]) -> str | None:
    effective = _effective_model_cfg(model_cfg)
    if effective.get("selected_variant"):
        return str(effective["selected_variant"])
    if model_cfg.get("variant_env"):
        return os.environ.get(str(model_cfg["variant_env"]))
    return None


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
        "model_path": _model_path_from_cfg(cfg["model"]),
        "model_variant": _model_variant_from_cfg(cfg["model"]),
        "prompt_version": cfg["model"]["prompt_version"],
        "config_hash": cfg["config_hash"],
        "ffmpeg_version": command_version(["ffmpeg", "-version"]),
    }
    out = ensure_dir(cfg["processed_dir"]) / "video_manifest.json"
    write_json(out, row)
    return row
