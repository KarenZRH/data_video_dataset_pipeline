from __future__ import annotations

from importlib import import_module
from typing import Any


def make_model_client(cfg: dict[str, Any]):
    module_name = cfg.get("model", {}).get("client_module", "datavideo.qwen_vl")
    module = import_module(module_name)
    class_name = cfg.get("model", {}).get("client_class")
    if class_name:
        return getattr(module, class_name)(cfg)
    if hasattr(module, "QwenVLClient"):
        return module.QwenVLClient(cfg)
    if hasattr(module, "GeminiFlashClient"):
        return module.GeminiFlashClient(cfg)
    raise RuntimeError(f"Model client module {module_name} does not expose a supported client class")


def make_qwen_client(cfg: dict[str, Any]):
    return make_model_client(cfg)
