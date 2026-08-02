from __future__ import annotations

from importlib import import_module
from typing import Any


def make_qwen_client(cfg: dict[str, Any]):
    module_name = cfg.get("model", {}).get("client_module", "datavideo.qwen_vl")
    module = import_module(module_name)
    return module.QwenVLClient(cfg)
