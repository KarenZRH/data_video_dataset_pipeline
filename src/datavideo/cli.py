from __future__ import annotations

import argparse
from typing import Any

from .manifest import load_config
from .quality import run_quality_check
from datavideo.multichart_pipeline import (
    run_asr_pipeline,
    run_context_pipeline,
    run_pipeline,
)
from datavideo.multichart_reviewed_outputs import apply_latest_reviews


def run_command(command: str, cfg: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    if command == "context":
        return run_context_pipeline(cfg, force=force)
    if command == "asr":
        return run_asr_pipeline(cfg, force=force)
    if command == "assets":
        return run_pipeline(cfg, force=force)
    if command == "reviewed":
        return apply_latest_reviews(cfg)
    if command == "quality":
        return run_quality_check(cfg, force=force)
    raise ValueError(f"Unsupported command: {command}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Web-annotated multichart data-video dataset pipeline")
    parser.add_argument(
        "command",
        choices=[
            "context",
            "asr",
            "assets",
            "quality",
            "reviewed",
        ],
        help="Pipeline stage to run.",
    )
    parser.add_argument("--config", default="configs/multichart_assets_base.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--clip-id", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.clip_id:
        cfg["clip_id"] = args.clip_id

    report = run_command(args.command, cfg, force=args.force)
    print(report)


if __name__ == "__main__":
    main()
