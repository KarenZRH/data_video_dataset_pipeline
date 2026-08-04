# Repository Map

This repository now exposes a single current dataset workflow: the
web-annotated multichart v2 pipeline.

## Current Mainline

| Area | Files |
| --- | --- |
| CLI entry point | `src/datavideo/cli.py` |
| Main workflow package | `src/datavideo_multichart_v2/` |
| Shared utilities | `src/datavideo/` |
| Main config | `configs/multichart_assets_v2.yaml` |
| Optional Gemini config | `configs/multichart_assets_gemini.yaml` |
| Main docs | `docs/workflow.md`, `docs/工作流.md` |
| Review app | `app/multichart_v2_review_app.py` |
| Metadata fetcher | `scripts/fetch_datavideo_clips.py` |

## CLI Commands

| Command | Purpose |
| --- | --- |
| `context` | Create context media/audio and strict visual clips from webpage intervals. |
| `asr` | Transcribe context audio and select complete narration sentences. |
| `assets` | Generate keyframes, semantic SVGs, animation descriptions, and chart data. |
| `quality` | Run deterministic and optional VLM quality checks. |
| `reviewed` | Rebuild reviewed outputs from the latest review database records. |

All commands default to:

```text
configs/multichart_assets_v2.yaml
```

## Shared Utility Modules

| Module | Purpose |
| --- | --- |
| `context.py` | Context and visual-clip media creation. |
| `narration.py` | ASR sentence boundaries and subtitle selection. |
| `frames.py`, `media.py` | FFmpeg frame/media helpers. |
| `keyframes.py` | Still extraction and image-motion scoring helpers used by v2. |
| `dynamic_data.py` | Dynamic/static chart data normalization and fusion. |
| `semantic.py`, `semantic_components.py`, `svg_trace.py` | SVG and semantic component generation. |
| `quality.py` | Dataset artifact quality checks. |
| `qwen_vl.py`, `gemini_vl.py`, `model_client.py` | Model backends and model-client factory. |
| `review_db.py` | SQLite review record storage. |
| `schemas.py` | JSON/CSV/path helper functions and stable hashing. |
| `visual_provenance.py` | Guardrails for visual evidence inputs. |

## Removed Legacy Lines

The old broad detector, old bar-dominant experiment, and multichart v1 package
were removed from the runnable repository surface. This keeps the project easier
to hand off: new users should not choose between competing historical pipelines.

## Tracked Metadata

| File | Purpose |
| --- | --- |
| `data/raw/datavideo_clips.jsonl` | Small selected clip metadata used by examples. |
| `data/raw/datavideo_clips_all.jsonl` | Full web-annotated clip metadata from the first clip site. |
| `data-video-list-with-clips.csv` | Consolidated source-video URL list collected from `websites.txt`. |
| `websites.txt` | Source webpages used for URL collection. |

Generated media and review outputs remain local and ignored by git.
