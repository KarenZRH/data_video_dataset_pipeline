# data_video_dataset

Pipeline and metadata for building a chart-centric data-video clip dataset from
web-annotated source videos.

The repository has been consolidated around one current workflow:

- Main workflow and shared utilities: `src/datavideo/`
- Main config: `configs/multichart_assets_v2.yaml`
- Workflow guide: `docs/workflow.md`
- Review app: `app/multichart_v2_review_app.py`

Older broad-detection, bar-dominant, and multichart-v1 experiment lines have
been removed from the runnable surface so new users start from the same path.
See `docs/repo_map.md` for the current repository map.

## Environment

```bash
conda activate DataVideo
export PYTHONPATH=src
export MODEL_PATH=/path/to/qwen-vl-model
export WHISPER_MODEL_PATH=/path/to/faster-whisper-model
export HTTP_PROXY=http://127.0.0.1:<port>
export HTTPS_PROXY=http://127.0.0.1:<port>
```

Install/check:

```bash
bash scripts/check_setup.sh
PYTHONPATH=src pytest -q
```

## Current Workflow

Run the canonical web-annotated multichart v2 stages:

```bash
PYTHONPATH=src python -m datavideo.cli context --config configs/multichart_assets_v2.yaml
PYTHONPATH=src python -m datavideo.cli asr --config configs/multichart_assets_v2.yaml
PYTHONPATH=src python -m datavideo.cli assets --config configs/multichart_assets_v2.yaml
PYTHONPATH=src python -m datavideo.cli quality --config configs/multichart_assets_v2.yaml
PYTHONPATH=src python -m datavideo.cli reviewed --config configs/multichart_assets_v2.yaml
```

For one clip:

```bash
PYTHONPATH=src python -m datavideo.cli assets \
  --config configs/multichart_assets_v2.yaml \
  --clip-id bar_1
```

Review UI:

```bash
PYTHONPATH=src streamlit run app/multichart_v2_review_app.py
```

## Metadata

Tracked source metadata:

- `data-video-list-with-clips.csv`: canonical source-video URL list collected
  from `websites.txt`; the pipeline uses rows with start/end times.
- `data/raw/datavideo_clips.jsonl`: legacy example JSONL kept for reference.
- `data/raw/datavideo_clips_all.jsonl`: legacy extracted clip metadata from the
  first annotation site.

To refresh webpage clip metadata:

```bash
python scripts/fetch_datavideo_clips.py \
  --clips-per-chart 2 \
  --jsonl data/raw/datavideo_clips.jsonl \
  --video-dir data/raw/videos \
  --cookies www.youtube.com_cookies.txt \
  --proxy http://127.0.0.1:<port>
```

The script can also download clips, but downloaded media is ignored by git.

## What Not To Commit

Do not commit cookies, model weights, downloaded videos, generated assets,
review databases, logs, or local cache directories. `.gitignore` already covers
the standard locations.
