# data_video_dataset

Prepare a workflow for turning long videos into chart-centric training data:

1. Find chart segments in long videos.
2. Save representative keyframes.
3. Convert chart frames to SVG.
4. Describe how the chart moves, what the narration says, and when the two align.

## Environment

The conda environment is:

```bash
/path/to/workspace/miniconda3/bin/conda activate DataVideo
```

Installed in the conda environment:

- Python 3.11
- git
- ffmpeg

System packages requested:

- git: installed
- build-essential: installed
- libgl1: installed
- ffmpeg: sudo password is required for apt installation, but ffmpeg is available inside the `DataVideo` conda environment

If you want to complete the system-level install later:

```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg build-essential libgl1
```

## Project Layout

```text
data/
  raw/videos/      long source videos
  keyframes/       selected chart frames
  svg/             vectorized chart outputs
  audio/           extracted audio tracks
  transcripts/     speech-to-text output with timestamps
  annotations/     segment, motion, narration, and alignment metadata
configs/           workflow configuration
docs/              project notes and workflow design
notebooks/         experiments
scripts/           setup and utility scripts
src/               Python package code
```

## First Check

```bash
bash scripts/check_setup.sh
```

## Current Baseline

The current experiment baseline is **bar-chart-dominant clip detection**. From frame sampling through clip extraction and asset generation, use the latest `bar-dominant` and `bar-assets` flow.

Target definition:

- Positive only when the main narrative unit is expressed by bar marks.
- Bar marks include vertical bars, horizontal bars, grouped/stacked bars, bar-race bars, or bar-like marks whose length, position, order, or label encodes data.
- Exclude clips where bars appear only briefly and the segment becomes circle, bubble, distance line, map, icon, illustration, decorative motion, or another non-bar visual encoding.
- Mixed segments should be trimmed to the bar-dominant subclip or excluded.

Current input:

```text
data/raw/videos/bar_sample.mp4
```

Current preprocessed media:

```text
data/processed/bar_001/normalized.mp4
data/processed/bar_001/audio_16k_mono.wav
data/processed/bar_001/frames/coarse_2fps/
data/generated/bar_001/frame_manifest.jsonl
```

The current four-step detection flow is:

1. Qwen detects bar-chart-dominant frames from existing 2 FPS frames.
2. Positive frames become bar candidates.
3. Adjacent candidates are merged if the gap is <= 2 seconds.
4. Qwen reviews each merged candidate contact sheet for complete bar-dominant data narrative semantics.

Run the current baseline:

```bash
source /path/to/workspace/miniconda3/bin/activate DataVideo
export PYTHONPATH=/path/to/workspace/projects/data_video_dataset/src
export MODEL_PATH=/path/to/qwen-vl-model
CUDA_VISIBLE_DEVICES=0 python -m datavideo.cli bar-dominant --config configs/stage1_bar.yaml --force
CUDA_VISIBLE_DEVICES=0 python -m datavideo.cli bar-assets --config configs/stage1_bar.yaml --force
```

Current outputs:

```text
data/generated/bar_001/bar_candidates.jsonl
data/generated/bar_001/bar_merged_clips.jsonl
data/generated/bar_001/final_bar_clips.jsonl
data/generated/bar_001/bar_final_xxx.mp4
data/generated/bar_001/bar_final_xxx_contact_sheet.jpg
```

Latest run result:

```text
frames: 200
bar candidates: 5
merged candidates: 4
final clips: 2

bar_final_000: 16.000s - 27.000s
bar_final_001: 70.000s - 78.000s
```

Merged candidate videos/contact sheets are used only in a temporary review directory. Only accepted final clips are saved.

Old generic data-video detection outputs were intentionally cleared. Avoid using the older `detect` or `merge-review` outputs as the active baseline.

Outputs are separated by purpose:

- `data/raw/videos/`: source long videos
- `data/processed/bar_001/`: normalized media and sampled frames
- `data/generated/bar_001/`: machine-generated manifests, clips, keyframes, SVG, chart data
- `data/generated/bar_001/clips/<clip_id>/`: per-clip video, `keyframes/initial.png`, SVG, and chart data
- `data/reviewed/bar_001/`: human-reviewed values
- `data/review.db`: SQLite audit records

Review UI:

```bash
streamlit run app/review_app.py --server.address 127.0.0.1 --server.port 8501
```

## Multichart Clip Assets

The newer multichart data-video clip workflow lives in a separate package:

```text
src/datavideo_multichart/
```

It reads clip metadata from `data/raw/datavideo_clips.jsonl`, normalizes each
clip from `data/raw/videos/`, extracts audio and candidate frames into
`data/processed/<clip_id>/`, and writes per-clip assets into
`data/generated/<clip_id>/`.

Run it with the `DataVideo` conda environment:

```bash
source /path/to/workspace/miniconda3/bin/activate DataVideo
export PYTHONPATH=/path/to/workspace/projects/data_video_dataset/src
export MODEL_PATH=/path/to/qwen-vl-model
python -m datavideo.cli multichart-assets --config configs/multichart_assets.yaml
```

Each completed clip directory contains `clip.mp4`, `keyframes/initial.png`,
`trace.svg`, `trace_preview.png`, and JSON reports. `chart_data.csv` is written
only when the selected keyframe contains concrete readable chart data.

The Qwen model path is read from `MODEL_PATH`. The implementation prefers BF16 when CUDA reports BF16 support, otherwise FP16. If Qwen chart detection is unavailable, the pipeline records the failure and does not create positive clips from fallback guesses.

Run tests:

```bash
PYTHONPATH=src pytest -q
```

## Older Detection Experiments

These commands are kept for reference only. They target broader data-video candidates, not the current bar-chart-dominant definition:

```bash
source /path/to/workspace/miniconda3/bin/activate DataVideo
export PYTHONPATH=/path/to/workspace/projects/data_video_dataset/src
export MODEL_PATH=/path/to/qwen-vl-model
python -m datavideo.cli detect --config configs/stage1_bar.yaml --force
python -m datavideo.cli merge-review --config configs/stage1_bar.yaml --force
```

Do not continue new experiments from `data/reviewed/bar_001/candidates/` or old generic `merged_clips.jsonl`.

## Next Build Steps

- Use `data/generated/bar_001/final_bar_clips.jsonl` as the source of accepted clips.
- `bar-assets` now selects a complete static-form keyframe for each accepted bar clip, writes VTracer `trace.svg`, recovers chart data, and prepares Streamlit review assets.
- Later: add animation description, Whisper transcription, narration-animation alignment, and Streamlit/manual review for the revised schema.
