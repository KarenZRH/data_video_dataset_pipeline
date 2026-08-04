# Data Video Dataset Workflow

Canonical workflow for the current web-annotated multichart v2 dataset:

- The webpage `reference_start/reference_end` interval is the authoritative visual clip boundary.
- `context.mp4` extends the webpage interval by 5 seconds before and after, and is used only for complete narration extraction.
- Qwen2.5-VL is used only for visual tasks on `visual_clip.mp4` or frames derived from it.
- The current web-annotated data does not use Qwen to re-detect chart boundaries.
- Complete narration may start before or end after the visual clip. This must not expand `visual_clip.mp4`.
- Later animation/narration alignment uses source-video time coordinates.

## Environment

Use local models only:

```bash
conda activate DataVideo
export PYTHONPATH=src
export MODEL_PATH=/path/to/qwen-vl-model
export WHISPER_MODEL_PATH=/path/to/faster-whisper-model
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export HTTP_PROXY=http://127.0.0.1:<port>
export HTTPS_PROXY=http://127.0.0.1:<port>
```

Main config:

```text
configs/multichart_assets_v2.yaml
```

Do not commit cookies, models, downloaded videos, generated assets, or reviewed dataset artifacts.

## Time Coordinates

All time values are seconds.

- `reference_source`: authoritative webpage visual interval in source-video time.
- `context_source`: source-video interval used to create `context.mp4`.
- `visual_clip_context`: visual clip interval inside `context.mp4`.
- `visual_clip_relative`: visual clip interval inside `visual_clip.mp4`, always starting at `0.0`.
- Full narration sentence times use source/context coordinates and may exceed the visual clip.
- Visual subtitles use visual-clip-relative coordinates and must stay within `[0, visual_clip_duration]`.

Context padding:

```yaml
context:
  padding_before_seconds: 5.0
  padding_after_seconds: 5.0
```

Formula:

```text
context_start = max(0, reference_start - 5.0)
context_end = min(source_duration, reference_end + 5.0)
visual_start_in_context = reference_start - context_start
visual_end_in_context = reference_end - context_start
```

## Directory Layout

Processed per-clip files:

```text
data/processed/<clip_id>/
  context.mp4
  context_audio_16k_mono.wav
  visual_clip.mp4
  visual_clip_report.json
  intervals.json
  visual_frames/
    keyframe_candidates/
    keyframe_frame_manifest.jsonl
  narration/
    context_transcript_raw.json
    context_segments.jsonl
    context_words.jsonl
    sentence_boundaries.jsonl
    selected_full_sentences.jsonl
    overlap_words.jsonl
    subtitles.srt
    transcript_provenance.json
    asr_report.json
```

Generated visual assets:

```text
data/generated_v2/<clip_id>/
  clip.mp4                  # copy/normalized equivalent of processed/<clip_id>/visual_clip.mp4
  keyframes/
    initial.png
    states/
    keyframe_manifest.json
    keyframe_scores.jsonl
  semantic.svg
  semantic_preview.png
  animation_detection_raw.json
  animation_detection.json
  chart_data_clip_raw.json
  chart_data.csv
  dynamic_data.json
  dynamic_data.csv
  final_data_table.csv
  data_change_events.csv
  data_events.jsonl
  chart_data_validation.json
  clip_report.json
```

Reviewed outputs:

```text
data/reviewed/datavideo_multichart_v2/clips/<clip_id>/
  clip.mp4                  # strict webpage reference visual clip
  intervals.json
  keyframes/final.png
  semantic.svg
  semantic_preview.png
  animation_detection.json
  animation_reviewed.json
  chart_data.csv
  dynamic_data.json
  dynamic_data.csv
  final_data_table.csv
  data_change_events.csv
  data_events.jsonl
  narration_reviewed.json
  narration/
    selected_full_sentences.jsonl
    overlap_words.jsonl
    subtitles.srt
    transcript_provenance.json
    narration_status.json
  context_narration/
  review.json
  clip.json
```

Review records are stored separately:

```text
data/review/review.db
```

`data/reviewed/...` is reserved for rebuilt final reviewed artifacts only.

## Dynamic Data Recovery

`dynamic_data.json` and `dynamic_data.csv` are the canonical machine outputs for recovered data states. `final_data_table.csv` stores the latest known value or qualitative fact per entity, `data_change_events.csv` stores insert/update/remove events, `chart_data.csv` remains the review-table compatible view, and `data_events.jsonl` mirrors change events for downstream animation tooling.

Each dynamic state row contains:

```text
clip_id,state_id,entity_id,metric,value,unit,state_start,state_end,
source_type,evidence_frames,evidence_sentence_id,confidence,review_status
```

Recovery uses the visual clip first. Qwen is asked to recover only values printed in selected evidence frames; it must not convert bar length or line position into a true value unless a printed value or reliable scale is visible. If no visual value is recoverable, narration still runs and may provide exact numeric facts from complete overlapping/corresponding sentences. English number words are supported, such as `two full days -> 2 day`. If at least one reliable numeric fact exists, the clip may be included as `data_completeness=partial`; missing entities remain qualitative/null. If neither source has verifiable quantitative data, the clip is excluded with `exclude_reason=no_recoverable_quantitative_data`.

Visual and narration evidence are fused by stable identity. Matching values become `source_type=both`; conflicts keep both pieces of evidence and set `review_status=needs_review`. A narration sentence that only says increase/decrease without a number is not a numeric data row. Qualitative narration such as `a lot of time` may be retained as a partial, non-numeric entity fact.

Sampling is two-stage: keep the coarse scan at 2 FPS, use cheap frame/chart-region/OCR-change signals to mark possible data-change windows, extract state/evidence frames at 8 FPS only inside those windows, and fall back to source FPS inside a window only when the fine sample is still too sparse. Qwen is called only on selected representative state/evidence frames, not on every high-FPS frame.

State merging is local in time: only consecutive rows with the same entity set and values are merged, preserving `state_start`, `state_end`, and evidence. `A -> A -> B -> B -> A` remains three states (`A`, `B`, `A`), while pure animation interpolation with unchanged printed data remains one state. Entity additions are `insert`, value changes are `update`, and disappearances are `remove`.

## Quality Control

After keyframes, data tables, semantic SVGs, animation detection, and narration are generated, run quality control to produce a review queue:

```bash
python -m datavideo.cli quality-check --config configs/multichart_assets_v2.yaml
```

The checker uses three layers:

1. Python rule checks for required files, JSON/CSV schema, include/exclude consistency, state counts, and event consistency.
2. Python cross-artifact checks for `dynamic_data` entity IDs against `semantic.svg` and state-level semantic SVGs.
3. Optional VLM review over keyframes, semantic previews, dynamic data summaries, animation detection, and narration-derived evidence.

The VLM quality model is configured separately under `quality.model`, so it can use a different provider from the production model. `quality.enable_vlm=false` runs only the deterministic Python checks. Outputs are written to:

```text
data/generated_v2/quality/quality_report.json
data/generated_v2/quality/quality_flags.csv
data/generated_v2/quality/quality_review_queue.csv
```

## Intervals

`data/processed/<clip_id>/intervals.json` stores machine/source facts only:

```json
{
  "time_unit": "seconds",
  "reference_source": {"start": 65.0, "end": 77.0},
  "context_source": {"start": 60.0, "end": 82.0},
  "visual_clip_context": {"start": 5.0, "end": 17.0},
  "visual_clip_relative": {"start": 0.0, "end": 12.0},
  "requires_context_redownload": false
}
```

Do not write narration-complete boundaries as visual clip boundaries. Current canonical intervals do not contain `chart_context`, `chart_source`, `proposed_clip_context`, `proposed_clip_source`, `reviewed_clip_context`, or `reviewed_clip_source`.

If true context cannot be downloaded and the pipeline falls back to the exact webpage clip, set:

```json
"requires_context_redownload": true
```

This does not block visual asset generation. It only means complete narration is incomplete and must not be forged.

## 1. Metadata

Fetch webpage metadata:

```bash
python scripts/fetch_datavideo_clips.py \
  --clips-per-chart 2 \
  --jsonl data/raw/datavideo_clips.jsonl \
  --video-dir data/raw/videos \
  --cookies www.youtube.com_cookies.txt \
  --proxy http://127.0.0.1:<port>
```

The webpage interval is the visual truth for this dataset.

## 2. Context And Visual Clip

Generate context media, context audio, and strict visual clip:

```bash
PYTHONPATH=src python -m datavideo.cli context \
  --config configs/multichart_assets_v2.yaml
```

Single clip:

```bash
PYTHONPATH=src python -m datavideo.cli context \
  --config configs/multichart_assets_v2.yaml \
  --clip-id bar_1
```

`visual_clip.mp4` is derived from `context.mp4` when possible:

```text
visual_clip.mp4 = context.mp4[visual_start_in_context:visual_end_in_context]
```

The visual clip duration should match `reference_end - reference_start`, allowing normal encoding tolerance. `visual_clip_report.json` records the expected duration, ffprobe duration, and duration error.

## 3. Context ASR And Full Narration

Run ASR only on context audio:

```bash
PYTHONPATH=src \
WHISPER_MODEL_PATH=/path/to/faster-whisper-model \
python -m datavideo.cli asr \
  --config configs/multichart_assets_v2.yaml
```

Input:

```text
data/processed/<clip_id>/context_audio_16k_mono.wav
```

Selection rules:

- Select sentences that actually overlap `reference_source`.
- Keep a selected sentence's full source/context time, even if it starts before or ends after the visual clip.
- Do not add neighboring sentences that do not overlap `reference_source`.
- Use punctuation, Whisper segments, word timestamps, and clear pauses to infer sentence boundaries.
- Mark `needs_review` for low confidence or ambiguous/context-edge boundaries.
- If `requires_context_redownload=true`, set narration status to `incomplete_context`.
- Do not convert out-of-clip full sentence times into fake visual subtitle times.

Outputs:

```text
selected_full_sentences.jsonl  # full source/context sentence ranges
overlap_words.jsonl            # only words overlapping the visual clip
subtitles.srt                  # visual-clip-relative, clamped to clip duration
transcript_provenance.json
```

## 4. Visual Assets

Generate keyframes, semantic SVG, and chart data from the strict visual clip:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=src \
MODEL_PATH=/path/to/qwen-vl-model \
python -m datavideo.cli assets \
  --config configs/multichart_assets_v2.yaml
```

`assets` reads:

```text
data/processed/<clip_id>/visual_clip.mp4
```

It writes `data/generated_v2/<clip_id>/clip.mp4` as the visual clip used for review and then generates:

- candidate keyframes from visual-clip frames;
- Qwen keyframe scores;
- `keyframes/initial.png`;
- optional `keyframes/states/state_*.png`;
- a Qwen whole-clip target-chart animation description;
- optional major target-chart animation actions with evidence timestamps;
- `semantic.svg` and `semantic_preview.png`;
- chart data from visual-clip frame sequences.

Animation-detection rule:

```text
Use only frames sampled from processed/<clip_id>/visual_frames for the strict visual clip.
Sample the complete visual clip at `animation.sample_fps` and ask Qwen to observe the ordered frames in one call.
Qwen must produce exactly one concise Chinese `overall_description` sentence for animation related to the target chart type.
`major_actions` may be empty or contain any number of genuinely distinct actions; do not split actions merely to fill a quota.
Only target-chart-relevant actions enter `major_actions`.
```

Canonical config:

```yaml
animation:
  sample_fps: 2
  types:
    - no_clear_animation
    - bar_grow
    - bar_shrink
    - line_draw_upward
    - line_draw_downward
    - pie_or_donut_segments_appear
    - map_region_highlight
    - chart_type_transition
    - element_appear
    - element_disappear
    - element_highlight
    - other
  purpose: one-sentence overall target-chart animation description from the complete visual clip
```

Target relevance:

```text
Detect only animation related to the target chart type, or happening inside the target chart.
Relevant animation includes:
1. changes inside the target chart;
2. changes directly affecting target chart data elements, axes, legend, labels, or annotations;
3. the target chart as a whole appearing, disappearing, or being highlighted;
4. chart-type transitions where the target chart is the start or end state.

Ignore ordinary subtitles, logos, people, backgrounds, decorative elements, unrelated chart types, and non-target chart changes.
```

Whole-clip consistency constraint:

```text
For data-element animation types such as bar_grow, bar_shrink, line_draw_upward, and line_draw_downward, evidence timestamps must refer to the same target chart scene.
Do not treat a transition from a non-chart/decorative scale/title scene into the target chart as data-element growth or line drawing.
Such a cross-scene change may be retained only when it is explicitly the whole target chart appearing or a chart-type transition whose start or end state is the target chart.
```

Action continuity and merge rule:

```text
Continuous changes acting on the same target chart object, with the same direction and continuous visual state, are one complete major action.
Do not split one bar growth/shrink or line rise/fall because of speed changes, easing, a short pause, or one intermediate low-change frame.

Split only when:
1. animation direction reverses;
2. target object clearly changes;
3. animation type changes;
4. a clear stable state or another independent animation appears in between.
```

Canonical coarse animation types:

```text
no_clear_animation
bar_grow
bar_shrink
line_draw_upward
line_draw_downward
pie_or_donut_segments_appear
map_region_highlight
chart_type_transition
element_appear
element_disappear
element_highlight
other
```

Animation outputs:

```text
animation_detection_raw.json       # Qwen raw response and provenance
animation_detection.json           # normalized machine animation result
```

`animation_detection.json` must use this whole-clip structure:

```json
{
  "clip_id": "bar_1",
  "target_chart_type": "bar",
  "sample_fps": 2,
  "frame_count": 18,
  "visual_start": 0.0,
  "visual_end": 9.0,
  "is_target_chart_related": true,
  "overall_description": "目标柱状图中的蓝色柱子持续下降，随后数值标签出现。",
  "major_actions": [
    {
      "action": "bar_shrink",
      "description": "蓝色柱子持续下降。",
      "evidence_timestamps": [1.0, 2.5, 4.0]
    },
    {
      "action": "element_appear",
      "description": "柱子上方的数值标签出现。",
      "evidence_timestamps": [6.5, 7.5]
    }
  ],
  "confidence": 0.88
}
```

If no clear animation is detected:

```json
{
  "is_target_chart_related": false,
  "overall_description": "没有检测到与目标图表相关的动画。",
  "major_actions": [],
  "confidence": 0.0
}
```

`multichart_v2_clips.jsonl` also exposes the reviewed workflow's clip-level lookup fields:

```text
animation_description       # copied from animation_detection.overall_description
animation_action_count
animation_confidence
is_target_chart_related
```

Chart-data rule:

```text
Only recover values directly printed in the frames. Do not estimate values from geometry.
```

## Qwen Input Rules

Allowed Qwen inputs:

- `data/processed/<clip_id>/visual_frames/...`;
- `data/generated_v2/<clip_id>/keyframes/...`;
- frames or stills derived from `visual_clip.mp4`.

Forbidden Qwen inputs:

- `context.mp4`;
- frames extracted from context;
- context boundary frames;
- any frame outside the webpage reference interval;
- `context_audio_16k_mono.wav`.

The v2 asset code validates Qwen image paths and records all input frame paths in keyframe/data reports.

## 5. Human Review

Canonical visual review page:

```bash
PYTHONPATH=src streamlit run app/multichart_v2_review_app.py
```

The page reviews:

- visual clip;
- selected keyframe and state frames;
- animation overall description and optional major-action table;
- narration sentences and editable reviewed narration text;
- PNG/SVG semantic outputs;
- recovered chart data;
- asset decision.

New review records use English decisions:

```text
approved        -> enters final visual dataset
needs_revision  -> blocks final dataset
saved           -> draft only
excluded        -> excluded list
```

Final visual dataset entry condition:

```text
asset_review.decision == "approved"
```

The same asset review record stores human-reviewed animation fields under `reviewed_value.animation`, with `overall_description`, `major_actions`, and `confidence`. Machine animation remains in `data/generated_v2/<clip_id>/animation_detection.json`; reviewed animation is written by `reviewed` to `animation_reviewed.json`.

The same asset review record also stores reviewed narration under `reviewed_value.narration`. Reviewers may edit sentence text or clear `keep_in_reviewed` for sentences that should not enter the final dataset. Machine ASR remains copied for audit; reviewed narration is written to `narration_reviewed.json`.

## Deprecated Boundary Review

The clip-boundary review UI is no longer part of the canonical workflow for
webpage-annotated data. The current authoritative visual boundary is the
webpage `reference_start/reference_end` interval, so clip-boundary review is
kept only for older records and should not be used to generate new official
outputs.

## 6. Reviewed Dataset

Rebuild reviewed outputs:

```bash
PYTHONPATH=src python -m datavideo.cli reviewed \
  --config configs/multichart_assets_v2.yaml
```

This reads latest asset reviews from `data/review/review.db` and writes final reviewed artifacts to `data/reviewed/datavideo_multichart_v2`.

It does not require or apply old clip-boundary reviews. Final `clip.mp4` is copied from `data/generated_v2/<clip_id>/clip.mp4`, which corresponds to the webpage reference interval.
Reviewed animation is copied from the approved asset-review value into `animation_reviewed.json`; machine raw reports are copied for audit. `clip.json` and `final_multichart_v2_clips.jsonl` expose `animation_description` for easy filtering and inspection.

Narration files are copied from processed ASR outputs when available:

- `selected_full_sentences.jsonl`;
- `overlap_words.jsonl`;
- `subtitles.srt`;
- `transcript_provenance.json`;
- `narration_status.json`.

Reviewed narration is written to:

- `narration_reviewed.json`;
- `narration/narration_reviewed.json`.

## Cache Invalidation

- Reference/context interval changes: regenerate context, visual clip, and ASR.
- Visual clip changes: regenerate keyframes, animation detection, SVG, and chart data.
- ASR model or context audio changes: regenerate narration.
- Human narration edits: do not rerun ASR.
- Narration edits: do not regenerate visual assets.
- Asset review edits: rerun `reviewed`.

## Later Alignment

Animation detection reads only `visual_clip.mp4` derived frames.

Use source-video time for alignment:

```text
animation_event.start_source = reference_source.start + animation_event.start_visual
animation_event.end_source = reference_source.start + animation_event.end_visual
narration_source_time = selected_full_sentences[*].start_source/end_source
```

Semantic matching must use human-reviewed full narration when available. Raw context ASR is audit/provenance only.

Complete narration may extend outside the visual clip. That is expected and must not change the visual clip.

## Legacy Migration

Old generated assets may have been based on proposed or reviewed narration-complete clip boundaries. Treat them as stale for this canonical dataset.

Migration order:

1. Run `context` to create `context.mp4`, `context_audio_16k_mono.wav`, and strict `visual_clip.mp4`.
2. Run `asr` to derive full overlapping narration.
3. Run `assets` to regenerate visual assets from `visual_clip.mp4`.
4. Review assets in `app/multichart_v2_review_app.py` and save an English `approved` decision for accepted clips.
5. Run `reviewed`.

Do not delete old `review.db` records. Old clip-boundary reviews are ignored by the canonical rebuild. If an old asset review used a non-English decision, re-save it with the English enum before expecting it to enter the final dataset.
