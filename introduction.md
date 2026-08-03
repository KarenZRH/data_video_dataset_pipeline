# 当前工作流

当前主线是 **bar-chart-dominant data-video clip**：从一个 raw video 中找出以柱状/条形 mark 为主要数据表达的片段，生成 clip、关键帧、semantic SVG、图表数据草稿，再进入 Streamlit 审核页。

## 0. 当前 baseline 和入口

当前推荐流程不是旧的通用 `detect` / `merge-review`，而是：

```bash
source /path/to/workspace/miniconda3/bin/activate DataVideo
export PYTHONPATH=/path/to/workspace/projects/data_video_dataset/src
export MODEL_PATH=/path/to/qwen-vl-model

CUDA_VISIBLE_DEVICES=0 python -m datavideo.cli stage0 --config configs/stage1_bar.yaml --force

# 如果 processed_dir/frames/coarse_2fps/ 或 generated_dir/frame_manifest.jsonl 已存在，
# 可以直接进入当前 baseline。
CUDA_VISIBLE_DEVICES=0 python -m datavideo.cli bar-dominant --config configs/stage1_bar.yaml --force
CUDA_VISIBLE_DEVICES=0 python -m datavideo.cli bar-assets --config configs/stage1_bar.yaml --force
streamlit run app/review_app.py --server.address 127.0.0.1 --server.port 8501
```

注意：当前代码里还没有“只做粗抽帧”的独立 CLI。`bar-dominant` 会复用 `generated_dir/frame_manifest.jsonl`、`processed_dir/frame_manifest.jsonl` 或 `processed_dir/frames/coarse_2fps/coarse_*.jpg`。如果是一个完全新的 raw video，需要先用旧 `stage1` / `detect` 的副作用生成 2 FPS 粗帧，或补一个专门的抽帧入口；当前已有样本目录里已经存在这些粗帧。

`configs/stage1_bar.yaml` 和 `configs/stage1_bar_sample2.yaml` 结构一致，只是 sample 路径不同。关键配置如下：

- `video_path`: raw video，例如 `data/raw/videos/bar_sample.mp4`
- `processed_dir`: 标准化视频、音频、抽帧输出目录
- `generated_dir`: 模型结果、候选片段、最终机器生成资产目录
- `reviewed_dir`: 人工审核后的最终输出目录
- `review_db`: SQLite 审核记录，默认 `data/review.db`
- `model.env_var`: 模型路径从环境变量 `MODEL_PATH` 读取
- `model.max_frames_per_call`: 每次 Qwen 调用最多输入 3 张连续帧
- `video_standardization.fps`: 标准化为 30 FPS
- `sampling.coarse_fps`: 粗抽帧 2 FPS
- `sampling.fine_fps`: 细抽帧 8 FPS，主要用于旧 `stage1/detect` 和 review package；当前 `bar-dominant` 的二次 contact sheet 复核固定用 2 FPS
- `sampling.short_side`: 抽帧短边缩放到 768
- `target_search.start_times_path`: 目标片段大致起点文件，默认 `data/raw/start_time.jsonl`
- `target_search.before_seconds` / `after_seconds`: 初始搜索窗，默认 `[clip_start-2s, clip_start+20s]`
- `target_search.extension_seconds`: positive 延伸到右边界时每次向后扩展 10 秒
- `target_search.boundary_positive_gap_seconds`: 最后一个 positive 距右边界不超过 1 秒时触发扩展
- `detection.min_confidence`: 0.5
- `detection.max_gap_seconds`: 1.5，旧通用 detect 使用
- `vtracer`: `color_precision=6`、`filter_speckle=4`、`mode=spline`、`hierarchical=stacked`、`path_precision=8`

## 1. Stage0：raw video 标准化

入口：`python -m datavideo.cli stage0 --config ...`

代码路径：`src/datavideo/cli.py` -> `stage0()` -> `src/datavideo/media.py::normalize_video()`

处理内容：

1. 创建 `processed_dir`、`generated_dir`、`reviewed_dir`。
2. 初始化 SQLite 表 `reviews`。
3. 写 `processed_dir/video_manifest.json`，包含 source video 路径、sha256、`MODEL_PATH`、prompt version、config hash、ffmpeg version。
4. 用 ffmpeg 标准化视频：
   - 输入：`cfg["video_path"]`
   - 输出：`processed_dir/normalized.mp4`
   - 视频编码：`libx264`
   - pixel format：`yuv420p`
   - FPS：30
   - CRF：18
   - preset：`medium`
   - audio codec：`aac`
   - audio sample rate：48000
   - `-movflags +faststart`
5. 从标准化视频抽音频：
   - 输出：`processed_dir/audio_16k_mono.wav`
   - wav sample rate：16000
   - channels：1
6. 写 `processed_dir/standardization_report.json`，保存 normalized video、wav 路径和 ffprobe 信息。

## 2. 粗抽帧：每秒 2 帧

当前 `bar-dominant` 流程默认复用已有 frame manifest；如果不存在，就从 `processed_dir/frames/coarse_2fps/` 读已有图片；通常由 `stage1`、`detect` 或之前的处理先抽好。

抽帧函数：`src/datavideo/frames.py::extract_frames()`

ffmpeg 参数逻辑：

```text
ffmpeg -y -i normalized.mp4 \
  -vf fps=2,scale='if(gt(iw,ih),-2,768)':'if(gt(iw,ih),768,-2)' \
  -q:v 2 \
  processed_dir/frames/coarse_2fps/coarse_%06d.jpg
```

输出 manifest 行结构：

```json
{
  "frame_id": "coarse_000001",
  "path": "data/processed/.../frames/coarse_2fps/coarse_000001.jpg",
  "timestamp": 0.0,
  "fps": 2,
  "sample_type": "coarse"
}
```

最终写入：`generated_dir/frame_manifest.jsonl`

## 3. 帧级 bar-dominant 识别

入口：`python -m datavideo.cli bar-dominant --config ...`

代码路径：`src/datavideo/bar_dominant.py::run_bar_dominant_pipeline()`

模型客户端：`src/datavideo/qwen_vl.py::QwenVLClient`

### 3.1 根据人工起点缩小搜索范围

目标 clip 的大致开始时间暂存在 `data/raw/start_time.jsonl`，每行格式如下：

```json
{"video_id":"bar_001","video_path":"data/raw/videos/bar_sample.mp4","clip_start_sec":65.0,"chart_type_hint":"bar"}
```

`bar-dominant` 按 `video_id == sample_id` 读取 `clip_start_sec`，并复用 `processed_dir` 中已有的 2 FPS 粗帧：

1. 初始搜索窗为 `[max(0, clip_start_sec - 2s), clip_start_sec + 20s]`。
2. 只对窗口内的帧做第一阶段 Qwen 识别，不再默认扫描整条视频。
3. 如果最后一个 positive 距当前窗口右边界不超过 1 秒，则将右边界向后扩展 10 秒并增量识别新帧。
4. 重复扩展，直到右边界附近出现稳定 negative，或到达视频末尾。
5. 找不到对应 `video_id` 时保留兼容行为，回退为全视频扫描。

搜索范围、扩展次数、扫描帧数、positive 帧数和 `no_positive_in_search_window` 状态会写入 `bar_dominant_report.json` 的 `target_search` 字段。`qwen_bar_frame_results.jsonl` 只保存本次实际搜索范围内的帧结果。

### 3.2 两阶段 Qwen 识别

模型加载细节：

- 模型路径从 `MODEL_PATH` 读取。
- 使用 `Qwen2_5_VLForConditionalGeneration` 和 `AutoProcessor`。
- `local_files_only=True`。
- processor 限制 `min_pixels=224*224`、`max_pixels=768*768`。
- dtype：CUDA 支持 BF16 时用 `torch.bfloat16`，否则用 `torch.float16`。
- `device_map="auto"`。
- 生成时 `do_sample=False`。

第一阶段调用粒度：

- `max_frames_per_call=3`，即每次把 1-3 张连续粗采样帧一起喂给 Qwen。
- 当前实现会把同一次调用得到的同一个 JSON 结果写回这个 group 里的每一帧。
- 输出：`generated_dir/qwen_bar_frame_results.jsonl`

第一阶段帧级 prompt 是 `BAR_DOMINANT_FRAME_PROMPT`。它只判断当前帧组是否属于目标 chart state，返回最小字段：

```text
You are detecting whether 1-3 consecutive video frames belong to the target class: bar-chart-dominant clip.

Target:
Positive only when the main narrative unit is expressed by bar marks. Bar marks include vertical bars, horizontal bars, stacked/grouped bars, bar-race bars, or bar-like marks whose length/position/order encodes quantities or categories.

Negative:
- only a short or incidental bar-like shape;
- circle, bubble, distance line, map, icon, illustration, vehicle, photo, or decorative motion;
- UI/progress bars without clear data encoding;
- mixed scenes where the current frames are not mainly bar-mark driven.

If a video segment briefly has bars and then switches to other marks, only the bar-dominant subsegment should be positive.

Return strict JSON only:
{
  "is_bar_chart_dominant_candidate": boolean,
  "bar_marks_visible": boolean,
  "bar_marks_dominant": boolean,
  "has_data_encoding_evidence": boolean,
  "scene_state": string,
  "confidence": number,
  "reason": string
}

scene_state must be one of: non_chart, chart_entering, stable_chart, chart_animating, chart_leaving, transition, uncertain.
```

结果会被规范化：

- 如果 `scene_state` 不在允许集合里，改成 `uncertain`。
- 如果没有同时满足 `bar_marks_visible`、`bar_marks_dominant`、`has_data_encoding_evidence`，强制 `is_bar_chart_dominant_candidate=false`。
- reason 中出现 `circle`、`bubble`、`map`、`icon`、`illustration`、`vehicle`、`distance line`、`decorative`、`not bar` 且 `bar_marks_dominant=false` 时，也强制为 negative。

第一阶段形成连续 positive candidate 后，第二阶段只在每个 candidate 的 `source_start/source_end` 内均匀选取最多 3 帧，调用 `BAR_IDENTITY_PROMPT` 补充：

- `chart_identity`
- `chart_title`
- `axis_labels`
- `category_labels`
- `chart_types`
- `mark_types`
- `animation_cue`

这些身份字段用于后续 candidate 合并阻断判断，避免让全窗口的每一帧都生成较长 JSON。

## 4. 正样本帧合并成 bar candidates

代码路径：`src/datavideo/bar_dominant.py::select_bar_candidates()`

一帧被认为 positive 必须同时满足：

```python
is_bar_chart_dominant_candidate
bar_marks_visible
bar_marks_dominant
has_data_encoding_evidence
```

合并逻辑：

1. 按 timestamp 排序 positive frames。
2. 相邻 positive frame 时间差 `<= max_gap` 就归到同一个 candidate。
3. `select_bar_candidates()` 默认参数是：
   - `max_gap=1.0` 秒
   - `expand=0.5` 秒
   - `min_duration=1.0` 秒
4. candidate 起止：
   - `start = first_positive_timestamp - 0.5`
   - `end = last_positive_timestamp + 0.5`
   - 不足 1 秒时向两边 pad。
5. 每个 candidate 先聚合第一阶段结果，再由第二阶段补充身份信息：
   - 平均 confidence
   - positive_frame_count
   - scene_states
   - animation_cues
   - chart_identity / title / axis labels / category labels / chart_types / mark_types

输出：`generated_dir/bar_candidates.jsonl`

## 5. candidates 再合并成 merged clips

代码路径：`src/datavideo/bar_dominant.py::merge_bar_candidates()`

合并逻辑：

- candidates 按 start 排序。
- 如果当前 candidate 与上一个 merged clip 的 gap `> 2.0` 秒，开一个新的 merged clip。
- 如果 gap `<= 2.0` 秒，会先判断是否应该阻断合并。

阻断合并的原因包括：

- `different_chart_identity`
- `title_changed`
- `axis_labels_changed`
- `category_set_changed`
- `chart_type_changed`
- `bar_orientation_changed`
- `scene_break_state`，即 scene state 包含 `chart_leaving`、`transition`、`non_chart`、`uncertain`
- `global_scene_change_cue`，即 animation cue 里有 scene / cut / transition

如果没有阻断，则合并为同一个 `bar_merged_xxx`，并记录 `merge_reasons`。

## 6. merged clip 的 Qwen contact sheet 复核

代码路径：`src/datavideo/bar_dominant.py::run_bar_dominant_pipeline()`

对每个 `bar_merged_xxx`：

1. 在临时目录中按 **2 FPS** 再抽该片段的 review frames。
2. 生成 contact sheet：
   - `max_cols=4`
   - `thumb_width=360`
   - 每格显示 timestamp、confidence、前几个 animation cues。
3. 把 contact sheet 作为单张图片输入 Qwen 复核。

复核 prompt 是 `BAR_DOMINANT_CLIP_REVIEW_PROMPT`，核心内容如下：

```text
You are reviewing a contact sheet from one merged candidate segment.

Question:
Is this a complete bar-chart-dominant data-video clip: a coherent semantic process where the main visual data message is expressed by bar marks?

Keep only if:
- bar marks are the dominant visual encoding for the candidate, not incidental;
- bar length/position/order/labels encode data;
- the segment is not a mixed segment where bars appear briefly and then the main narrative switches to circle, bubble, distance line, map, icon, illustration, or decorative motion;
- the segment forms a coherent data narrative process.

Return strict JSON only:
{
  "is_complete_bar_dominant_clip": boolean,
  "bar_marks_dominant": boolean,
  "has_data_encoding_evidence": boolean,
  "coherent_visual_data_message": boolean,
  "mixed_with_non_bar_marks": boolean,
  "suggested_start": number,
  "suggested_end": number,
  "chart_types": string[],
  "mark_types": string[],
  "decision": string,
  "confidence": number,
  "reason": string
}

decision must be one of: keep, trim, exclude, uncertain.
If the segment is mixed but contains a clear bar-dominant subclip, use decision="trim" and set suggested_start/suggested_end relative to the original video seconds.
```

复核结果规范化：

- decision 只允许 `keep`、`trim`、`exclude`、`uncertain`。
- 如果不是同时满足 `bar_marks_dominant`、`has_data_encoding_evidence`、`coherent_visual_data_message`：
  - confidence `>=0.4` 时改为 `exclude`
  - 否则改为 `uncertain`
- 如果 `mixed_with_non_bar_marks=true` 但 decision 是 `keep`，改为 `trim`。

落盘规则：

- decision 为 `keep` 或 `trim` 的片段进入最终机器候选。
- `trim` 会用 `suggested_start` / `suggested_end` 修改片段边界，但必须落在原片段内部。
- 最终 clip id 改写为 `bar_final_000`、`bar_final_001` ...
- 输出：
  - `generated_dir/bar_merged_clips.jsonl`
  - `generated_dir/final_bar_clips.jsonl`

## 7. bar-assets：生成审核页需要的每个 clip 资产

入口：`python -m datavideo.cli bar-assets --config ...`

代码路径：`src/datavideo/bar_assets.py::run_bar_assets_pipeline()`

输入：`generated_dir/final_bar_clips.jsonl`

对每个 `bar_final_xxx` 建目录：

```text
generated_dir/clips/bar_final_xxx/
```

### 7.1 截取 clip.mp4

如果 `final_bar_clips.jsonl` 里已有 `clip_mp4` 且文件存在，优先复制；否则从 `processed_dir/normalized.mp4` 按 `start/end` 重新截取。

截取命令逻辑：

```text
ffmpeg -y -ss <start> -i normalized.mp4 -t <end-start> -c copy clip.mp4
```

输出：`generated_dir/clips/bar_final_xxx/clip.mp4`

### 7.2 选择 initial keyframe

代码路径：`src/datavideo/keyframes.py::extract_scored_keyframe()`

采样：

- 在 clip 的 `start/end` 内抽候选帧。
- FPS 来自 `keyframes.sample_fps`；如果没配，就用 `sampling.fine_fps`。
- 函数会 clamp 到 4-8 FPS，所以当前配置下是 **8 FPS**。
- 短边仍缩放到 768。
- 输出候选帧：`keyframes/candidate_frames/keyframe_candidate_%06d.jpg`

每帧会计算两类分数：

1. 图像 motion score：相邻候选帧转灰度、resize 到 96x96，做差分，均值除以 255，得到 0-1 的 motion 分。
2. Qwen keyframe score：用 `score_keyframe_candidate()` 判断这帧是否是同一个图表、是否在主要数据变化之前、是否完整呈现 initial chart state。motion 只作为辅助项，不再优先选择“全局最静止”的后变化帧。

关键帧 prompt 是 `KEYFRAME_SCORE_PROMPT`，其中 `__CHART_IDENTITY__` 会替换成从 clip 聚合出的 chart identity / title / axis labels / category labels / chart types / mark types：

```text
You are scoring one video frame as a seed keyframe for a data-video clip.

The goal is NOT to choose the most visually stable frame overall.
The goal is to choose a good INITIAL seed keyframe: a frame before the main data-change animation, where the target chart is still complete and representative.

Target chart identity:
__CHART_IDENTITY__

A good seed keyframe must satisfy these conditions:
1. It belongs to the same target chart.
2. It is before the main data-change animation.
3. It shows the complete initial chart state.
4. All target categories, labels, bars/marks, axes, and title that define the chart are visible.
5. It is not a later stable state after some categories/bars have disappeared, changed, moved away, or been replaced.
6. It is not a scene cut, next chart, zoomed/cropped view, transition frame, or partial chart.

Important rule:
A later frame can be very stable and clear, but if it is after the main change, it is a bad seed keyframe.
A slightly less stable frame before the change is better than a perfectly stable frame after the change.

Reject or strongly penalize frames where:
- any target category is missing or cropped;
- some bars/marks have already changed to a new state;
- the chart is already in the result/final state rather than the initial state;
- the frame belongs to the next scene or a different chart;
- the chart identity has changed, including title, axis labels, category set, chart type, or main visual encoding.

Score this frame using the following fields.

Return strict JSON only:
{
  "same_chart": boolean,
  "scene_change": boolean,
  "pre_change": boolean,
  "post_change_state": boolean,
  "complete_initial_chart": boolean,
  "all_target_categories_visible": boolean,
  "completeness": number,
  "staticness": number,
  "chart_identity_consistency": number,
  "initial_state_representative": number,
  "motion_score": number,
  "reason": string
}
```

综合分：

```text
combined_score =
  1.5 * initial_state_representative
  + 1.2 * completeness
  + 1.0 * chart_identity_consistency
  + 0.5 * staticness
  + 0.5 if pre_change else -2.0
  + 0.5 if all_target_categories_visible else -2.0
  - 1.0 * motion_score
  - 2.0 if post_change_state else 0
  - 2.0 if scene_change else 0
```

选择排序优先级：

1. `same_chart=true`
2. `scene_change=false`
3. `pre_change=true`
4. `post_change_state=false`
5. `complete_initial_chart=true`
6. `all_target_categories_visible=true`
7. `initial_state_representative` 更高
8. combined_score 更高
9. motion score 更低
10. timestamp 更早

这里最核心的是：`pre_change` 是强条件，`motion_score` 只做辅助排序，不能主导最终关键帧。

输出：

- `keyframes/initial.png`
- `keyframes/keyframe_scores.jsonl`
- `keyframes/keyframe_manifest.json`

如果 Qwen 不可用，会退化为启发式关键帧分数；但检测阶段没有可靠 fallback，Qwen 不可用通常不会产生 positive clips。

### 7.3 initial frame 转 SVG

代码路径：`src/datavideo/semantic.py::build_semantic_svg()`

工具：`vtracer`

输入：`keyframes/initial.png`

参数：

- `colormode="color"`
- `hierarchical="stacked"`
- `mode="spline"`
- `filter_speckle=4`
- `color_precision=6`
- `layer_difference=16`
- `corner_threshold=60`
- `length_threshold=4.0`
- `max_iterations=10`
- `splice_threshold=45`
- `path_precision=8`

随后用 `cairosvg.svg2png()` 渲染 preview。

输出：

- `semantic.svg`
- `semantic_preview.png`
- `svg_report.json`

### 7.4 从 initial frame 恢复柱状图数据

代码路径：`src/datavideo/chart_data.py::recover_chart_data()`

模型：同一个 Qwen VL client。

数据恢复 prompt 是 `DATA_PROMPT`：

```text
Recover the data from the bar chart keyframe. Return only strict JSON.
Use null for values or labels that cannot be read reliably. Do not invent values.
Schema: {"title": null, "x_axis": null, "y_axis": null, "unit": null, "bars": [{"label": null, "value": null}], "uncertain_fields": [], "notes": ""}
```

输出：

- `chart_data_raw.json`: 原始模型响应、模型状态、失败原因、prompt version
- `chart_data.csv`: 审核页可编辑表格，列为 `index,label,value`
- `chart_metadata.json`: title、x_axis、y_axis、unit、model_status、failure_reason
- `chart_data_validation.json`: schema 是否有效、不确定字段、value_count、是否需要人工审核

注意：模型被明确要求不能编造读不清的值，读不出时应写 null。审核页就是用来改这些值的。

### 7.5 run_report

`bar-assets` 最后写：

```text
generated_dir/run_report.json
generated_dir/refined_clips.jsonl
```

`run_report.json` 是 Streamlit 审核页判断 pipeline 是否已跑完的入口文件。

## 8. Streamlit 审核页

入口：

```bash
streamlit run app/review_app.py --server.address 127.0.0.1 --server.port 8501
```

配置来源：

- 侧边栏 `Config` 输入框。
- 默认使用环境变量 `DATAVIDEO_REVIEW_CONFIG`。
- 如果没有环境变量，默认 `configs/stage1_bar.yaml`。

审核页读取：

- `generated_dir/run_report.json`：不存在则提示先跑 `bar-assets`
- 优先读取 `generated_dir/final_bar_clips.jsonl`
- 若不存在，再读 `generated_dir/refined_clips.jsonl`
- 每个 clip 的资产目录：`generated_dir/clips/<clip_id>/`

页面内容：

1. 左侧播放 `clip.mp4`。
2. 可编辑 `Start` / `End`，step 为 0.1 秒，显示 3 位小数。
3. 右侧显示 `svg_report.json`。
4. 展示 `keyframes/initial.png`。
5. 对比 `initial.png` 与 `semantic_preview.png`。
6. 读取 `chart_data.csv`，用 `st.data_editor` 让人工修改 `label/value` 行。
7. expander 中显示 `chart_metadata.json` 和 `chart_data_validation.json`。
8. 审核决策：
   - `通过`
   - `需要修改`
   - `排除`
   - `保存`
9. reviewer 默认 `local`，notes 自由填写。

提交审核时写入 SQLite：

- 表：`reviews`
- stage：`stage1_review`
- original_value：原始 clip 和原始 chart_data
- reviewed_value：
  - `clip_id`
  - 修改后的 `clip.start/end`
  - 修改后的 `chart_data`
  - keyframe 名称 `initial.png`
- reviewer、notes、model_version、config_hash、reviewed_at

同时写一个便捷文件：

```text
reviewed_dir/latest_review.json
```

## 9. 审核后重建 reviewed 输出

提交按钮会立即调用：

```python
apply_latest_reviews(cfg)
```

也可以在页面点 `Rebuild Reviewed From Latest Reviews`。

代码路径：`src/datavideo/reviewed_outputs.py`

输入：

- `generated_dir/final_bar_clips.jsonl`
- SQLite 中当前 sample_id、stage=`stage1_review` 的最新审核记录

决策解释：

- 接受：`通过`、`需要修改`、`保存`
- 排除：`排除`
- 其他 decision：进入 unreviewed
- 没有审核记录：进入 unreviewed

对接受的 clip：

1. 建目录：`reviewed_dir/clips/<clip_id>/`
2. 从 generated clip 目录复制：
   - `keyframes/`
   - `semantic.svg`
   - `semantic_preview.png`
   - `svg_report.json`
   - `chart_data_raw.json`
   - `chart_metadata.json`
   - `chart_data_validation.json`
3. 如果人工改了 start/end，就从 `normalized.mp4` 重新截 clip；否则复制已有 `clip.mp4`。
4. 写人工修正后的 `chart_data.csv`。
5. 写：
   - `review.json`
   - `clip.json`

最终 reviewed 输出：

```text
reviewed_dir/final_bar_clips.jsonl
reviewed_dir/excluded_clips.jsonl
reviewed_dir/unreviewed_clips.jsonl
reviewed_dir/reviewed_report.json
reviewed_dir/clips/<clip_id>/
```

## 10. 旧流程：仍在代码中，但不是当前主线

代码里还有两个相关流程，理解历史输出时会遇到：

### `stage1`

入口：`python -m datavideo.cli stage1 --config ...`

它会：

1. 标准化视频。
2. 2 FPS 粗抽帧。
3. 用 `CHART_PROMPT` 判断画面是否包含 bar chart。
4. 合并 positive frames。
5. 对每个候选片段 8 FPS 细抽帧。
6. 再次模型识别并 refine 边界。
7. 截 clip、选 keyframe、生成 semantic SVG、恢复数据。

这个流程更偏“chart visible”而不是当前严格的 bar-dominant data-video semantic process。

### `detect`

入口：`python -m datavideo.cli detect --config ...`

它用 `DATA_VIDEO_CLIP_PROMPT` 做更宽松的 data-video candidate 检测，高召回地保留图表出现、增长、强调、排序或离开的片段。它会在 `reviewed_dir/candidates/` 下生成候选 mp4、8 FPS contact sheet 和 JSON，主要用于早期 candidate review，不是现在 Streamlit 审核页的主输入。

通用 data-video prompt 重点是：

- Positive 包括柱状图、水平柱状图、bar-race、类似进度条但实际表达数据的条形 mark，以及出现、进入、增长、缩短、强调、排序或离开的动画过程。
- Negative 包括纯人物、普通字幕、装饰图形、不表达数据的进度条或 UI、普通地图、照片、插图。
- 不确定但存在数据 mark 动画证据时优先保留 candidate，低 confidence。

## 11. 从文件看一个完整样本应该有什么

以 `bar_002` 为例，一次完整机器生成后大致有：

```text
data/processed/bar_002/
  normalized.mp4
  audio_16k_mono.wav
  standardization_report.json
  frames/coarse_2fps/

data/generated/bar_002/
  frame_manifest.jsonl
  qwen_bar_frame_results.jsonl
  bar_candidates.jsonl
  bar_merged_clips.jsonl
  final_bar_clips.jsonl
  run_report.json
  refined_clips.jsonl
  clips/bar_final_000/
    clip.mp4
    keyframes/initial.png
    keyframes/keyframe_manifest.json
    keyframes/keyframe_scores.jsonl
    semantic.svg
    semantic_preview.png
    svg_report.json
    chart_data_raw.json
    chart_data.csv
    chart_metadata.json
    chart_data_validation.json
    clip_report.json

data/reviewed/bar_002/
  latest_review.json
  final_bar_clips.jsonl
  excluded_clips.jsonl
  unreviewed_clips.jsonl
  reviewed_report.json
  clips/bar_final_xxx/
```

## 12. 当前流程的几个重要注意点

- `bar-dominant` 依赖已有粗抽帧。如果从全新 raw video 开始，先跑 `stage0`，再确保生成 `coarse_2fps` 和 `frame_manifest.jsonl`。实际可通过旧 `stage1`/`detect` 产生这些粗帧，或补一个专门的抽帧命令。
- `bar-dominant` 的 `select_bar_candidates()` 当前没有直接使用 YAML 里的 `detection.min_confidence`，而是依赖 Qwen 返回的 boolean 字段组合。
- `bar-dominant` 内部复核 contact sheet 使用临时目录；最终保留的是 `final_bar_clips.jsonl` 以及 `bar-assets` 生成的 `generated_dir/clips/<clip_id>/`。
- `bar-assets` 才是审核页资产生成步骤；只跑 `bar-dominant` 不足以打开完整审核页。
- 审核页保存的是最新人工修订；最终训练/交付应优先使用 `data/reviewed/<sample_id>/`，不要直接拿未审核的 `data/generated/<sample_id>/`。
