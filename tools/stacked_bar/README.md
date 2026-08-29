# Stacked Bar Charts (模板 + 自动提取)

堆叠柱状图的两种数据来源，共用同一套数据表结构和渲染器：

## 1. 模板生成（精确样本，人工给数据）

```bash
python stacked_bar_template.py examples/bar75_demo.json out_dir
```

输入 JSON 支持两种格式：

- **A. `values`**：`values[bar][series] = 段值`，每根柱的高度 = 各段值之和，渲染时按最高柱归一化。
- **B. `bars`**：`bars[{label, rel_len, segments:[{name, fraction}]}]`，显式给出柱高（相对最高柱）+ 柱内各色段占比（和=1），与自动提取的输出格式一致。

输出三件套到 `out_dir/`：

- `data_table.csv`：扁平段行（`clip_id,bar,series,value,unit,series_order,stack_pos,value_type`）
- `semantic.svg`：堆叠渲染（按 stack_pos 自下而上叠色）
- `intent.json`：自动生成动画意图（主导介质变化）

## 2. 自动提取（从视频帧，半自动兜底）

```bash
python extract_stacked.py <关键帧.png> out_dir
```

流程：去掉帧内红色手绘标注（inpaint）→ vision 定性（图例/柱框/柱内构成）→ CV 测实际柱高（框内非背景像素跨度）→ 相对值（最长=1）→ 输出同一套三件套。

已知限制：vision 对细薄色段可能漏报、柱高偶有噪声，异常样本应标 `needs_review` 人工确认；需要精确数据的样本建议走模板路径。

环境变量（可选）：

- `DATAVIDEO_VISION_NODE`：node 可执行文件路径（默认 `node`）
- `DATAVIDEO_VISION_SCRIPT`：vision.js 路径（默认 `vision.js`）
