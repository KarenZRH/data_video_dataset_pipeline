from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .schemas import ensure_dir, write_jsonl


def scale_filter(short_side: int) -> str:
    return f"scale='if(gt(iw,ih),-2,{short_side})':'if(gt(iw,ih),{short_side},-2)'"


def extract_frames(
    video: str | Path,
    out_dir: str | Path,
    fps: float,
    short_side: int,
    prefix: str,
    force: bool = False,
    start: float | None = None,
    end: float | None = None,
) -> list[dict[str, Any]]:
    out_dir = ensure_dir(out_dir)
    pattern = out_dir / f"{prefix}_%06d.jpg"
    existing = sorted(out_dir.glob(f"{prefix}_*.jpg"))
    if force or not existing:
        cmd = ["ffmpeg", "-y"]
        if start is not None:
            cmd += ["-ss", f"{start:.3f}"]
        cmd += ["-i", str(video)]
        if end is not None and start is not None:
            cmd += ["-t", f"{max(0.1, end - start):.3f}"]
        cmd += ["-vf", f"fps={fps},{scale_filter(short_side)}", "-q:v", "2", str(pattern)]
        subprocess.run(cmd, check=True)
    rows = []
    for idx, path in enumerate(sorted(out_dir.glob(f"{prefix}_*.jpg")), start=1):
        base_time = 0.0 if start is None else start
        rows.append(
            {
                "frame_id": path.stem,
                "path": str(path),
                "timestamp": base_time + (idx - 1) / fps,
                "fps": fps,
                "sample_type": prefix,
            }
        )
    return rows


def write_contact_sheet(
    frames: list[dict[str, Any]],
    out: str | Path,
    max_cols: int = 4,
    thumb_width: int = 320,
    scores: dict[str, float] | None = None,
    labels: dict[str, str] | None = None,
) -> Path:
    out = Path(out)
    ensure_dir(out.parent)
    if not frames:
        Image.new("RGB", (thumb_width, thumb_width), "white").save(out)
        return out
    thumbs = []
    for row in frames:
        im = Image.open(row["path"]).convert("RGB")
        ratio = thumb_width / im.width
        thumb = im.resize((thumb_width, max(1, int(im.height * ratio))))
        label_h = 54 if labels else 34
        canvas = Image.new("RGB", (thumb.width, thumb.height + label_h), "white")
        canvas.paste(thumb, (0, 0))
        draw = ImageDraw.Draw(canvas)
        conf = ""
        if scores and row["frame_id"] in scores:
            conf = f" conf={scores[row['frame_id']]:.2f}"
        draw.text((6, thumb.height + 8), f"{row['timestamp']:.2f}s{conf}", fill=(0, 0, 0))
        if labels and row["frame_id"] in labels:
            draw.text((6, thumb.height + 28), labels[row["frame_id"]][:80], fill=(0, 0, 0))
        thumbs.append(canvas)
    cols = min(max_cols, len(thumbs))
    rows = math.ceil(len(thumbs) / cols)
    cell_h = max(im.height for im in thumbs)
    sheet = Image.new("RGB", (cols * thumb_width, rows * cell_h), "white")
    for idx, im in enumerate(thumbs):
        sheet.paste(im, ((idx % cols) * thumb_width, (idx // cols) * cell_h))
    sheet.save(out, quality=90)
    return out


def write_frame_manifest(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    return write_jsonl(path, rows)
