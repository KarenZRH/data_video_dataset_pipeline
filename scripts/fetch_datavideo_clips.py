#!/usr/bin/env python3
"""Fetch clip metadata and videos from the Data Videos study website."""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


DEFAULT_SITE_URL = "https://datavideos.github.io/narration-animation-study/"
DEFAULT_BUNDLE_RE = re.compile(r'src="\./(static/js/main\.[^"]+\.chunk\.js)"')
CLIP_MODULE_RE = re.compile(
    r"60:function\(e\)\{e\.exports=JSON\.parse\('([\s\S]*?)'\)\}"
)
CHART_ORDER = [
    "map",
    "bar",
    "line",
    "donut",
    "area",
    "pictograph",
    "pie",
    "timeline",
    "treemap",
    "scatter",
    "sankey",
    "combined",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Data Videos clip metadata from the website and optionally "
            "download the corresponding YouTube clips."
        )
    )
    parser.add_argument("--site-url", default=DEFAULT_SITE_URL)
    parser.add_argument("--clips-per-chart", type=int, default=2)
    parser.add_argument("--jsonl", type=Path, default=Path("data/raw/datavideo_clips.jsonl"))
    parser.add_argument("--video-dir", type=Path, default=Path("data/raw/videos"))
    parser.add_argument("--cookies", type=Path, default=Path("www.youtube.com_cookies.txt"))
    parser.add_argument("--proxy", default="http://127.0.0.1:<port>")
    parser.add_argument("--max-height", type=int, default=720)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify", action="store_true", default=True)
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    parser.add_argument("--log", type=Path, default=Path("logs/download_datavideo_clips.log"))
    return parser.parse_args()


def install_proxy(proxy: str | None) -> None:
    if not proxy:
        return
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    )
    urllib.request.install_opener(opener)


def read_url(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def find_bundle_url(site_url: str) -> str:
    html = read_url(site_url)
    match = DEFAULT_BUNDLE_RE.search(html)
    if not match:
        raise RuntimeError("Could not find main React bundle URL in site HTML")
    return urllib.request.urljoin(site_url, match.group(1))


def parse_clip_data(bundle_text: str) -> list[dict]:
    match = CLIP_MODULE_RE.search(bundle_text)
    if not match:
        raise RuntimeError("Could not find clip JSON module in React bundle")
    js_string_payload = match.group(1)
    json_text = ast.literal_eval("'" + js_string_payload + "'")
    clips = json.loads(json_text)
    if not isinstance(clips, list):
        raise RuntimeError("Clip module did not decode to a list")
    return clips


def time_to_seconds(value: str) -> int:
    parts = [int(part) for part in value.split(":")]
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def seconds_to_hms(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def build_rows(clips: list[dict], clips_per_chart: int, video_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for chart in CHART_ORDER:
        chart_clips = [clip for clip in clips if clip.get("chart") == chart]
        for index, clip in enumerate(chart_clips[:clips_per_chart], start=1):
            start_seconds = time_to_seconds(clip["start"])
            end_seconds = time_to_seconds(clip["end"])
            output_stem = f"{chart}_{index}"
            rows.append(
                {
                    "chart_type": chart,
                    "chart_index": index,
                    "raw_video_title": clip["title"],
                    "video_id": clip["id"],
                    "youtube_url": f"https://www.youtube.com/watch?v={clip['id']}",
                    "start_time": clip["start"],
                    "end_time": clip["end"],
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "duration_seconds": end_seconds - start_seconds,
                    "channel": clip["channel"],
                    "year": clip["year"],
                    "output_stem": output_stem,
                    "output_path": str(video_dir / f"{output_stem}.mp4"),
                }
            )
    return rows


def write_jsonl(rows: list[dict], jsonl_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    jsonl_path.write_text(text, encoding="utf-8")


def read_jsonl(jsonl_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def download_row(
    row: dict,
    cookies: Path,
    proxy: str | None,
    max_height: int,
    overwrite: bool,
    log_file,
) -> tuple[str, bool]:
    output_path = Path(row["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        return row["output_stem"], True

    section = f"*{seconds_to_hms(row['start_seconds'])}-{seconds_to_hms(row['end_seconds'])}"
    format_selector = (
        f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/"
        f"b[height<={max_height}][ext=mp4]/"
        f"bv*[height<={max_height}]+ba/b[height<={max_height}]"
    )
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--cookies",
        str(cookies),
        "--js-runtimes",
        "node",
        "--download-sections",
        section,
        "--force-keyframes-at-cuts",
        "-f",
        format_selector,
        "--merge-output-format",
        "mp4",
        "-o",
        str(output_path.with_suffix(".%(ext)s")),
        row["youtube_url"],
    ]
    if proxy:
        cmd[3:3] = ["--proxy", proxy]
    if overwrite:
        cmd.insert(-1, "--force-overwrites")

    print(f"downloading {row['output_stem']} {section} {row['youtube_url']}", flush=True)
    log_file.write("\n" + "=" * 80 + "\n")
    log_file.write(f"{row['output_stem']} {row['youtube_url']} {section}\n")
    log_file.flush()
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_file.write(result.stdout)
    log_file.flush()
    ok = result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
    return row["output_stem"], ok


def download_rows(
    rows: list[dict],
    cookies: Path,
    proxy: str | None,
    max_height: int,
    overwrite: bool,
    log_path: Path,
) -> list[str]:
    if not cookies.exists():
        raise FileNotFoundError(f"Cookie file not found: {cookies}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        for index, row in enumerate(rows, start=1):
            print(f"[{index}/{len(rows)}]", end=" ", flush=True)
            name, ok = download_row(row, cookies, proxy, max_height, overwrite, log_file)
            if ok:
                print(f"ok {name}", flush=True)
            else:
                print(f"FAILED {name}", flush=True)
                failed.append(name)
    return failed


def probe_duration(path: Path) -> float:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        text=True,
    ).strip()
    return float(output)


def verify_rows(rows: list[dict]) -> tuple[list[str], list[tuple[str, float, float]]]:
    missing: list[str] = []
    mismatches: list[tuple[str, float, float]] = []
    for row in rows:
        path = Path(row["output_path"])
        if not path.exists():
            missing.append(row["output_stem"])
            continue
        duration = probe_duration(path)
        expected = float(row["duration_seconds"])
        if abs(duration - expected) > 0.15:
            mismatches.append((row["output_stem"], duration, expected))
    return missing, mismatches


def require_download_tools(metadata_only: bool) -> None:
    if metadata_only:
        return
    if shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg and ffprobe must be available on PATH")
    result = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise RuntimeError("yt-dlp is not installed for this Python interpreter")


def main() -> int:
    args = parse_args()
    install_proxy(args.proxy)
    require_download_tools(args.metadata_only)

    if args.download_only:
        rows = read_jsonl(args.jsonl)
    else:
        bundle_url = find_bundle_url(args.site_url)
        print(f"bundle: {bundle_url}")
        clips = parse_clip_data(read_url(bundle_url))
        rows = build_rows(clips, args.clips_per_chart, args.video_dir)
        write_jsonl(rows, args.jsonl)
        counts = {chart: sum(row["chart_type"] == chart for row in rows) for chart in CHART_ORDER}
        print(f"source clips: {len(clips)}")
        print(f"wrote {len(rows)} rows to {args.jsonl}")
        print("counts: " + json.dumps(counts, ensure_ascii=False))

    if args.metadata_only:
        return 0

    failed = download_rows(
        rows=rows,
        cookies=args.cookies,
        proxy=args.proxy,
        max_height=args.max_height,
        overwrite=args.overwrite,
        log_path=args.log,
    )
    if failed:
        print(f"download failures: {failed}", file=sys.stderr)
        return 1

    if args.verify:
        missing, mismatches = verify_rows(rows)
        print(f"verified files: {len(rows) - len(missing)} / {len(rows)}")
        if missing or mismatches:
            print(f"missing: {missing}", file=sys.stderr)
            print(f"duration mismatches: {mismatches}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
