#!/usr/bin/env bash
set -euo pipefail

echo "Project: $(pwd)"
if [ "${CONDA_DEFAULT_ENV:-}" = "DataVideo" ] && [ -n "${CONDA_PREFIX:-}" ]; then
    ENV_BIN="${CONDA_PREFIX}/bin"
else
    ENV_BIN="<conda-env>/bin"
fi
PYTHON_BIN="${ENV_BIN}/python"
GIT_BIN="${ENV_BIN}/git"
FFMPEG_BIN="${ENV_BIN}/ffmpeg"

if [ ! -x "$PYTHON_BIN" ]; then PYTHON_BIN="$(command -v python || true)"; fi
if [ ! -x "$GIT_BIN" ]; then GIT_BIN="$(command -v git || true)"; fi
if [ ! -x "$FFMPEG_BIN" ]; then FFMPEG_BIN="$(command -v ffmpeg || true)"; fi

echo "Python: $($PYTHON_BIN --version)"
echo "Git: $($GIT_BIN --version)"
echo "FFmpeg: $($FFMPEG_BIN -version | head -n 1)"

"$PYTHON_BIN" - <<'PY'
from pathlib import Path

required = [
    "data/raw",
    "configs",
    "docs",
    "scripts",
    "src/datavideo",
    "src/datavideo_multichart_v2",
    "app",
    "tests",
]

missing = [path for path in required if not Path(path).exists()]
if missing:
    raise SystemExit(f"Missing directories: {missing}")

print("Directory layout: ok")
PY
