# DataVideo pipeline environment setup (PowerShell)
# Run once per new terminal:  . .\setup_env.ps1
#
# Requirements before running:
#   - MODELS already downloaded to D:\dev\models
#   - YouTube cookies file at .\www.youtube.com_cookies.txt (for yt-dlp)
#   - Local proxy at 127.0.0.1:7897 (optional; only needed if downloads fail)

$RepoRoot = $PSScriptRoot

# Python venv
$env:PYTHONPATH = "$RepoRoot\src"
$VenvPython = "$RepoRoot\.venv\Scripts\python.exe"

# ffmpeg / ffprobe
$env:PATH = "D:\dev\ffmpeg\bin;" + $env:PATH

# Models
$env:MODEL_PATH = "D:\dev\models\Qwen2.5-VL-3B-Instruct"
$env:WHISPER_MODEL_PATH = "D:\dev\models\faster-whisper-small"

# 8GB VRAM: load Qwen in 4-bit
$env:DATAVIDEO_QUANTIZE_4BIT = "1"

# HuggingFace cache on D: (C: is nearly full)
$env:HF_HOME = "D:\dev\.hf_cache"
$env:HF_HUB_DISABLE_XET = "1"

# Proxy (only if direct network fails)
# $env:HTTP_PROXY  = "http://127.0.0.1:7897"
# $env:HTTPS_PROXY = "http://127.0.0.1:7897"

Write-Host "PYTHONPATH    = $env:PYTHONPATH"
Write-Host "MODEL_PATH    = $env:MODEL_PATH"
Write-Host "WHISPER_PATH  = $env:WHISPER_MODEL_PATH"
Write-Host "Python        = $VenvPython"

# Convenience aliases
function Run-Context { & $VenvPython -m datavideo.cli context --config "$RepoRoot\configs\multichart_assets_v2.yaml" @args }
function Run-Asr    { & $VenvPython -m datavideo.cli asr    --config "$RepoRoot\configs\multichart_assets_v2.yaml" @args }
function Run-Assets { & $VenvPython -m datavideo.cli assets --config "$RepoRoot\configs\multichart_assets_v2.yaml" @args }
function Run-Quality{ & $VenvPython -m datavideo.cli quality --config "$RepoRoot\configs\multichart_assets_v2.yaml" @args }
function Run-Review { & $VenvPython -m datavideo.cli reviewed --config "$RepoRoot\configs\multichart_assets_v2.yaml" @args }
function Run-ReviewApp { & $VenvPython -m streamlit run "$RepoRoot\app\multichart_v2_review_app.py" @args }

Write-Host "Ready. Commands: Run-Context / Run-Asr / Run-Assets / Run-Quality / Run-Review / Run-ReviewApp"
