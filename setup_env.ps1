# DataVideo pipeline environment setup for PowerShell.
# Run once per new terminal:
#   . .\setup_env.ps1
#
# Edit these paths for your machine before use.

$RepoRoot = $PSScriptRoot

# Python environment
$env:PYTHONPATH = "$RepoRoot\src"
$VenvPython = "$RepoRoot\.venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    $VenvPython = "python"
}

# Optional: uncomment if ffmpeg is not already on PATH.
# $env:PATH = "C:\path\to\ffmpeg\bin;" + $env:PATH

# Models used by the independent model configs.
# $env:MODEL_3B_PATH = "C:\path\to\Qwen2.5-VL-3B-Instruct"
# $env:MODEL_7B_PATH = "C:\path\to\Qwen2.5-VL-7B-Instruct"
# $env:WHISPER_MODEL_PATH = "C:\path\to\faster-whisper-model"

# Hugging Face cache and network options
# $env:HF_HOME = "D:\hf_cache"
# $env:HF_HUB_DISABLE_XET = "1"
# $env:HTTP_PROXY  = "http://127.0.0.1:7890"
# $env:HTTPS_PROXY = "http://127.0.0.1:7890"

Write-Host "PYTHONPATH   = $env:PYTHONPATH"
Write-Host "MODEL_3B     = $env:MODEL_3B_PATH"
Write-Host "MODEL_7B     = $env:MODEL_7B_PATH"
Write-Host "WHISPER_PATH = $env:WHISPER_MODEL_PATH"
Write-Host "Python       = $VenvPython"

function Invoke-DataVideo {
    param(
        [Parameter(Mandatory=$true)][string]$Stage,
        [Parameter(Mandatory=$true)][string]$Config
    )
    & $VenvPython -m datavideo.cli $Stage --config $Config @args
}

function Run-Context7B { Invoke-DataVideo context "$RepoRoot\configs\multichart_assets_qwen7b.yaml" @args }
function Run-Asr7B { Invoke-DataVideo asr "$RepoRoot\configs\multichart_assets_qwen7b.yaml" @args }
function Run-Assets7B { Invoke-DataVideo assets "$RepoRoot\configs\multichart_assets_qwen7b.yaml" @args }
function Run-Quality7B { Invoke-DataVideo quality "$RepoRoot\configs\multichart_assets_qwen7b.yaml" @args }
function Run-Review7B { Invoke-DataVideo reviewed "$RepoRoot\configs\multichart_assets_qwen7b.yaml" @args }

function Run-Context3B { Invoke-DataVideo context "$RepoRoot\configs\multichart_assets_qwen3b.yaml" @args }
function Run-Asr3B { Invoke-DataVideo asr "$RepoRoot\configs\multichart_assets_qwen3b.yaml" @args }
function Run-Assets3B { Invoke-DataVideo assets "$RepoRoot\configs\multichart_assets_qwen3b.yaml" @args }
function Run-Quality3B { Invoke-DataVideo quality "$RepoRoot\configs\multichart_assets_qwen3b.yaml" @args }
function Run-Review3B { Invoke-DataVideo reviewed "$RepoRoot\configs\multichart_assets_qwen3b.yaml" @args }
function Run-ReviewApp { & $VenvPython -m streamlit run "$RepoRoot\app\multichart_v2_review_app.py" @args }

Write-Host "Ready. Commands: Run-Context7B / Run-Assets7B / Run-Context3B / Run-Assets3B / Run-ReviewApp"
