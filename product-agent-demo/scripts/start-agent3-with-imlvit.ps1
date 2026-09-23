param(
    [string]$ImlVitWorkerUrl = "",
    [string]$AideWorkerUrl = "",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "未找到项目虚拟环境：$python"
}

$env:AGENT3_MODEL_TIMEOUT_SECONDS = "30"
if ($ImlVitWorkerUrl) {
    $env:IML_VIT_WORKER_URL = $ImlVitWorkerUrl
    $env:IML_VIT_MODEL_VERSION = "iml-vit-v1"
} else {
    Remove-Item Env:IML_VIT_WORKER_URL -ErrorAction SilentlyContinue
}
if ($AideWorkerUrl) {
    $env:AIDE_WORKER_URL = $AideWorkerUrl
    $env:AIDE_MODEL_VERSION = "aide-genimage-v1"
} else {
    Remove-Item Env:AIDE_WORKER_URL -ErrorAction SilentlyContinue
}

Set-Location $projectRoot
& $python -m uvicorn app.main:app --reload --port $Port
