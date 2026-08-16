# deploy_modal.ps1 — one-shot setup for the Modal GPU worker (video → SMPL).
#
# Run this ONCE from  c:\dan\scripts\video_to_smpl .  It walks you through the
# whole Modal setup so the "learn from any video" feature runs on a real GPU
# that scales to zero (you pay only per second of inference; $30/mo free credit).
#
#   cd c:\dan\scripts\video_to_smpl
#   .\deploy_modal.ps1
#
# Prereqs you must have ready BEFORE running:
#   1. A (free) Modal account            -> https://modal.com/signup
#   2. The license-gated GVHMR checkpoints + SMPL/SMPL-X body models in a local
#      .\checkpoints\ folder, laid out per GVHMR docs/INSTALL.md. These are free
#      but require academic registration; they are NOT redistributable so they
#      can't be baked into the repo. Layout expected inside .\checkpoints\:
#         checkpoints\  (GVHMR weights: gvhmr/, dpvo/, hmr2/, vitpose/, yolo/ ...)
#         body_models\  (smpl/, smplx/)
#
# What this does:
#   • installs the Modal CLI
#   • links your Modal account (opens a browser)
#   • creates a persistent Volume and uploads your checkpoints into it
#   • creates the shared-secret used to authenticate the backend -> Modal call
#   • deploys modal_app.py and prints the submit endpoint URL
#   • prints the exact 3 App Service settings to paste into studioos-api
# ---------------------------------------------------------------------------

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

Write-Host "`n=== Modal GPU worker setup ===`n" -ForegroundColor Cyan

# 1) Install the Modal CLI.
Write-Host "[1/6] Installing the Modal CLI..." -ForegroundColor Yellow
python -m pip install --upgrade modal | Out-Host

# 2) Link the Modal account (opens a browser; skips if already linked).
Write-Host "`n[2/6] Linking your Modal account (a browser window will open)..." -ForegroundColor Yellow
try { python -m modal token new } catch { Write-Host "  (already linked — continuing)" -ForegroundColor DarkGray }

# 3) Checkpoints sanity check.
Write-Host "`n[3/6] Checking for local GVHMR checkpoints..." -ForegroundColor Yellow
if (-not (Test-Path ".\checkpoints")) {
    Write-Host "  ! .\checkpoints not found." -ForegroundColor Red
    Write-Host "    Download the GVHMR checkpoints + SMPL/SMPL-X bodies (free academic" -ForegroundColor Red
    Write-Host "    registration, see GVHMR docs/INSTALL.md), place them under" -ForegroundColor Red
    Write-Host "    .\checkpoints\ , then re-run this script." -ForegroundColor Red
    exit 1
}

# 4) Create the Volume and upload the checkpoints (idempotent).
Write-Host "`n[4/6] Creating the 'gvhmr-checkpoints' Volume and uploading..." -ForegroundColor Yellow
try { python -m modal volume create gvhmr-checkpoints } catch { Write-Host "  (volume exists — continuing)" -ForegroundColor DarkGray }
python -m modal volume put gvhmr-checkpoints .\checkpoints /inputs --force | Out-Host

# 5) Create the shared secret the backend uses to authenticate.
Write-Host "`n[5/6] Creating the backend->Modal shared secret..." -ForegroundColor Yellow
$token = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 40 | ForEach-Object {[char]$_})
try {
    python -m modal secret create modal-submit-token "MODAL_SUBMIT_TOKEN=$token"
} catch {
    Write-Host "  Secret exists. To rotate it, delete + recreate:" -ForegroundColor DarkGray
    Write-Host "    python -m modal secret delete modal-submit-token" -ForegroundColor DarkGray
}

# 6) Deploy the app.
Write-Host "`n[6/6] Deploying modal_app.py ..." -ForegroundColor Yellow
$deployOut = python -m modal deploy modal_app.py 2>&1 | Tee-Object -Variable _tee
$deployOut | Out-Host

# Try to extract the submit endpoint URL from the deploy output.
$url = ($deployOut | Select-String -Pattern 'https://[^\s]*submit[^\s]*' | Select-Object -First 1).Matches.Value
if (-not $url) {
    $url = ($deployOut | Select-String -Pattern 'https://[^\s]*modal\.run[^\s]*' | Select-Object -First 1).Matches.Value
}

Write-Host "`n=== DONE ===" -ForegroundColor Green
Write-Host "`nNow set these 3 App Service settings on studioos-api (studio-os-rg):" -ForegroundColor Cyan
Write-Host "  GPU_PROVIDER        = modal"
if ($url) { Write-Host "  MODAL_ENDPOINT_URL  = $url" } else {
    Write-Host "  MODAL_ENDPOINT_URL  = <the submit URL printed above by 'modal deploy'>"
}
Write-Host "  MODAL_SUBMIT_TOKEN  = $token"
Write-Host "`nOne command to set them (edit the URL if not auto-detected):" -ForegroundColor Cyan
$sub = 'd8b4564f-81c5-4325-9c57-fcaef1a384fa'
$urlToken = if ($url) { $url } else { '<MODAL_ENDPOINT_URL>' }
Write-Host "  az webapp config appsettings set --name studioos-api --resource-group studio-os-rg ``"
Write-Host "    --subscription $sub ``"
Write-Host "    --settings GPU_PROVIDER=modal MODAL_ENDPOINT_URL=`"$urlToken`" MODAL_SUBMIT_TOKEN=`"$token`" -o none"
Write-Host "`nAfter that, a submitted video auto-runs on Modal's GPU and the coach" -ForegroundColor Green
Write-Host "teaches the choreography. Idle cost = `$0." -ForegroundColor Green
