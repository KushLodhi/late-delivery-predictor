<#
.SYNOPSIS
    One-command Windows 11 setup, training, testing, and launch script for DelhiCold Logistics.
.EXAMPLE
    .\run.ps1                  # Trains on 100,000 rows, runs pytest, starts FastAPI + Streamlit
    .\run.ps1 -Rows 1500000    # Full 1.5M-row training benchmark
#>
param(
    [int]$Rows = 100000,
    [switch]$SkipTrain,
    [switch]$RunNotebook
)

$ErrorActionPreference = "Stop"

Write-Host "=== DelhiCold Logistics ML System (Windows 11) ===" -ForegroundColor Cyan

if (-not (Test-Path ".venv")) {
    Write-Host "[1/5] Creating Python virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
}

$Py = ".\.venv\Scripts\python.exe"
$Pip = ".\.venv\Scripts\pip.exe"

Write-Host "[2/5] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
& $Py -m pip install --upgrade pip --quiet
& $Pip install -r requirements.txt --quiet

if (-not $SkipTrain -and -not (Test-Path "models\late_delivery_model.joblib")) {
    Write-Host "[3/5] Training 4 models + RandomizedSearchCV + SHAP on $Rows rows..." -ForegroundColor Yellow
    & $Py -m src.train --rows $Rows
} else {
    Write-Host "[3/5] Model already exists (or -SkipTrain set). Skipping training." -ForegroundColor Green
}

if ($RunNotebook) {
    Write-Host "[3b] Executing training notebook in-place..." -ForegroundColor Yellow
    $env:TRAIN_ROWS = "$Rows"
    & $Py -m jupyter nbconvert --to notebook --execute --inplace notebooks\01_training_and_evaluation.ipynb
}

Write-Host "[4/5] Running unit test suite (pytest)..." -ForegroundColor Yellow
& $Py -m pytest -v

Write-Host "[5/5] Starting FastAPI (http://localhost:8000/docs) and Streamlit (http://localhost:8501)..." -ForegroundColor Green
Start-Process -FilePath $Py -ArgumentList "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"
Start-Sleep -Seconds 3
& $Py -m streamlit run app\streamlit_app.py --server.port 8501
