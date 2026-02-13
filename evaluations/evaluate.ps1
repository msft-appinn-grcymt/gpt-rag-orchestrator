param(
    [switch]$SkipEval
)

# Dual-venv evaluation script
# This script uses two separate virtual environments:
# 1. .venv-generate: Uses main app SDK (azure-ai-projects==1.1.0b3) for generate_eval_input.py
# 2. .venv-evaluate: Uses new evaluation SDK (azure-ai-projects>=2.0.0b1) for evaluate.py
# 
# This separation is required because the SDKs have incompatible APIs:
# - Main app uses: agents.threads, agents.messages, agents.runs.stream() (1.x API)
# - Evaluation uses: client.evals.create(), client.evals.runs.create() (2.x API)

$ErrorActionPreference = "Stop"

# 1) Validate
if (-not $Env:APP_CONFIG_ENDPOINT) {
    Write-Error "❌ APP_CONFIG_ENDPOINT environment variable is required"
    exit 1
}

# Set PYTHONPATH
$pwdPath = (Get-Location).Path
$Env:PYTHONPATH = "$pwdPath;$pwdPath\src"

# 2) Create venv for generating eval input (uses MAIN app requirements)
Write-Host "▶ Setting up environment for eval input generation..."
python -m venv evaluations\.venv-generate
& "evaluations\.venv-generate\Scripts\Activate.ps1"

# 3) Install MAIN app dependencies (for generate_eval_input.py to work with the app)
pip install --upgrade pip
pip install -r requirements.txt
pip install pandas

# 4) Generate eval-input (uses main app SDK)
Write-Host "▶ Generating evaluation input dataset..."
python evaluations/generate_eval_input.py

# 5) Deactivate and clean up generation environment
deactivate
Remove-Item -Recurse -Force evaluations\.venv-generate

# 6) Create separate venv for running evaluation (uses NEW evaluation SDK)
Write-Host "▶ Setting up environment for evaluation run..."
python -m venv evaluations\.venv-evaluate
& "evaluations\.venv-evaluate\Scripts\Activate.ps1"

# 7) Install evaluation-specific dependencies (SDK 2.0.0b1+)
pip install --upgrade pip
pip install --pre -r evaluations/requirements.txt

# 8) Start app for red team scanning if no external endpoint provided
$AppProcess = $null
$EnableRedTeam = if ($Env:ENABLE_RED_TEAM) { $Env:ENABLE_RED_TEAM } else { "true" }
if (-not $Env:APP_ENDPOINT -and $EnableRedTeam -eq "true") {
    Write-Host "▶ Starting app locally for red team scanning..."
    # Create separate venv for app
    python -m venv evaluations\.venv-app
    & "evaluations\.venv-app\Scripts\Activate.ps1"
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    $Env:PYTHONPATH = "$pwdPath;$pwdPath\src"
    $AppProcess = Start-Process -FilePath "python" -ArgumentList "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000" -PassThru -NoNewWindow
    & "evaluations\.venv-evaluate\Scripts\Activate.ps1"  # Switch back to eval venv
    $Env:APP_ENDPOINT = "http://localhost:8000"
    Write-Host "▶ App started (PID: $($AppProcess.Id)), waiting for startup..."
    Start-Sleep -Seconds 10
}

# 9) Conditionally run evaluation
if (-not $SkipEval) {
    Write-Host "▶ Running evaluation..."
    if ($Env:COMMIT_ID) {
        $Env:COMMIT_ID = $Env:COMMIT_ID.Substring(0, 7)
    }
    python evaluations/evaluate.py
} else {
    Write-Host "▶ Skipping evaluation as requested (-SkipEval)."
}

# 10) Cleanup app if we started it
if ($AppProcess) {
    Write-Host "▶ Stopping local app (PID: $($AppProcess.Id))..."
    Stop-Process -Id $AppProcess.Id -Force -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force evaluations\.venv-app -ErrorAction SilentlyContinue
}

# 11) Teardown
deactivate
Remove-Item -Recurse -Force evaluations\.venv-evaluate

Write-Host "✅ All done."
