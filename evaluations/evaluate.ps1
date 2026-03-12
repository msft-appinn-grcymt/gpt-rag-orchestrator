param(
    [switch]$SkipEval
)

# =============================================================================
# UNIFIED EVALUATION SCRIPT
# =============================================================================
# Single virtual environment � main app and evaluation SDK unified on
# azure-ai-projects>=2.0.0. Red team scanning runs in-process via TestClient.
# =============================================================================

$ErrorActionPreference = "Stop"

if (-not $Env:APP_CONFIG_ENDPOINT) {
    Write-Error "APP_CONFIG_ENDPOINT environment variable is required"
    exit 1
}

$pwdPath = (Get-Location).Path
$Env:PYTHONPATH = "$pwdPath;$pwdPath\src"

Write-Host "Setting up unified evaluation environment..."
python -m venv evaluations\.venv
& "evaluations\.venv\Scripts\Activate.ps1"

pip install --upgrade pip
pip install --no-deps 'semantic-kernel>=1.40.0'
pip install -r requirements.txt

Write-Host "Generating evaluation input dataset..."
python evaluations/generate_eval_input.py

if (-not $SkipEval) {
    Write-Host "Running evaluation..."
    if ($Env:COMMIT_ID) {
        $Env:COMMIT_ID = $Env:COMMIT_ID.Substring(0, 7)
    }
    python evaluations/evaluate.py
} else {
    Write-Host "Skipping evaluation as requested (-SkipEval)."
}

deactivate
Remove-Item -Recurse -Force evaluations\.venv

Write-Host "All done."
