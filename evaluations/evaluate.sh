#!/usr/bin/env bash
# =============================================================================
# UNIFIED EVALUATION SCRIPT
# =============================================================================
# Single virtual environment � main app and evaluation SDK unified on
# azure-ai-projects>=2.0.0. Red team scanning runs in-process via TestClient.
# =============================================================================
set -euo pipefail

SKIP_EVAL=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-eval) SKIP_EVAL=true; shift ;;
    *) echo "Error: Unknown argument: $1"; echo "Usage: $0 [--skip-eval]"; exit 1 ;;
  esac
done

if [ -z "${APP_CONFIG_ENDPOINT:-}" ]; then
  echo "Error: APP_CONFIG_ENDPOINT is not set"; exit 1
fi

echo "Setting up unified evaluation environment..."
python -m venv evaluations/.venv
source evaluations/.venv/bin/activate

pip install --upgrade pip
pip install --no-deps 'semantic-kernel>=1.40.0'
pip install -r requirements.txt

export PYTHONPATH="$(pwd):$(pwd)/src"

echo "Generating evaluation input dataset..."
python evaluations/generate_eval_input.py

if [ "$SKIP_EVAL" = false ]; then
  echo "Running evaluation..."
  if [ -n "${COMMIT_ID:-}" ]; then
    SHORT_COMMIT_ID="${COMMIT_ID:0:7}"
    COMMIT_ID="$SHORT_COMMIT_ID" python evaluations/evaluate.py
  else
    python evaluations/evaluate.py
  fi
else
  echo "Skipping evaluation as requested (--skip-eval)."
fi

deactivate
rm -rf evaluations/.venv
echo "All done."
