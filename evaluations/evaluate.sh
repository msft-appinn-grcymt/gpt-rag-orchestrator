#!/usr/bin/env bash
# =============================================================================
# DUAL-VENV EVALUATION SCRIPT
# =============================================================================
# This script uses two separate virtual environments due to SDK incompatibility:
#
# 1. .venv-generate: Uses main app SDK (azure-ai-projects==1.1.0b3)
#    - Required for generate_eval_input.py which imports from src/
#    - Uses the "classic" agents API (threads/messages/runs)
#
# 2. .venv-evaluate: Uses new evaluation SDK (azure-ai-projects>=2.0.0b1)
#    - Required for evaluate.py which uses the new evals API
#    - Uses client.evals.create() and client.evals.runs.create()
#
# The SDKs have incompatible APIs and cannot coexist in the same environment.
# SDK 2.0.0b1+ replaces the agents paradigm with the "Responses" protocol.
#
# For full migration to 2.0.0b1+, the main app's agent strategies would need
# complete rewrites (~500+ lines) to use the new conversations/responses API.
# =============================================================================
set -euo pipefail

# Default: run evaluation
SKIP_EVAL=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-eval)
      SKIP_EVAL=true
      shift
      ;;
    *)
      echo "❌ Error: Unknown argument: $1"
      echo "Usage: $0 [--skip-eval]"
      exit 1
      ;;
  esac
done

# 1) Validate
if [ -z "${APP_CONFIG_ENDPOINT:-}" ]; then
  echo "❌  Error: APP_CONFIG_ENDPOINT is not set"
  exit 1
fi

# 2) Create venv for generating eval input (uses MAIN app requirements)
echo "▶ Setting up environment for eval input generation..."
python -m venv evaluations/.venv-generate
source evaluations/.venv-generate/bin/activate

# 3) Install MAIN app dependencies (for generate_eval_input.py to work with the app)
pip install --upgrade pip
pip install -r requirements.txt
# Install additional evaluation dependencies that don't conflict
pip install pandas

# 4) Ensure Python can see your src/ package
export PYTHONPATH="$(pwd):$(pwd)/src"

# 5) Generate eval-input (uses main app SDK)
echo "▶ Generating evaluation input dataset..."
python evaluations/generate_eval_input.py

# 6) Deactivate and clean up generation environment
deactivate
rm -rf evaluations/.venv-generate

# 7) Create separate venv for running evaluation (uses NEW evaluation SDK)
echo "▶ Setting up environment for evaluation run..."
python -m venv evaluations/.venv-evaluate
source evaluations/.venv-evaluate/bin/activate

# 8) Install evaluation-specific dependencies (SDK 2.0.0b1+)
pip install --upgrade pip
pip install --pre -r evaluations/requirements.txt

# 9) Ensure Python can see your src/ package
export PYTHONPATH="$(pwd):$(pwd)/src"

# 10) Start app for red team scanning if no external endpoint provided
APP_PID=""
if [ -z "${APP_ENDPOINT:-}" ] && [ "${ENABLE_RED_TEAM:-true}" = "true" ]; then
  echo "▶ Starting app locally for red team scanning..."
  # Start app in background using the generate venv (has main app SDK)
  python -m venv evaluations/.venv-app
  source evaluations/.venv-app/bin/activate
  pip install --upgrade pip -q
  pip install -r requirements.txt -q
  export PYTHONPATH="$(pwd):$(pwd)/src"
  uvicorn src.main:app --host 0.0.0.0 --port 8000 &
  APP_PID=$!
  source evaluations/.venv-evaluate/bin/activate  # Switch back to eval venv
  export APP_ENDPOINT="http://localhost:8000"
  echo "▶ App started (PID: $APP_PID), waiting for startup..."
  sleep 10  # Wait for app to start
fi

# 11) Conditionally run evaluation
if [ "$SKIP_EVAL" = false ]; then
  echo "▶ Running evaluation…"
  # Pass commit ID from environment or use short form if available
  if [ -n "${COMMIT_ID:-}" ]; then
    # Use first 7 characters of commit hash for short format
    SHORT_COMMIT_ID="${COMMIT_ID:0:7}"
    COMMIT_ID="$SHORT_COMMIT_ID" python evaluations/evaluate.py
  else
    python evaluations/evaluate.py
  fi
else
  echo "▶ Skipping evaluation as requested (--skip-eval)."
fi

# 12) Cleanup app if we started it
if [ -n "$APP_PID" ]; then
  echo "▶ Stopping local app (PID: $APP_PID)..."
  kill $APP_PID 2>/dev/null || true
  rm -rf evaluations/.venv-app
fi

# 13) Teardown
deactivate
rm -rf evaluations/.venv-evaluate

echo "✅  All done."
