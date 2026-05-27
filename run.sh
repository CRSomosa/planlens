#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  PlanLens — setup & launch
#  Usage:  bash run.sh
# ─────────────────────────────────────────────────────────────
set -e

# ── 1. Check Python ───────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "❌  Python 3 not found. Please install Python 3.9+."
  exit 1
fi

PYVER=$(python3 -c "import sys; print(sys.version_info.minor)")
if [ "$PYVER" -lt 9 ]; then
  echo "❌  Python 3.9+ required. Found 3.$PYVER."
  exit 1
fi

# ── 2. Check ANTHROPIC_API_KEY ────────────────────────────────
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo ""
  echo "⚠️   ANTHROPIC_API_KEY is not set."
  echo "    Export it before running:"
  echo "    export ANTHROPIC_API_KEY=sk-ant-..."
  echo ""
  exit 1
fi

# ── 3. Install dependencies ───────────────────────────────────
echo "📦  Installing dependencies…"
cd "$(dirname "$0")/backend"
pip install -q -r requirements.txt

# ── 4. Launch server ──────────────────────────────────────────
echo ""
echo "🚀  Starting PlanLens API on http://localhost:8000"
echo "🌐  Open the UI at:  http://localhost:8000/app"
echo ""
echo "    Press Ctrl+C to stop."
echo ""

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
