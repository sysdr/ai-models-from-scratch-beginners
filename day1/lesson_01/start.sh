#!/usr/bin/env bash
# ScratchAI-Beginner | Day 1: start NumPy Playground (Streamlit app)
set -e
cd "$(dirname "$0")"

# Prefer venv: use python -m streamlit (works even when .venv/bin/streamlit is missing)
if [[ -x .venv/bin/python ]]; then
  .venv/bin/pip install -q -r requirements.txt 2>/dev/null || true
  echo "NumPy Playground starting at http://localhost:8501"
  exec .venv/bin/python -m streamlit run app.py --server.headless true "$@"
fi

# System streamlit or python -m streamlit
if command -v streamlit &>/dev/null; then
  echo "NumPy Playground starting at http://localhost:8501"
  exec streamlit run app.py --server.headless true "$@"
fi
if python3 -c "import streamlit" 2>/dev/null; then
  echo "NumPy Playground starting at http://localhost:8501"
  exec python3 -m streamlit run app.py --server.headless true "$@"
fi

# No streamlit: create venv and install
echo "Creating .venv and installing requirements..."
python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
echo "NumPy Playground starting at http://localhost:8501"
exec .venv/bin/python -m streamlit run app.py --server.headless true "$@"
