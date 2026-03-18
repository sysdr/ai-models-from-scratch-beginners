#!/usr/bin/env bash
# ScratchAI-Beginner | Day 3 Lesson 03: Matrix Transformer (Streamlit app)
set -e
PORT=8501
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# Avoid duplicate: if port is already in use, assume app is running
if command -v curl &>/dev/null && curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/" 2>/dev/null | grep -q 200; then
  echo "Matrix Transformer is already running at http://localhost:${PORT}"
  echo "Open that URL in your browser. To restart, stop the existing process first."
  exit 0
fi

# Prefer venv: use python -m streamlit with fixed port
if [[ -x .venv/bin/python ]]; then
  .venv/bin/pip install -q -r requirements.txt 2>/dev/null || true
  echo "Matrix Transformer starting at http://localhost:${PORT}"
  exec .venv/bin/python -m streamlit run app.py --server.headless true --server.port "${PORT}" "$@"
fi

# System streamlit or python -m streamlit
if command -v streamlit &>/dev/null; then
  echo "Matrix Transformer starting at http://localhost:${PORT}"
  exec streamlit run app.py --server.headless true --server.port "${PORT}" "$@"
fi
if python3 -c "import streamlit" 2>/dev/null; then
  echo "Matrix Transformer starting at http://localhost:${PORT}"
  exec python3 -m streamlit run app.py --server.headless true --server.port "${PORT}" "$@"
fi

# No streamlit: create venv and install
echo "Creating .venv and installing requirements..."
python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
echo "Matrix Transformer starting at http://localhost:${PORT}"
exec .venv/bin/python -m streamlit run app.py --server.headless true --server.port "${PORT}" "$@"
