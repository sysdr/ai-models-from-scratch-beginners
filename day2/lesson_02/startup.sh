#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
# Keep the virtualenv off /mnt/c (Windows mount) to avoid file/metadata issues.
VENV_DIR="${VENV_DIR:-${HOME}/.venvs/scratchai_day2_lesson01}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/pip" install -r "${SRC_DIR}/requirements.txt"

echo
echo "Running tests..."
"${VENV_DIR}/bin/python" test_model.py
"${VENV_DIR}/bin/python" test_stress.py

echo
echo "Starting dashboard (Streamlit) on http://localhost:8501"
exec "${VENV_DIR}/bin/streamlit" run "${SRC_DIR}/app.py" --server.port 8501 --server.address 0.0.0.0
