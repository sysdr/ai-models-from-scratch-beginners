#!/usr/bin/env bash
# Day 3 lesson_03: remove local artifacts (root cleanup.sh also runs this)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"
echo "Cleanup for project: day3/lesson_03"
rm -rf __pycache__ .venv *.npy 2>/dev/null || true
echo "Done."
