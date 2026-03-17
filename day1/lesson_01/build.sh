#!/usr/bin/env bash
# ScratchAI-Beginner | Day 1 build: generate lesson_01 (via setup.py) and run tests
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Running setup.py..."
python3 setup.py

echo ""
echo "==> Running tests (test_model.py)..."
cd lesson_01 && python3 test_model.py

echo ""
echo "Build finished successfully."
