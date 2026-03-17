#!/usr/bin/env bash
# Quick check that the NumPy Playground Streamlit app is running and responding
set -e
URL="${1:-http://localhost:8501}"
echo "Checking $URL ..."
if curl -sf -o /dev/null "$URL"; then
  echo "OK: App is responding (HTTP 200)"
  echo "Open in browser: $URL"
else
  echo "Not responding. Start the app with: ./start.sh"
  exit 1
fi
