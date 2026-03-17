#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# If something is already listening on 8501, stop it to avoid duplicates.
if command -v ss >/dev/null 2>&1; then
  # Extract the first pid bound to :8501 from ss output (works across formats).
  existing_pid="$(
    ss -ltnp 2>/dev/null \
      | awk -F'pid=' '/:8501/ { for (i=2; i<=NF; i++) { split($i,a,","); if (a[1] ~ /^[0-9]+$/) { print a[1]; exit } } }' \
      || true
  )"
  if [[ -n "${existing_pid}" ]]; then
    echo "Port 8501 already in use (pid=${existing_pid}). Stopping it."
    kill -TERM "${existing_pid}" 2>/dev/null || true
    sleep 1
    kill -KILL "${existing_pid}" 2>/dev/null || true
    sleep 1
  fi
fi

exec "${ROOT_DIR}/startup.sh"

