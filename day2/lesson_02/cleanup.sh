#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "${ROOT_DIR}")"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker not installed; nothing to clean up."
  exit 0
fi

echo "Cleanup for project: ${PROJECT_NAME}"

# 1) Stop project containers (compose if present)
cd "${ROOT_DIR}"
if command -v docker >/dev/null 2>&1; then
  if [[ -f "docker-compose.yml" || -f "docker-compose.yaml" || -f "compose.yml" || -f "compose.yaml" ]]; then
    echo
    echo "Stopping docker compose stack (if running)..."
    docker compose down --remove-orphans --rmi local --volumes || true
  fi
fi

# 2) Stop any containers explicitly labeled for this project
# Users can label containers with: --label scratchai.project=day2
echo
echo "Stopping labeled project containers (scratchai.project=${PROJECT_NAME})..."
mapfile -t labeled_ids < <(docker ps -q --filter "label=scratchai.project=${PROJECT_NAME}" || true)
if [[ ${#labeled_ids[@]} -gt 0 ]]; then
  docker stop "${labeled_ids[@]}" >/dev/null 2>&1 || true
  docker rm -f "${labeled_ids[@]}" >/dev/null 2>&1 || true
fi

# 3) OPTIONAL: stop *all* running containers (disabled by default)
if [[ "${STOP_ALL_CONTAINERS:-0}" == "1" ]]; then
  echo
  echo "STOP_ALL_CONTAINERS=1 set; stopping ALL running containers..."
  mapfile -t all_ids < <(docker ps -q || true)
  if [[ ${#all_ids[@]} -gt 0 ]]; then
    docker stop "${all_ids[@]}" >/dev/null 2>&1 || true
  fi
fi

# 4) Prune unused resources
echo
echo "Pruning unused Docker resources..."
docker container prune -f >/dev/null 2>&1 || true
docker image prune -af >/dev/null 2>&1 || true
docker volume prune -f >/dev/null 2>&1 || true
docker network prune -f >/dev/null 2>&1 || true
docker builder prune -af >/dev/null 2>&1 || true

echo
echo "Done."

