#!/usr/bin/env bash
# Build the production image, exercise the gateway and MCP server end to end,
# replace both containers, and verify that API-registered models survived.
#
# Usage: benchmarks/run_e2e.sh [output-dir]
# Requires Docker Compose and a Python with `pip install -e '.[mcp]'` (3.11+ for
# the MCP checks; on older Pythons they are reported as skipped).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/e2e-results}"
PYTHON="${PYTHON:-python}"
PROJECT="${ROUTEMCP_E2E_PROJECT:-routemcp-e2e}"
REQUESTS="${E2E_REQUESTS:-500}"
CONCURRENCY="${E2E_CONCURRENCY:-25}"
GATEWAY_URL="http://127.0.0.1:8000"
MCP_URL="http://127.0.0.1:8001/mcp"

compose() {
  docker compose -f "$ROOT/docker-compose.yml" -p "$PROJECT" --profile mcp "$@"
}

wait_for() {
  local url="$1"
  for _ in $(seq 1 60); do
    if "$PYTHON" -c "import sys, httpx; sys.exit(0 if httpx.get('$url', timeout=2).status_code < 500 else 1)" 2>/dev/null; then
      return 0
    fi
    sleep 0.5
  done
  echo "timed out waiting for $url" >&2
  compose logs --tail 50 >&2
  return 1
}

wait_for_mcp() {
  # GET without an MCP session returns a 4xx once the server is accepting connections.
  wait_for "$MCP_URL"
}

cleanup() {
  compose down -v --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Exercise authentication on both interfaces unless keys were supplied.
export ROUTEMCP_API_KEY="${ROUTEMCP_API_KEY:-e2e-$("$PYTHON" -c 'import secrets; print(secrets.token_hex(16))')}"
export ROUTEMCP_MCP_API_KEY="${ROUTEMCP_MCP_API_KEY:-e2e-$("$PYTHON" -c 'import secrets; print(secrets.token_hex(16))')}"

mkdir -p "$OUT"
# Start from an empty volume so results do not depend on earlier runs.
cleanup

echo "==> building and starting containers"
compose up --build -d
wait_for "$GATEWAY_URL/healthz"
wait_for_mcp

echo "==> functional, load, failover, and MCP checks"
"$PYTHON" "$ROOT/benchmarks/e2e_gateway.py" \
  --url "$GATEWAY_URL" --mcp-url "$MCP_URL" \
  --requests "$REQUESTS" --concurrency "$CONCURRENCY" \
  --output "$OUT/e2e-run.json"

echo "==> replacing containers (volume kept)"
compose up -d --force-recreate
wait_for "$GATEWAY_URL/healthz"
wait_for_mcp

echo "==> verifying persisted registration after replacement"
"$PYTHON" "$ROOT/benchmarks/e2e_gateway.py" \
  --url "$GATEWAY_URL" --mcp-url "$MCP_URL" \
  --verify-persisted --output "$OUT/e2e-restart.json"

echo "==> end-to-end run passed; reports in $OUT"
