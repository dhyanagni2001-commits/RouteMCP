#!/usr/bin/env bash
# Run the published routing evaluation against OpenAI's gpt-4.1 family.
# Reads OPENAI_API_KEY from the environment or a gitignored .env file.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -z "${OPENAI_API_KEY:-}" ] && [ -f "$ROOT/.env" ]; then
  set -a; . "$ROOT/.env"; set +a
fi
: "${OPENAI_API_KEY:?set OPENAI_API_KEY or add it to .env}"

SEED="${EVAL_SEED:-0}"
OUT="${1:-$ROOT/evaluation/results/openai-gpt-4.1-seed$SEED.json}"
routemcp evaluate \
  --config "$ROOT/evaluation/openai.yaml" \
  --dataset "$ROOT/evaluation/tasks.jsonl" \
  --strongest-model gpt-4.1 \
  --cheapest-model gpt-4.1-nano \
  --calibration-fraction 0.5 \
  --profiles balanced,quality,cheapest,balanced+cascade,quality+cascade \
  --seed "$SEED" \
  --concurrency "${EVAL_CONCURRENCY:-8}" \
  --output "$OUT"
python "$ROOT/evaluation/summarize.py" "$OUT"
