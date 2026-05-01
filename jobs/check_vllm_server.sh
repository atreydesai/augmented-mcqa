#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-${VLLM_HOSTNAME:-}}"
PORT="${VLLM_PORT:-8000}"
API_KEY="${VLLM_API_KEY:-local-dev-key}"

if [[ -z "$HOST" ]]; then
  echo "Usage: jobs/check_vllm_server.sh <gpu-node-hostname>" >&2
  echo "Or set VLLM_HOSTNAME=<gpu-node-hostname>." >&2
  exit 1
fi

curl -fsS "http://${HOST}:${PORT}/v1/models" \
  -H "Authorization: Bearer ${API_KEY}" | python -m json.tool
