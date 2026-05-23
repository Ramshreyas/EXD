#!/usr/bin/env bash
# Smoke-test the running vLLM server.
#   ./scripts/test.sh                 # uses SERVED_NAME from configs/active
#   ./scripts/test.sh <model-id>      # explicit override
set -e

MODEL="${1:-}"
if [ -z "$MODEL" ]; then
  ACTIVE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/configs/active"
  if [ -L "$ACTIVE" ] || [ -f "$ACTIVE" ]; then
    # shellcheck source=/dev/null
    SERVED_NAME=""
    source "$ACTIVE"
    MODEL="$SERVED_NAME"
  fi
fi
: "${MODEL:?no model id (pass as arg or set up an active config)}"

curl -s http://localhost:8000/v1/models | python3 -m json.tool
echo "---"
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL"'",
       "messages":[{"role":"user","content":"In one sentence: what is a transformer?"}],
       "max_tokens":80, "temperature":0.2}' | python3 -m json.tool
