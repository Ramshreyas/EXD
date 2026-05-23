#!/usr/bin/env bash
# Generic vLLM launcher.
#   Usage: ./scripts/up.sh <config-name>
#   Configs live in ../configs/<name>.env (sourced bash).
#   Required vars per config: MODEL SERVED_NAME IMAGE GPU_MEM_UTIL
#                             EXTRA_DOCKER_ARGS=()  EXTRA_VLLM_ARGS=()
set -euo pipefail

SERVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="$SERVE_ROOT/configs"

usage() {
  echo "Usage: $0 <config-name>"
  echo "Available configs:"
  for f in "$CONFIG_DIR"/*.env; do
    [ -f "$f" ] || continue
    name="$(basename "$f" .env)"
    [ "$name" = "active" ] && continue
    echo "  - $name"
  done
  exit 1
}

[ $# -eq 1 ] || usage
NAME="$1"
CONFIG="$CONFIG_DIR/$NAME.env"
[ -f "$CONFIG" ] || { echo "ERROR: no config at $CONFIG"; usage; }

# Defaults; configs may override.
GPU_MEM_UTIL="0.85"
EXTRA_DOCKER_ARGS=()
EXTRA_VLLM_ARGS=()
# CMD_PREFIX: tokens before the model id. Empty for images whose entrypoint
# already invokes `vllm serve` (e.g. vllm/vllm-openai). Set to (vllm serve)
# for images with a generic entrypoint (e.g. nvcr.io/nvidia/vllm).
CMD_PREFIX=(vllm serve)

# shellcheck source=/dev/null
source "$CONFIG"

: "${MODEL:?MODEL not set in $CONFIG}"
: "${SERVED_NAME:?SERVED_NAME not set in $CONFIG}"
: "${IMAGE:?IMAGE not set in $CONFIG}"

NAME_C="vllm"
mkdir -p "$SERVE_ROOT/logs"

# HF token (optional but recommended; gated models require it)
HF_TOKEN_ARGS=()
if [ -f "$HOME/.hf_token" ]; then
  HF_TOKEN_ARGS=(--env-file "$HOME/.hf_token")
fi

docker rm -f "$NAME_C" 2>/dev/null || true

set -x
docker run -d --name "$NAME_C" \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 \
  -v "$HOME/cache/hf":/root/.cache/huggingface \
  -v "$HOME/models":/models \
  -e HF_HOME=/root/.cache/huggingface \
  "${HF_TOKEN_ARGS[@]}" \
  "${EXTRA_DOCKER_ARGS[@]}" \
  --restart unless-stopped \
  "$IMAGE" \
  "${CMD_PREFIX[@]}" "$MODEL" \
    --host 0.0.0.0 --port 8000 \
    --served-model-name "$SERVED_NAME" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    "${EXTRA_VLLM_ARGS[@]}"
set +x

# Mark active config (used by down/logs/test).
ln -sfn "$NAME.env" "$CONFIG_DIR/active"

echo
echo "started [$NAME] -> $SERVED_NAME"
echo "tail: ./scripts/logs.sh"
echo "test: ./scripts/test.sh"
