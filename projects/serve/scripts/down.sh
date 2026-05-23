#!/usr/bin/env bash
docker rm -f vllm 2>/dev/null && echo "stopped" || echo "not running"
rm -f "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/configs/active"
