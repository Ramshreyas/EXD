# Ep. 5 — Speculative Decoding

> Why the GPU wastes 90% of its matrix multiplication, how speculation fixes it,
> and when "just add more speculation" makes things worse.

## 1. The problem — autoregressive decode

Last episode we saw continuous batching fill GPU idle time. But even
perfect batching can't fix this: **each step produces EXACTLY one token**.

### The Gilbert Strang perspective

> "If your matrix-multiply is 100× more expensive than the extra token check,
> why not do MORE work in that single forward pass and get MORE tokens out?"

That's speculative decoding in one sentence.

### Simulator

Open `simulator.html` (attached). Step through auto-regressive (1 token
per step) vs speculative (draft model proposes N tokens ahead, full model
verifies them all at once).

- Green tokens = draft guessed right → free speed
- Red tokens = draft guessed wrong → wasted compute
- Slide spec depth from 0 to 3 to see the tradeoff

---

## 2. Benchmark setup

```bash
# yoneda
export BASE_URL=http://aitopatom-0a62.local:8000/v1
export MODEL=Qwen/Qwen3.6-35B-A3B
export COMMON_ARGS="--base-url $BASE_URL --api-key EMPTY --model $MODEL --tokenizer $MODEL --latency-mode generation --skip-coherence"
```

---

## 3. Single-user benchmark

### MTP 0 — baseline (no speculation)

```bash
# atom — show config
cat configs/qwen3.6-35b-a3b-mtp0.env
```

```bash
# atom — start with MTP 0 (no speculation)
./scripts/up.sh qwen3.6-35b-a3b-mtp0
./scripts/logs.sh   # wait for "Uvicorn running" then Ctrl+C
```

```bash
# yoneda
llama-benchy $COMMON_ARGS \
  --pp 512 --tg 256 --depth 0 --concurrency 1 --runs 3 \
  --save-result /tmp/qwen35b-mtp0-single.json --format json
```

```bash
./scripts/bench-view /tmp/qwen35b-mtp0-single.json
```

### MTP 2 — default

```bash
# atom — show config
cat configs/qwen3.6-35b-a3b.env
```

```bash
# atom — switch
./scripts/down.sh && ./scripts/up.sh qwen3.6-35b-a3b
```

```bash
# yoneda
llama-benchy $COMMON_ARGS \
  --pp 512 --tg 256 --depth 0 --concurrency 1 --runs 3 \
  --save-result /tmp/qwen35b-mtp2-single.json --format json
```

```bash
./scripts/bench-view /tmp/qwen35b-mtp2-single.json
```

### MTP 3 — aggressive

```bash
# atom — show config
cat configs/qwen3.6-35b-a3b-mtp3.env
```

```bash
# atom — switch
./scripts/down.sh && ./scripts/up.sh qwen3.6-35b-a3b-mtp3
```

```bash
# yoneda
llama-benchy $COMMON_ARGS \
  --pp 512 --tg 256 --depth 0 --concurrency 1 --runs 3 \
  --save-result /tmp/qwen35b-mtp3-single.json --format json
```

```bash
./scripts/bench-view /tmp/qwen35b-mtp3-single.json
```

### Comparison

```bash
# MTP 0 vs MTP 2
./scripts/bench-view /tmp/qwen35b-mtp0-single.json /tmp/qwen35b-mtp2-single.json

# MTP 2 vs MTP 3
./scripts/bench-view /tmp/qwen35b-mtp2-single.json /tmp/qwen35b-mtp3-single.json
```

**Expected:**

| Depth | c1 tg/s |
|-------|---------|
| 0 | ~22 |
| 2 | ~30 |
| 3 | ~28 |

MTP 0 = baseline. MTP 2 = ~40% faster. MTP 3 = can regress at c1.

---

## 4. Cleanup

```bash
# atom
cd ~/EXD/projects/serve
./scripts/down.sh

# yoneda — kill the dashboard tunnel
pkill -f "ssh.*11000:localhost:11000"
```
