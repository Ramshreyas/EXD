# LLM Inference Benchmarking

Build a mental model of what inference actually is, then benchmark it. Start vanilla, measure, tweak one knob, measure again.

---

## What Is Inference? — The Toy Model

User input: "The cat sat"

```
                     Tokenize
    "The cat sat"  →  [576, 3797, 7236]

                     PREFILL (happens once)
    Run ALL 3 tokens through the model in one shot.
    Every token attends to every other token.
    Produces the first output token AND saves KV cache.

                     DECODE (loops until done)
    Step 1: Append new token, run through model.
    Step 2: Only the NEW token needs fresh computation.
            Previous tokens reuse cached KV entries.
    Step 3: Model outputs next token.
    Step 4: Repeat.

                     Detokenize
    [389, 278, 3098, ...]  →  "on the mat."
```

### How the KV Cache Actually Works

During prefill, the model saves a "summary" of each token — two vectors called Key (K) and Value (V). Think of these as the model's scratch notes on each word.

```
Prefill builds the cache:
  "The"  →  [K₁ V₁]     <-- saved
  "cat"  →  [K₂ V₂]     <-- saved
  "sat"  →  [K₃ V₃]     <-- saved

Decode reuses it:
  New token "on":
    → compute [K₄ V₄] from "on" itself     <-- cheap (1 token's worth)
    → read   [K₁..K₃, V₁..V₃] from cache   <-- fast (just a memory read)
    → no need to recompute "The cat sat"
```

Without the KV cache, every decode step would re-process the entire growing sequence. The cache makes decode O(n) instead of O(n²). But the cache grows with every token, so eventually the GPU spends more time reading it than computing. That's why decode is memory-bound.

### Prefill vs Decode — The Key Insight

| | Prefill | Decode |
|---|---|---|
| Runs | Entire prompt at once | One new token at a time |
| How many times | 1 | N (once per output token) |
| Bottleneck | Compute (matrix multiplies) | Memory (KV cache reads) |
| GPU behaviour | High utilization | Low utilization, bandwidth-bound |

Two latency numbers matter:
- **TTFT** (Time To First Token) — dominated by prefill
- **TPOT** (Time Per Output Token) — dominated by decode / memory bandwidth

---

## The Model

Qwen3.6 35B-A3B — a Mixture-of-Experts model:
- 35B total parameters, only 3B active per token
- Supports MTP (Multi-Token Prediction) for faster decode
- 128k context window

---

## Baseline Benchmark

Start with conservative flags: `GPU_MEM_UTIL=0.80`, implicit scheduler limits, MTP2 speculation.

```bash
# Bring it up
cd ~/serve
./scripts/up.sh qwen3.6-35b-a3b-baseline

# vLLM's built-in bench (runs inside container — no network overhead)
docker exec vllm vllm bench serve \
  --backend vllm \
  --model Qwen/Qwen3.6-35B-A3B \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 50 \
  --request-rate inf
```

---

## Richer Benchmarks with llama-benchy

llama-benchy gives us multi-concurrency sweeps, proper warmup, and statistical averaging.

```bash
export BASE_URL=http://<hostname>.local:8000/v1
export MODEL=Qwen/Qwen3.6-35B-A3B
export COMMON_ARGS="--base-url $BASE_URL --api-key EMPTY --model $MODEL --tokenizer $MODEL --latency-mode generation --skip-coherence"

# Short-context sweep
llama-benchy \
  $COMMON_ARGS \
  --pp 512 2048 \
  --tg 256 \
  --depth 0 \
  --concurrency 1 2 4 \
  --runs 3 \
  --save-total-throughput-timeseries
```

Key metrics to watch:
- `tg t/s` — decode throughput per user
- `tg t/s (total)` — aggregate across all concurrent users
- `e2e_ttft` — end-to-end time to first token

---

## Interactive Simulator

Open the [Inference Pipeline Simulator](https://huggingface.co/spaces/EXD-AI/inference-simulator) to visualize prefill, decode, and KV cache behaviour step by step.

---

## Summary

1. Build a mental model of inference: prefill vs decode, TTFT vs TPOT
2. Serve a model with baseline config
3. Run built-in benchmarks for quick local checks
4. Run llama-benchy for proper end-to-end sweeps

The point is to get comfortable with the tools and build intuition for what the numbers mean. Next: performance tuning — concurrency, batching, and flag sweeps.
