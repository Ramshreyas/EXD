## 1. llama-benchy — a richer benchmark

vLLM's built-in bench is fast to use but limited. llama-benchy gives us:
- Multiple prompt lengths and output lengths in one run
- Concurrency sweeps (1, 2, 4 concurrent users)
- Proper warmup and statistical averaging
- Timeseries output for plotting

It's already installed on Yoneda:

```bash
# yoneda
which llama-benchy
llama-benchy --version
```

### Setup

```bash
# yoneda
export BASE_URL=http://aitopatom-0a62.local:8000/v1
export MODEL=Qwen/Qwen3.6-35B-A3B
export COMMON_ARGS="--base-url $BASE_URL --api-key EMPTY --model $MODEL --tokenizer $MODEL --latency-mode generation --skip-coherence"
```

### Smoke run

```bash
# yoneda
llama-benchy \
  $COMMON_ARGS \
  --pp 128 \
  --tg 256 \
  --depth 0 \
  --concurrency 1 \
  --runs 1 \
  --no-warmup
```

What the flags mean:
- `--pp 128`: prefill prompt of 128 tokens
- `--tg 256`: target generation of 256 tokens
- `--depth 0`: no additional context before the prompt (simple case)
- `--concurrency 1`: single user
- `--runs 1`: one pass (quick sanity)

### Concurrency scaling demo

Before the full sweep, let me quickly prove these systems are built for
parallelism, not sequential speed. Same prompt, just vary concurrency:

```bash
# yoneda
llama-benchy \
  $COMMON_ARGS \
  --pp 128 \
  --tg 256 \
  --depth 0 \
  --concurrency 1 2 4 8 \
  --runs 1 \
  --no-warmup
```

Watch `tg t/s (total)` — it should climb from ~44 tok/s (c1) to
a much higher number at c4 and c8. This is the whole point: the GPU
was never meant to run a single request at peak speed; it's designed
to batch multiple requests and keep the pipeline full.

> **Next:** open `inference-simulator.html` → switch to the
> **Multi-User Concurrency** tab. Walk through it step-by-step —
> watch 4 users queue, batch, and share the GPU. Then come back
> for the real benchmarks.

---

## 2. Concurrency and batching — the mental model

Last episode we ran benchmarks at concurrency 1 — a single user sending
one prompt at a time. Real servers handle multiple users simultaneously.
Here's how vLLM does it.

### What happens when two users hit the server at once

```
Time ─────────────────────────────────────────────────────►

User A: [prefill──────────] [decode][decode][decode]...
User B:          [prefill──────────] [decode][decode][decode]...

GPU:    [A-prefill] [A-decode               ] [B-prefill]+[A-decode]...
                  [B-prefill               ] [B-decode]
```

Without batching, the GPU idles during each decode step (it only needs
a tiny bit of compute for one token). vLLM fills those gaps:

```
Time ─────────────────────────────────────────────────────►

User A: [prefill──────────] [d][d][d][d][d][d][d][d]...
User B:        [prefill──────────] [d][d][d][d][d][d][d]...
User C:                  [prefill──────────] [d][d][d][d][d]...

GPU:    [A-prefill         ]
          [A-d + B-prefill   ]  ← prefill + decode overlap
            [A-d + B-d + C-prefill]  ← 2 decodes + 1 prefill
              [A-d + B-d + C-d]     ← 3 decodes batched
```

This is **continuous batching**: vLLM dynamically mixes prefill and
decode steps from different users in the same GPU invocation. The GPU
stays busy even when individual requests have slow phases.

### The two scheduler knobs

vLLM exposes two flags that control this behaviour:

| Flag | What it limits | Why it matters |
|------|---------------|----------------|
| `--max-num-seqs` | Max concurrent sequences (requests) | Too low = GPU starved. Too high = KV cache exhaustion |
| `--max-num-batched-tokens` | Max tokens processed in one GPU step | Controls prefill chunk size. Larger = faster prefill, more VRAM |

The baseline config uses **implicit defaults** — vLLM picks these based
on available GPU memory and model size. When we add explicit limits,
we can tune for different workloads:

- **Low latency, interactive**: smaller `max-num-seqs`, smaller batches
  → lower TTFT, but less aggregate throughput
- **High throughput, batch**: larger `max-num-seqs`, larger batches
  → higher aggregate tok/s, but individual TTFT goes up

---

## 3. The baseline numbers

Before tuning anything, we need solid baseline measurements.
Here's the short-context sweep — varies prefill length and
concurrency — running against the baseline config
(`qwen3.6-35b-a3b-baseline`):

Key numbers to watch:
- `tg t/s` (tokens generated / second) — decode throughput per user
- `tg t/s (total)` — aggregate across all concurrent users
- `e2e_ttft` — end-to-end time to first token (includes prefill)
- `peak tg t/s` — peak decode speed during the benchmark

```bash
# atom
cd ~/EXD/projects/serve
./scripts/up.sh qwen3.6-35b-a3b-baseline
./scripts/logs.sh   # wait for "Uvicorn running" then Ctrl+C
```

```bash
# yoneda
export BASE_URL=http://aitopatom-0a62.local:8000/v1
export MODEL=Qwen/Qwen3.6-35B-A3B
export COMMON_ARGS="--base-url $BASE_URL --api-key EMPTY --model $MODEL --tokenizer $MODEL --latency-mode generation --skip-coherence"

llama-benchy \
  $COMMON_ARGS \
  --pp 512 2048 \
  --tg 256 \
  --depth 0 \
  --concurrency 1 2 4 \
  --runs 3 \
  --save-result /tmp/qwen36-35b-baseline-short.json \
  --format json \
  --save-total-throughput-timeseries
```

```bash
# view results
./scripts/bench-view /tmp/qwen36-35b-baseline-short.json
```

Take note of these numbers — they're our reference point:

| pp | c1 tg/s | c2 tg/s | c4 tg/s | c1 TTFT | c2 TTFT | c4 TTFT |
|----|---------|---------|---------|---------|---------|---------|
| 512 | 44.59 | 66.13 | 78.50 | 500 ms | 1114 ms | 1476 ms |
| 2048 | 44.34 | 69.17 | 76.94 | 1017 ms | 1313 ms | 2342 ms |

Observations:
- **Decode speed per user (c1)**: ~44.5 tok/s — this is our single-user
  baseline
- **Concurrency helps**: going from 1→2 users nearly 1.5x aggregate
  throughput; 2→4 adds another ~15%
- **Longer prompts hurt TTFT**: pp=2048 takes ~2x longer for first
  token vs pp=512 — prefill is compute-bound
- The server logs warned that batched token count was low for
  speculation — the implicit defaults may be too conservative

---

## 4. Tuning knob 1: GPU memory utilization

The baseline uses `GPU_MEM_UTIL=0.80` — only 80% of GPU memory is
given to vLLM. The remaining 20% is headroom for the OS, Docker, and
torch compile buffers.

On a machine with **unified memory** (CPU and GPU share the same 128 GB),
0.80 is cautious. The model itself takes ~70 GB in bf16. The KV cache
uses whatever's left after the model weights — and that remaining
space determines how many concurrent sequences we can fit.

Let's push it to 0.85:

```bash
# atom — stop baseline, bring up the optimized default config
cd ~/EXD/projects/serve
./scripts/down.sh
./scripts/up.sh qwen3.6-35b-a3b
```

This config (`qwen3.6-35b-a3b`) combines `GPU_MEM_UTIL=0.85` with
explicit scheduler limits:

```bash
# atom
cat configs/qwen3.6-35b-a3b.env
```

The key additions over baseline:
```bash
--gpu-memory-utilization 0.85   # was 0.80
--max-num-seqs 16               # was implicit (defaulted low)
--max-num-batched-tokens 8192   # was implicit (defaulted low)
```

Rerun the same short-context sweep:

```bash
# yoneda
export PROFILE=qwen3.6-35b-a3b-conservative
llama-benchy \
  $COMMON_ARGS \
  --pp 512 2048 \
  --tg 256 \
  --depth 0 \
  --concurrency 1 2 4 \
  --runs 3 \
  --save-result /tmp/${PROFILE}-short.json \
  --format json \
  --save-total-throughput-timeseries
```

```bash
# view results
./scripts/bench-view /tmp/qwen36-35b-${PROFILE}-short.json
```

Results:

| pp | c1 tg/s | c2 tg/s | c4 tg/s | c1 TTFT | c2 TTFT | c4 TTFT |
|----|---------|---------|---------|---------|---------|---------|
| 512 | 44.04 | 67.82 | 82.55 | 494 ms | 1069 ms | 1581 ms |
| 2048 | 43.95 | 67.69 | 88.20 | 725 ms | 1228 ms | 1957 ms |

What changed:
- **c4 throughput up**: 82.55 vs 78.50 (pp=512) — more KV cache space
  lets more requests run concurrently
- **c4 TTFT at pp=2048 dropped significantly**: 1957 ms vs 2342 ms —
  the larger `max-num-batched-tokens` lets prefill chew through long
  prompts in bigger chunks
- **Single-user decode unchanged** — makes sense, memory util doesn't
  help when there's only one request

---

## 5. Tuning knob 2: push scheduler limits further

The conservative config uses `max-num-seqs=16` and `max-num-batched-tokens=8192`.
What if we push higher for throughput-heavy workloads?

```bash
# atom
cd ~/EXD/projects/serve
./scripts/down.sh
./scripts/up.sh qwen3.6-35b-a3b-throughput
```

This config bumps both limits:

```bash
# atom
cat configs/qwen3.6-35b-a3b-throughput.env
```

```bash
--gpu-memory-utilization 0.88
--max-num-seqs 32
--max-num-batched-tokens 16384
```

Rerun:

```bash
# yoneda
export PROFILE=qwen3.6-35b-a3b-throughput
llama-benchy \
  $COMMON_ARGS \
  --pp 512 2048 \
  --tg 256 \
  --depth 0 \
  --concurrency 1 2 4 \
  --runs 3 \
  --save-result /tmp/${PROFILE}-short.json \
  --format json \
  --save-total-throughput-timeseries
```

```bash
# view results
./scripts/bench-view /tmp/qwen36-35b-${PROFILE}-short.json
```

Results:

| pp | c1 tg/s | c2 tg/s | c4 tg/s | c1 TTFT | c2 TTFT | c4 TTFT |
|----|---------|---------|---------|---------|---------|---------|
| 512 | 45.13 | 69.51 | 83.32 | 601 ms | 1143 ms | 1631 ms |
| 2048 | 43.86 | 66.04 | 88.54 | 738 ms | 1228 ms | 1988 ms |

What changed:
- **Marginal c4 gain**: 88.54 vs 88.20 at pp=2048 — the extra headroom
  helps slightly at high concurrency
- **Worse TTFT**: cold start is heavier because the larger
  `max-num-batched-tokens` means more CUDA graphs to compile
- **Slight c1/c2 decode gains** at pp=512: 45.13/69.51 vs 44.04/67.82

Verdict: the throughput config is a minor win for batch-heavy
scenarios, but the heavier cold start and worse single-user TTFT make
it a niche pick — not a new default.


---

## 6. Comparison

Compare any two runs with colored deltas:

```bash
# Compare tuned vs baseline — green = improvement, red = regression
./scripts/bench-view /tmp/qwen36-35b-baseline-short.json /tmp/qwen36-35b-a3b-conservative-short.json
```

Here's the full sweep across all tuned profiles:

| Profile | pp | c1 tg/s | c2 tg/s | c4 tg/s | c1 TTFT | c2 TTFT | c4 TTFT | Notes |
|---------|----|---------|---------|---------|---------|---------|---------|-------|
| baseline | 512 | 44.59 | 66.13 | 78.50 | 500 ms | 1114 ms | 1476 ms | GPU_MEM=0.80, implicit limits, MTP2 |
| baseline | 2048 | 44.34 | 69.17 | 76.94 | 1017 ms | 1313 ms | 2342 ms | Low batched tokens warning |
| **optimized** | 512 | 44.04 | 67.82 | 82.55 | 494 ms | 1069 ms | 1581 ms | **Winner: balanced** GPU_MEM=0.85, seqs=16, batch=8192, MTP2 |
| **optimized** | 2048 | 43.95 | 67.69 | 88.20 | 725 ms | 1228 ms | 1957 ms | Much better pp2048 TTFT than baseline |
| throughput | 512 | 45.13 | 69.51 | 83.32 | 601 ms | 1143 ms | 1631 ms | Marginal gains, worse TTFT |
| throughput | 2048 | 43.86 | 66.04 | 88.54 | 738 ms | 1228 ms | 1988 ms | Heavier cold start |
| mtp3 | 512 | 40.97 | 66.21 | 88.93 | 989 ms | 1211 ms | 1558 ms | Best c4, worst single-stream |
| mtp3 | 2048 | 41.00 | 65.11 | 92.43 | 764 ms | 1187 ms | 1974 ms | Batch-only profile |
| no-mtp | 512 | 30.07 | 51.51 | 76.94 | 505 ms | 498 ms | 716 ms | ~32% slower decode |
| no-mtp | 2048 | 29.86 | 52.24 | 74.55 | 816 ms | 1245 ms | 1891 ms | Confirms MTP matters |

### Recommendations by workload

| Workload | Best profile | Why |
|----------|-------------|-----|
| **Interactive coding / chat** | `qwen3.6-35b-a3b` (optimized) | Best balance: fast single-user decode, good TTFT, stable at all concurrencies |
| **Batch / agent throughput** | `qwen3.6-35b-a3b-throughput` | Slightly higher aggregate c4 decode at the cost of TTFT |
| **Pure batch at high concurrency** | `qwen3.6-35b-a3b-mtp3` | Highest c4 aggregate (92 tok/s at pp=2048) but poor interactive experience |
| **Debugging / no spec decode** | no-mtp (one-off) | Use only to check if MTP is causing issues; 32% slower decode |

The optimized default (`qwen3.6-35b-a3b`) is the winner for everyday use.
It's now the canonical config — the baseline is preserved only for
comparison.

---

## 7. What we just did

```
1. Built a mental model of continuous batching — how vLLM overlaps
   prefill and decode across multiple users
2. Understood the two scheduler knobs: max-num-seqs and
   max-num-batched-tokens
3. Ran the short-context benchmark sweep against the baseline config
4. Tuned GPU memory utilization from 0.80 → 0.85: more KV cache space,
   better c4 throughput
5. Added explicit scheduler limits: better concurrency handling,
   significantly improved pp=2048 TTFT
6. Pushed scheduler limits further for throughput: marginal c4 gains,
   worse cold start
```

We started with vLLM's implicit defaults, measured the baseline,
then systematically tuned each knob — GPU memory, scheduler limits,
speculative depth — measuring the impact at every step. The winner
is `qwen3.6-35b-a3b` (the optimized config): MTP2, 0.85 memory util,
explicit seq/batch limits. It serves as the new default for all future
work.

---

## Cleanup

```bash
# atom
cd ~/EXD/projects/serve
./scripts/down.sh
```

Remove the one-off no-mtp config if you created it:

```bash
# atom
rm ~/EXD/projects/serve/configs/qwen3.6-35b-a3b-no-mtp.env
```

---
