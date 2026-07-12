# Performance Tuning for LLMs

Concurrency, batching, and flag sweeps — systematically tuning vLLM for better throughput and latency.

---

## Concurrency and Batching — The Mental Model

Last episode established baselines at single-user concurrency. Real servers handle multiple users simultaneously. Here's how vLLM does it.

### Continuous Batching

When two users hit the server at once:

```
User A: [prefill-------] [decode][decode][decode]...
User B:       [prefill-------] [decode][decode]...

GPU:    [A-prefill] [A-d + B-prefill] [A-d + B-d] ...
```

vLLM dynamically mixes prefill and decode steps from different users in the same GPU invocation. The GPU stays busy even when individual requests have slow phases.

### The Two Scheduler Knobs

| Flag | What it limits | Why it matters |
|------|---------------|----------------|
| `--max-num-seqs` | Max concurrent sequences | Too low = GPU starved. Too high = KV cache exhaustion |
| `--max-num-batched-tokens` | Max tokens per GPU step | Larger = faster prefill, more VRAM |

Tradeoff:
- **Low latency, interactive**: smaller limits → lower TTFT, less aggregate throughput
- **High throughput, batch**: larger limits → higher aggregate tok/s, individual TTFT goes up

---

## Baseline Numbers

Before tuning, a reference sweep against the baseline config (`GPU_MEM_UTIL=0.80`, implicit limits):

| pp | c1 tg/s | c2 tg/s | c4 tg/s | c1 TTFT | c2 TTFT | c4 TTFT |
|----|---------|---------|---------|---------|---------|---------|
| 512 | 44.59 | 66.13 | 78.50 | 500 ms | 1114 ms | 1476 ms |
| 2048 | 44.34 | 69.17 | 76.94 | 1017 ms | 1313 ms | 2342 ms |

Observations:
- Decode speed per user (c1): ~44.5 tok/s — single-user baseline
- Concurrency helps: 1→2 users nearly 1.5x aggregate throughput
- Longer prompts hurt TTFT: pp=2048 takes ~2x longer
- Server logs warned of low batched token count for speculation

---

## Tuning Knob 1: GPU Memory Utilization

Move from 0.80 → 0.85, add explicit scheduler limits:

```bash
--gpu-memory-utilization 0.85
--max-num-seqs 16
--max-num-batched-tokens 8192
```

Results:

| pp | c1 tg/s | c2 tg/s | c4 tg/s | c1 TTFT | c2 TTFT | c4 TTFT |
|----|---------|---------|---------|---------|---------|---------|
| 512 | 44.04 | 67.82 | 82.55 | 494 ms | 1069 ms | 1581 ms |
| 2048 | 43.95 | 67.69 | 88.20 | 725 ms | 1228 ms | 1957 ms |

Gains: c4 throughput up, pp=2048 TTFT significantly improved (1957 vs 2342 ms).

---

## Tuning Knob 2: Push Scheduler Limits

Push further for throughput-heavy workloads:

```bash
--gpu-memory-utilization 0.88
--max-num-seqs 32
--max-num-batched-tokens 16384
```

Results:

| pp | c1 tg/s | c2 tg/s | c4 tg/s | c1 TTFT | c2 TTFT | c4 TTFT |
|----|---------|---------|---------|---------|---------|---------|
| 512 | 45.13 | 69.51 | 83.32 | 601 ms | 1143 ms | 1631 ms |
| 2048 | 43.86 | 66.04 | 88.54 | 738 ms | 1228 ms | 1988 ms |

Marginal c4 gains, worse TTFT. Heavier cold start from larger `max-num-batched-tokens`.

---

## Winner: Balanced Config

The optimized default wins for everyday use:

- `GPU_MEM_UTIL=0.85`
- `max-num-seqs=16`
- `max-num-batched-tokens=8192`
- MTP2 speculation

### Recommendations by Workload

| Workload | Best config | Why |
|----------|-------------|-----|
| Interactive coding / chat | Optimized (0.85) | Best balance: fast single-user decode, good TTFT |
| Batch / agent throughput | Throughput (0.88) | Slightly higher aggregate c4 decode at cost of TTFT |

---

## Interactive Simulator

Open the [Inference Simulator v2](https://huggingface.co/spaces/EXDai/inference-simulator-v2) to visualize concurrency, batching, and memory constraints step by step.

---

## Summary

1. Built a mental model of continuous batching
2. Ran baseline sweep, identified bottlenecks
3. Tuned GPU memory from 0.80 → 0.85: more KV cache, better throughput
4. Added explicit scheduler limits: significantly improved long-prompt TTFT
5. Pushed further for throughput: marginal gains, worse cold start

The balanced config is now the canonical default for all future experiments.
