# Qwen3.6 35B-A3B llama-benchy runbook

Use this from the host or benchmark machine against the GB10 vLLM endpoint. The
commands assume `llama-benchy` is already installed.

## Endpoint

If the benchmark machine can reach the GB10 host directly:

```bash
export BASE_URL=http://atom:8000/v1
```

If it needs a tunnel:

```bash
ssh -L 8000:localhost:8000 atom
export BASE_URL=http://localhost:8000/v1
```

If running directly on the GB10 host, use:

```bash
export BASE_URL=http://localhost:8000/v1
```

Common arguments:

```bash
export MODEL=Qwen/Qwen3.6-35B-A3B
export COMMON_ARGS="--base-url $BASE_URL --api-key EMPTY --model $MODEL --served-model-name $MODEL --tokenizer $MODEL --latency-mode generation --skip-coherence"
```

## Smoke

Run this after every vLLM profile change before longer sweeps:

```bash
llama-benchy \
  $COMMON_ARGS \
  --pp 128 \
  --tg 256 \
  --depth 0 \
  --runs 1 \
  --no-warmup \
  --save-result /tmp/qwen36-35b-a3b-smoke.json \
  --format json
```

On the baseline profile, a verified smoke run produced roughly `625 pp t/s`,
`42 tg t/s`, `53 peak tg t/s`, and `365 ms e2e_ttft`.

## Baseline Suites

Short-context decode and concurrency:

```bash
llama-benchy \
  $COMMON_ARGS \
  --pp 512 2048 \
  --tg 128 512 \
  --depth 0 \
  --concurrency 1 2 4 \
  --runs 3 \
  --save-result qwen36-35b-a3b-${PROFILE:-baseline}-short.json \
  --format json \
  --save-total-throughput-timeseries
```

For the short-suite sweep used during tuning, use `--tg 256` instead of the
larger `--tg` list above to keep profile iteration quick:

```bash
llama-benchy \
  $COMMON_ARGS \
  --pp 512 2048 \
  --tg 256 \
  --depth 0 \
  --concurrency 1 2 4 \
  --runs 3 \
  --save-result /tmp/qwen36-35b-a3b-${PROFILE}-short.json \
  --format json \
  --save-total-throughput-timeseries
```

## Short-Suite Results

Test shape: `--pp 512 2048 --tg 256 --depth 0 --concurrency 1 2 4 --runs 3`.

| Profile | pp | c1 tg/s | c2 tg/s | c4 tg/s | c1 TTFT | c2 TTFT | c4 TTFT | Notes |
|---------|----|---------|---------|---------|---------|---------|---------|-------|
| `qwen3.6-35b-a3b-baseline` | 512 | 44.59 | 66.13 | 78.50 | 500 ms | 1114 ms | 1476 ms | Initial recipe: `GPU_MEM_UTIL=0.80`, implicit scheduler limits, MTP 2. |
| `qwen3.6-35b-a3b-baseline` | 2048 | 44.34 | 69.17 | 76.94 | 1017 ms | 1313 ms | 2342 ms | Baseline warned that scheduled tokens were low for speculation. |
| `qwen3.6-35b-a3b-conservative` | 512 | 44.04 | 67.82 | 82.55 | 494 ms | 1069 ms | 1581 ms | Balanced winner: `GPU_MEM_UTIL=0.85`, `max-num-seqs=16`, `max-num-batched-tokens=8192`, MTP 2. |
| `qwen3.6-35b-a3b-conservative` | 2048 | 43.95 | 67.69 | 88.20 | 725 ms | 1228 ms | 1957 ms | Better c4 throughput and much better pp2048 TTFT than baseline. |
| `qwen3.6-35b-a3b-throughput` | 512 | 45.13 | 69.51 | 83.32 | 601 ms | 1143 ms | 1631 ms | Slight c1/c2/c4 decode gains, worse TTFT; heavier cold start due 16k compile range. |
| `qwen3.6-35b-a3b-throughput` | 2048 | 43.86 | 66.04 | 88.54 | 738 ms | 1228 ms | 1988 ms | Marginal c4 gain over conservative, not a default pick. |
| `qwen3.6-35b-a3b-mtp3` | 512 | 40.97 | 66.21 | 88.93 | 989 ms | 1211 ms | 1558 ms | Higher c4 aggregate, poor single-stream decode and TTFT. |
| `qwen3.6-35b-a3b-mtp3` | 2048 | 41.00 | 65.11 | 92.43 | 764 ms | 1187 ms | 1974 ms | Useful only for batch/throughput work at c4. |
| `qwen3.6-35b-a3b-no-mtp` | 512 | 30.07 | 51.51 | 76.94 | 505 ms | 498 ms | 716 ms | Decode is much slower; keep MTP enabled. |
| `qwen3.6-35b-a3b-no-mtp` | 2048 | 29.86 | 52.24 | 74.55 | 816 ms | 1245 ms | 1891 ms | Confirms MTP is pulling real weight on GB10. |

Decision: `qwen3.6-35b-a3b` is now the optimized default and matches the tested
conservative MTP2 profile. Keep `qwen3.6-35b-a3b-throughput` for slightly higher
aggregate c2/c4 decode when TTFT and cold-start time matter less. Keep
`qwen3.6-35b-a3b-mtp3` only for batch-heavy c4 tests. No-MTP, fp8-KV, and
native-262k long-context configs were not kept as committed launch profiles; use
the benchmark notes below to recreate them for diagnostics or future capacity
sweeps.

Long-context prefill and decode without intentional cache reuse:

```bash
llama-benchy \
  $COMMON_ARGS \
  --pp 2048 \
  --tg 64 \
  --depth 0 4096 16384 32768 65536 98304 \
  --concurrency 1 \
  --runs 3 \
  --no-cache \
  --save-result qwen36-35b-a3b-${PROFILE:-baseline}-depth.json \
  --format json
```

Prefix-cache follow-up workload:

```bash
llama-benchy \
  $COMMON_ARGS \
  --pp 2048 \
  --tg 128 \
  --depth 4096 16384 32768 65536 \
  --concurrency 1 2 \
  --runs 3 \
  --enable-prefix-caching \
  --save-result qwen36-35b-a3b-${PROFILE:-baseline}-prefix.json \
  --format json
```

## Profile Order

Start each profile on GB10 with `./scripts/up.sh <profile>` from the `serve` directory, then run smoke before the full suites.

1. `qwen3.6-35b-a3b`
2. `qwen3.6-35b-a3b-throughput`
3. `qwen3.6-35b-a3b-mtp3`
4. `qwen3.6-35b-a3b-baseline`

Set `PROFILE` before each run so result filenames stay aligned with the server config:

```bash
export PROFILE=qwen3.6-35b-a3b-conservative
```

## Recreating One-Off Experiments

- No MTP: copy `qwen3.6-35b-a3b.env` and remove `--speculative-config`. This was
  much slower for decode in the short-suite sweep.
- fp8 KV: copy `qwen3.6-35b-a3b-throughput.env` and add `--kv-cache-dtype fp8`.
  Treat this as a capacity experiment and re-run quality checks.
- Native 262k context: copy `qwen3.6-35b-a3b.env`, set `--max-model-len 262144`,
  and lower `--max-num-seqs` if startup runs out of KV/cache headroom.

## What To Compare

Compare `pp t/s`, `tg t/s`, aggregate `tg t/s (total)`, `peak t/s`, `ttfr`, `est_ppt`, `e2e_ttft`, standard deviation, and failures/timeouts. Keep the best profile by workload rather than one global winner:

- Interactive coding: low `e2e_ttft`, good `tg` at concurrency 1-2, stable 128k.
- Agent throughput: best aggregate decode at concurrency 2-4 and strong prefix-cache follow-up speed.
- Long documents: stable deep context and prefill speed over high batching.
