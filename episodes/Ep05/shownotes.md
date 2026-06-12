# Ep. 5 — Speculative Decoding

> Why the GPU wastes 90% of its matrix multiplication, how speculation fixes it,
> and when "just add more speculation" makes things worse.

## 1. The problem — autoregressive decode

Last episode we saw continuous batching fill GPU idle time. But even
perfect batching can't fix this: **each step produces EXACTLY one token**.

### Visual proof: GPU idle time during decode

On atom, start the server and watch the GPU with `nvidia-smi`:

```bash
# atom — start server with MTP disabled (slow, auto-regressive)
cd ~/EXD/projects/serve
./scripts/up.sh qwen3.6-35b-a3b-mtp0
# wait for "Uvicorn running"
```

```bash
# atom terminal 2 — watch GPU utilization
nvidia-smi dmon -s pucv -c 60
#         ^ ^  power, util, clocks, video clocks (-1 = 1 sample every 1s, 60 samples)
```

```bash
# yoneda — long context prefill benchmark
export BASE_URL=http://aitopatom-0a62.local:8000/v1
export MODEL=Qwen/Qwen3.6-35B-A3B
export COMMON_ARGS="--base-url $BASE_URL --api-key EMPTY --model $MODEL --tokenizer $MODEL --latency-mode generation --skip-coherence"

# Run long-context single-user benchmark
llama-benchy $COMMON_ARGS \
  --pp 32768 --tg 512 --depth 0 --concurrency 1 --runs 1
```

While this runs, watch `nvidia-smi` on the atom. Notice the pattern:

```
prefill (60ms):  ████████████  (util = 95-100%, SM = active)
decode (~14s):    ░ ░ ░ ░ ░ ░ ░ ▄  (util = 5-10%, 99% idle!)
```

That 95%→5% drop is the autoregressive bottleneck. Each decode step
launches one matrix multiply → reads result → launches next step. The
GPU spends most time waiting for the round-trip, not computing.

Now run the same benchmark with speculation enabled and compare GPU
utilization during decode. Under speculation, utilization doubles or
triples because more tokens get verified per pass.

### The Gilbert Strang perspective

> "If your matrix-multiply is 100× more expensive than the extra token check,
> why not do MORE work in that single forward pass and get MORE tokens out?"

That's speculative decoding in one sentence.

---

## 2. Mental model — the draft model

```
Standard (auto-regressive):
  Prompt ████████████ → tok ▄ → tok ▄ → tok ▄ → ...
       compute wasted          GPU idle between each step

Speculative (depth=2):
  Prompt ████████████ → ▄ ░ ░ → ▄ ░ ░ → ...
        full matmul        1 full pass = 1+2 spec tokens verified
```

The "draft" model guesses 2 tokens ahead. The full model verifies all
at once. If any guess is wrong, it takes what's correct and discards
the rest — but most of the time, guesses ARE right.

### Simulator

Open `simulator.html` (attached) to step through auto-regressive vs
speculative. Watch the green (correct) and red (wrong) tokens move.

---

## 3. Per-model speculation configs

We test across all three MTP configs:

| Config | Spec Depth | How to Start |
|--------|-----------|---------------|
| `qwen3.6-35b-a3b-mtp0` | 0 | `./scripts/up.sh qwen3.6-35b-a3b-mtp0` |
| `qwen3.6-35b-a3b` | 2 **← default** | `./scripts/up.sh qwen3.6-35b-a3b` |
| `qwen3.6-35b-a3b-mtp3` | 3 | `./scripts/up.sh qwen3.6-35b-a3b-mtp3` |

## 4. Benchmark setup

```bash
# atom — start with NO speculation
./scripts/up.sh qwen3.6-35b-a3b-mtp0
./scripts/logs.sh   # wait for "Uvicorn running"
```

```bash
# yoneda
export BASE_URL=http://aitopatom-0a62.local:8000/v1
export MODEL=Qwen/Qwen3.6-35B-A3B
export COMMON_ARGS="--base-url $BASE_URL --api-key EMPTY --model $MODEL --tokenizer $MODEL --latency-mode generation --skip-coherence"
```

## 5. Single-user benchmark — short context

```bash
# yoneda — run sweep (repeat for mtp0, mtp2, mtp3)
llama-benchy $COMMON_ARGS \
  --pp 512 2048 --tg 256 --depth 0 --concurrency 1 --runs 3 \
  --save-result /tmp/qwen35b-mtpX-single.json --format json
```

**Expected results:**

| Spec Depth | c1 tok/s | c1 TTFT | Notes |
|-----------|---------|---------|-------|
| 0       | 21-23 | ~500ms | No overhead, safe |
| 2       | 28-32 | ~500ms | ~40% faster, best single-user |
| 3       | 28-35 | ~500ms | Sometimes faster, can regress |

## 6. Concurrency sweep

```bash
llama-benchy $COMMON_ARGS \
  --pp 512 --tg 256 --depth 0 --concurrency 1 2 4 --runs 3 \
  --save-result /tmp/qwen35b-mtpX-concurrent.json --format json
```

**Expected:**

| Spec Depth | c1 | c2 | c4 | Notes |
|-----------|----|----|-----|-------|
| 0         | 22 | 44 | 98 | Baseline |
| 2         | 30 | 56 | 112 | ~30% total throughput gain |
| 3         | 28 | 52 | 115 | Higher c4, worse c1 |

**Key:** MTP3 speculation improves batch throughput but hurts single-user
performance → MTP2 is the sweet spot.

## 7. Quality check

Show that MTP0, MTP2, and MTP3 produce **identical** answers:

```bash
for d in 0 2 3; do
  echo "=== Depth=$d ==="
  curl -s http://aitopatom-0a62.local:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3.6-35B-A3B","messages":[{"role":"user","content":"What is 45 * 92?"}],"max_tokens":100}' \
    | jq -r '.choices[0].message.content'
done
```

MTP doesn't change output quality (same model weights). This is NOT a
draft model change. DeepSeek's non-MTP spec DOES change quality — we'll
show that comparison in the bonus segment.

## 8. Bonus — Nemotron-3 (optional)

```bash
# atom — Switch to Nemotron-3 (draft model, native, MTP 3)
./scripts/up.sh nemotron-3-super
```

```bash
# yoneda — same benchmark sweep
llama-benchy $COMMON_ARGS \
  --pp 512 --tg 256 --depth 0 --concurrency 1 --runs 3 \
  --save-result /tmp/nemotron3-mtp3-single.json --format json
```

Compare Qwen vs. Nemotron. The draft model quality matters.

---

## 9. Cleanup

```bash
# atom
cd ~/EXD/projects/serve
./scripts/down.sh
```
