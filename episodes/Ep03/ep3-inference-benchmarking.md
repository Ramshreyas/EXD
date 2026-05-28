# Ep. 3 — Inference Benchmarking (Intro)

> Build a mental model of what inference actually *is*, then benchmark it.
> Start vanilla, measure, tweak one knob, measure again.

---

## 1. What is inference? — the toy model

Don't think about attention or transformers yet. Here's the simplest
possible model of what happens when you send a prompt to an LLM.

Say the user types: `"The cat sat"`

```
User input: "The cat sat"
        │
        ▼
   ┌──────────┐
   │ Tokenize │   "The cat sat"  →  [576, 3797, 7236]
   └──────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  PREFILL (happens once)                     │
│                                             │
│  Run ALL 3 tokens through the model in      │
│  one shot. Every token attends to every     │
│  other token. The model produces the first  │
│  output token AND saves its internal state  │
│  (the KV cache) for later reuse.            │
│                                             │
│  Compute-bound: lots of matrix multiplies.  │
└─────────────────────────────────────────────┘
        │
        ▼  first token: "on"
        │
        ▼
┌─────────────────────────────────────────────┐
│  DECODE (loops until <eos> or max_tokens)   │
│                                             │
│  Step 1: "on" is appended to the input.     │
│          Input is now [576, 3797, 7236, 389]│
│                                             │
│  Step 2: Run through the model. Only the    │
│          NEW token needs fresh computation. │
│          All previous tokens reuse their    │
│          cached KV entries.                 │
│                                             │
│  Step 3: Model outputs next token: "the"    │
│                                             │
│  Step 4: Append "the", repeat from Step 1.  │
│                                             │
│  Memory-bound: spends most time reading KV  │
│  cache entries from previous tokens.        │
└─────────────────────────────────────────────┘
        │
        ▼
   ┌─────────────┐
   │ Detokenize  │   [389, 278, 3098, ...]  →  "on the mat."
   └─────────────┘
        │
        ▼
   Final output: "The cat sat on the mat."
```

**How the KV cache actually works** — no transformer math, just the idea:

During prefill, the model saves a "summary" of each token it reads — two
vectors called the Key (K) and Value (V). Think of these as the model's
scratch notes on each word.

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

  Next token "the":
    → compute [K₅ V₅] from "the" itself    <-- still cheap (1 token)
    → read   [K₁..K₄, V₁..V₄] from cache   <-- reading from memory
    → the cache grows by one entry each step
```

Without the KV cache, every decode step would re-process the entire
growing sequence: 3 tokens, then 4, then 5... Generation would get
slower with every token. The cache makes decode O(n) instead of O(n²).

But there's a catch: the cache *grows* with every token, so eventually the
GPU spends more time reading it than computing. That's why decode is
**memory-bound**.

### Prefill vs Decode — the key insight

| | Prefill | Decode |
|---|---|---|
| What runs | Entire prompt at once | One new token at a time |
| How many times | 1 | N (once per output token) |
| Bottleneck | Compute (matrix multiplies) | Memory (KV cache reads) |
| GPU behaviour | High utilization, worth optimizing | Low utilization, waiting on bandwidth |

This is why benchmarks report separate numbers for each phase:

- **prefill tokens/s** — how fast the model ingests your prompt
- **decode tokens/s** — how fast it generates the response

And two latency numbers:

- **TTFT** (time to first token) — how long before the first word appears
  (dominated by prefill)
- **TPOT** (time per output token) — how long each subsequent word takes
  (dominated by decode / memory bandwidth)

---

## 2. The model we're benchmarking

Qwen3.6 35B-A3B — a Mixture-of-Experts model:
- **35B total** parameters, but only **3B active** per token
- MoE means not all parameters fire at once — cheaper inference
- Supports MTP (Multi-Token Prediction) for faster decode
- 128k context window

It's already pulled on atom from Ep 2:

```bash
# atom
hf cache ls
```

---

## 3. Start with vanilla flags

First, let's see what the baseline config looks like — no special tuning:

```bash
# atom
cat ~/EXD/projects/serve/configs/qwen3.6-35b-a3b-baseline.env
```

The key flags to notice:
- `GPU_MEM_UTIL=0.80` — conservative
- `MTP 2` speculative tokens — basic multi-token prediction
- No explicit scheduler limits

Bring it up:

```bash
# atom
cd ~/EXD/projects/serve
./scripts/up.sh qwen3.6-35b-a3b-baseline
```

Wait for it (first time loads the model — ~30-60 seconds):

```bash
# atom
./scripts/logs.sh
# Look for: "Uvicorn running on http://0.0.0.0:8000"
# Ctrl+C once you see it
```

Smoke test:

```bash
# atom
./scripts/test.sh
```

### Check that Yoneda can reach it

```bash
# yoneda
curl -s http://aitopatom-0a62.local:8000/v1/models
```

```bash
# yoneda
curl -s http://aitopatom-0a62.local:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.6-35B-A3B",
    "messages": [{"role": "user", "content": "One sentence: what is prefill in LLM inference?"}],
    "max_tokens": 300, "temperature": 0
  }'
# Qwen3.6 is a reasoning model — it thinks internally (the "reasoning" field)
# before producing content. 50 tokens wasn't enough to finish the thought.
# 300 gives it room to reason AND respond.
```

---

## 4. vLLM's built-in benchmark

vLLM ships with `vllm bench serve` — a quick way to measure throughput
without extra tooling. We'll run it from *inside* the Docker container on
atom (so there's no network overhead):

### Smoke run — quick sanity check

```bash
# atom
docker exec vllm vllm bench serve \
  --backend vllm \
  --model Qwen/Qwen3.6-35B-A3B \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 20 \
  --request-rate 1.0
```

What this does:
- Generates 20 random prompts (512 tokens each)
- Asks for 128 tokens of output each
- Sends them at 1 request/second
- Reports TTFT, TPOT, throughput

### Scale it up — more load

```bash
# atom
docker exec vllm vllm bench serve \
  --backend vllm \
  --model Qwen/Qwen3.6-35B-A3B \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 50 \
  --request-rate inf
```

`--request-rate inf` sends all requests at once — maximum pressure. Watch
the output for:
- `Request throughput` — requests/second the server handled
- `Output token throughput` — tokens/second being generated
- `Mean TTFT` — average time to first token
- `Mean TPOT` — average time per output token

These numbers are our baseline. Write them down (or screenshot the output).

---

## 5. llama-benchy — a richer benchmark

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
  --no-warmup \
  --format json
```

What the flags mean:
- `--pp 128`: prefill prompt of 128 tokens
- `--tg 256`: target generation of 256 tokens
- `--depth 0`: no additional context before the prompt (simple case)
- `--concurrency 1`: single user
- `--runs 1`: one pass (quick sanity)

### Short-context sweep

This is our main benchmark — varies prefill length and concurrency:

```bash
# yoneda
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

Key numbers to watch:
- `tg t/s` (tokens generated / second) — decode throughput per user
- `tg t/s (total)` — aggregate across all concurrent users
- `e2e_ttft` — end-to-end time to first token (includes prefill)
- `peak tg t/s` — peak decode speed during the benchmark

Take note of these baseline numbers. We'll compare after the next step.

---



## 6. What we just did

```
1. Built a mental model of inference: prefill vs decode, TTFT vs TPOT
2. Served Qwen3.6 35B-A3B with the baseline config
3. Ran vLLM's built-in bench for a quick local throughput check
4. Ran llama-benchy from Yoneda for a proper end-to-end sweep
```

That's it. No tuning, no flags — just two benchmarks against the same
server. The point was to get comfortable with the tools and build
intuition for what the numbers mean.

### What's next

Ep 4 — "Performance Tuning." We'll:

- Introduce concurrency and batching properly
- Sweep vLLM flags (MTP depth, GPU memory, scheduler limits)
- Measure long-context prefill and prefix caching
- Compare results side-by-side and pick a winning config

---

## Cleanup

```bash
# atom
cd ~/EXD/projects/serve
./scripts/down.sh
```

