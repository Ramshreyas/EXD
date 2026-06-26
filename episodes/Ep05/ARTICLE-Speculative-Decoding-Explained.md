# Speculative Decoding Explained

Why the GPU wastes 90% of its matrix multiplication, how speculation fixes it, and when "just add more speculation" makes things worse.

---

## The Problem — Autoregressive Decode

Even perfect batching can't fix this: **each step produces exactly one token.**

### The Key Insight

> "If your matrix-multiply is 100x more expensive than the extra token check, why not do MORE work in that single forward pass and get MORE tokens out?"

That's speculative decoding in one sentence.

---

## How Speculation Works

### Vanilla autoregressive (1 token per step)

```
Step 1:  [token_1]  →  model  →  token_2
Step 2:  [token_1, token_2]  →  model  →  token_3
Step 3:  [token_1, token_2, token_3]  →  model  →  token_4
```

The model runs a full forward pass for each single token — the GPU is massively underutilized because the computation is memory-bandwidth-bound.

### Speculative decoding (MTP depth N)

```
Step 1:  Model proposes N+1 tokens in one forward pass.
         [token_1]  →  model  →  [token_2, token_3, token_4]
                 
         All N+1 tokens verified in a single forward pass.
         Accepted tokens are appended immediately.

Result: 3 tokens produced in 2 forward passes instead of 3.
```

The draft head is a lightweight extension of the main model — not a separate model. It adds minimal compute overhead (~5%) while potentially doubling or tripling decode throughput.

---

## The Tradeoff

- **Green tokens** = draft guessed right → free speed
- **Red tokens** = draft guessed wrong → wasted compute
- **Spec depth** = how many tokens ahead the model tries to predict

| Depth | Risk | Reward |
|-------|------|--------|
| 0 (no speculation) | Baseline | Baseline |
| 1 | Low | ~1.3x throughput |
| 2 | Medium | ~1.8x throughput |
| 3 | Higher | ~2x throughput, diminishing returns |

Beyond depth 3, the draft head's accuracy drops and the extra compute isn't worth it.

---

## Benchmark Results

Single-user benchmark comparing MTP depths on Qwen3.6 35B-A3B:

| MTP Depth | tg/s (total) | Mean TTFT | Acceptance Rate |
|-----------|--------------|-----------|-----------------|
| 0 | 44.2 | 510 ms | — |
| 2 | 88.2 | 562 ms | 56.4% |
| 3 | ~95 | ~580 ms | ~50% |

Key finding: MTP2 gives the best balance. MTP3 adds marginal throughput gains with increased TTFT — not worth it for interactive use.

---

## Interactive Simulator

Open the [Speculative Decoding Simulator](https://huggingface.co/spaces/EXD-AI/speculative-decoding-simulator) to see the tradeoff visually:
- Adjust spec depth from 0 to 5
- Watch green (accepted) vs red (rejected) tokens
- See throughput vs latency in real time

---

## Summary

1. Autoregressive decode is inherently memory-bound — each step produces one token
2. Speculative decoding uses a draft head to propose multiple tokens ahead
3. The full model verifies all proposals in a single forward pass
4. MTP2 provides the best balance of throughput gain vs. complexity
5. Beyond MTP3, diminishing returns and increased TTFT make it counterproductive

The sweet spot: MTP depth 2 for most workloads, MTP 0 for latency-sensitive interactive use.
