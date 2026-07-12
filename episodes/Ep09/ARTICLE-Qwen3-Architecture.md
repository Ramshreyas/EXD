# Qwen3.6-35B-A3B — Architecture Overview

Open the [interactive architecture diagram](https://huggingface.co/spaces/EXDai/qwen3-architecture) alongside this article. Click each box as we go — the annotations panel updates with details for each component.

---

## What We're Looking At

Qwen3.6-35B-A3B is a **hybrid Mixture-of-Experts** language model. 35 billion total parameters, but only 3 billion are active per token. 40 layers, 256 experts, 256K context window, multi-modal.

At a glance:

| Parameter | Value |
|-----------|-------|
| Total params | ~35B |
| Active per token | ~3B |
| Hidden dim | 2048 |
| Layers | 40 (30 GDN + 10 Full) |
| Q heads / KV heads | 16 / 2 (GQA 8:1) |
| Total experts | 256 (8 active) |
| Context | 262K tokens |
| Vocab | 248,320 |

---

## Stage 1: Input Pipeline

The model never sees text — it sees integers.

**BPE Tokenizer** splits input into subword tokens from a vocabulary of 248,320. That's massive — GPT-2 had 50K. This large vocabulary handles 119 languages and includes special tokens for images (`248053`–`248057`) and video.

**Embedding Lookup** maps each token ID to a 2048-dimensional vector. The lookup table is `248,320 × 2048` — **509 million parameters** just for the input embedding. It's **untied** from the output embedding, so the LM head is another 509M.

**mRoPE** — Multi-resolution Rotary Position Embedding — adds position information. Only **25%** of each head dimension (64 of 256) receives rotary encoding. The remaining 192 dims stay position-free, encoding pure content. The three sections `[11, 11, 10]` encode temporal (T), height (H), and width (W) — this is 3D position encoding, built for multi-modal input where visual tokens have spatial and temporal positions.

💡 *This stage is covered in detail in Ep07 (Tokenization & Embeddings) and Ep08 (Positional Embeddings).*

---

## Stage 2: Transform Blocks ×40

This is where the computation happens. 40 identical blocks, each with two sub-blocks:

```
RMSNorm → Attention Mixer → +residual → RMSNorm → MoE FFN → +residual
```

Two residual connections preserve gradient flow. Pre-LN (LayerNorm before each sub-block) for training stability.

### Attention Mixer: Hybrid Design

The attention type varies by layer index:

- **30 layers**: **Gated DeltaNet** — linear recurrence, `O(n)` time, `O(1)` memory
- **10 layers**: **Full softmax attention** — every 4th layer (indices 3, 7, 11, ..., 39)

This hybrid design is the key architectural innovation. Full attention is `O(n²)` — prohibitive for 256K context on all 40 layers. Gated DeltaNet handles local-to-medium-range context efficiently via a recurrent state update, while the 10 full-attention layers provide unrestricted global token mixing.

#### Full Attention (10 layers)

Uses **Grouped Query Attention** with 16 query heads and only 2 key-value heads — an **8:1 compression** that cuts KV cache memory by 8×. Head dimension is **256** (double the typical 128) for more expressive attention per head.

The Q projection is special — it outputs **double the expected size**. The first half is the actual query; the second half becomes an element-wise **gate** applied after attention:

```
attn_output = softmax(QK^T / √256) × V
attn_output = attn_output × sigmoid(gate)    ← learned gating
```

**QK-Norm** applies RMSNorm per head before RoPE, preventing attention entropy collapse during training.

#### Gated DeltaNet (30 layers)

***Not Mamba.*** Gated DeltaNet is a different linear attention mechanism using a **gated delta rule** recurrence:

- Project input to query (16 heads × 128 dims), key (16 heads × 128), value (32 heads × 128), and gate z (32 heads × 128)
- Apply a **causal conv1d** (kernel=4) for local context mixing
- Run the delta recurrence: `S_t = (1-β)S_{t-1} + β v k^T`, output `o_t = S_t q`
- Gate output via z projection using RMSNorm-gating
- Two implementations: **recurrent** (single-token decode) and **chunked** (parallel prefill)

Result: `O(n)` time and `O(1)` memory per layer — critical for 256K sequences.

### MoE FFN: 256 Experts

Every layer has **exactly the same** MoE FFN structure, regardless of attention type.

**Router**: A learned linear layer `2048 → 256` scores all 256 experts for each token. Softmax, then **top-8** are selected and their weights renormalized to sum to 1.

**Experts**: Each of the 256 experts is a tiny **SwiGLU** feed-forward network:

```
expert_e(x) = SiLU(x · W_gate_e) ⊙ (x · W_up_e) · W_down_e
```

Gate and up projections are merged into a single parameter tensor `[256, 1024, 2048]` for efficiency. Each expert has intermediate dimension 512 and output dimension 2048 — just **3.15M parameters** per expert. Only 8 fire per token, so **25.2M active expert parameters** out of 806M total.

**Shared Expert**: Same SwiGLU structure, but always active for every token. Controlled by a learned scalar gate: `sigmoid(Linear(x, 1)) × shared(x)`. Handles common patterns so routed experts can specialize.

**Load Balancing**: An auxiliary loss (coefficient `0.001`) penalizes uneven expert utilization, ensuring all 256 experts get used across the batch.

💡 *The sparse MoE design is why 35B total params → only 3B active per token. The model has vast capacity but uses it selectively.*

---

## Stage 3: Output

After the 40th transform block:

1. **Final RMSNorm** normalizes the hidden state
2. **LM Head** — linear projection `2048 → 248,320` — produces raw logits over the vocabulary. Untied from the input embedding: another **509M parameters**.
3. **Sampling** — softmax → greedy, top-p, or temperature sampling → next token
4. **MTP Head** (optional) — an extra layer that predicts **token₂** from the same hidden state, enabling self-speculative decoding: the model acts as its own draft model, producing 2 tokens per forward pass.

---

## What Changed from Qwen3

The [Qwen3 technical report](https://arxiv.org/abs/2505.09388) describes the predecessor (Qwen3-30B-A3B). Qwen3.6-35B-A3B introduces significant architectural evolution:

| Aspect | Qwen3-30B-A3B | Qwen3.6-35B-A3B |
|--------|---------------|------------------|
| Total params | 30B | ~35B |
| Layers | 48 | 40 |
| Attention | Full only (48 layers) | **Hybrid**: 30 linear + 10 full |
| Total experts | 128 | **256** |
| Shared expert | ❌ Removed | ✅ Added back |
| Head dim | 128 (typical) | **256** |
| Context | 128K | **262K** |
| RoPE | Standard | **mRoPE** 3D |
| MTP | ❌ | ✅ |
| Vision | ❌ | ✅ ViT encoder |

---

## Next Steps

This overview covered the **what** — the architecture. Upcoming episodes will cover the **how**:

- **Forward hooks**: trace a real token through every layer with PyTorch hooks
- **Deep dives**: full attention internals, Gated DeltaNet recurrence, MoE routing dynamics
- **Training**: the thinking/non-thinking mode integration, distillation pipeline

Open the [interactive diagram](https://huggingface.co/spaces/EXDai/qwen3-architecture) and click through each box — the step-by-step annotations are your recording guide.
