# Positional Embeddings

How transformers know where words are in a sequence — from sinusoidal encodings to RoPE.

---

## The Problem

Transformers are permutation-invariant. Unlike RNNs, which process tokens sequentially and inherently know their order, a transformer's attention mechanism has no built-in notion of position.

```
"The cat sat on the mat"
"mat the on sat cat The"

Both sequences would produce the same attention patterns 
without positional information.
```

Positional embeddings solve this by injecting position information into each token's representation before it enters the transformer layers.

---

## Sinusoidal Positional Encodings

The original "Attention Is All You Need" approach. Each position gets a fixed encoding based on sine and cosine functions of different frequencies:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Key properties:
- **Deterministic**: No learned parameters, just a fixed formula
- **Extrapolatable**: Can handle positions beyond training max length
- **Relative signal**: The dot product of two position encodings depends only on their offset, not absolute positions

---

## Learned Positional Embeddings

Used by BERT and early GPT models. Each position gets a learnable embedding vector, just like token embeddings.

```
Position 0 → [p₀, p₀, ..., p₀]     # learned vector
Position 1 → [p₁, p₁, ..., p₁]     # learned vector
Position 2 → [p₂, p₂, ..., p₂]     # learned vector
...
```

Pros:
- More flexible — model can learn position semantics
- Simpler to implement

Cons:
- Cannot extrapolate beyond training max length
- Requires training data to learn meaningful position representations

---

## RoPE (Rotary Position Embedding)

The modern standard, used by Llama, Mistral, Qwen, and most current models.

**Core idea:** Instead of adding a position vector to each token, rotate the token's query and key vectors by an angle proportional to their position.

```
q_m = R(m) · q    # rotate query by position m
k_n = R(n) · k    # rotate key by position n

Attention(q_m, k_n) = q^T · R(n-m) · k
```

The dot product between a query at position m and a key at position n depends only on their relative offset (n-m), not absolute positions.

Key advantages:
- **Relative attention**: Attention naturally decays with distance
- **Extrapolatable**: Works at sequence lengths beyond training
- **No additional parameters**: Rotary transforms are deterministic
- **Theoretically grounded**: Preserves the dot product structure

### RoPE Frequency Bands

RoPE uses multiple rotation frequencies, similar to sinusoidal encodings:

| Frequency band | What it captures |
|----------------|-----------------|
| High (fast rotation) | Nearby positions, fine-grained ordering |
| Medium | Phrase-level structure |
| Low (slow rotation) | Long-range dependencies, document structure |

The model learns which frequency bands matter for different attention heads. Some heads focus on local context (high frequencies), others on global dependencies (low frequencies).

---

## ALiBi (Attention with Linear Biases)

An alternative to RoPE used in some models (BLOOM, MPT).

**Core idea:** Instead of modifying query/key representations, add a bias term directly to the attention scores based on token distance.

```
score(q_m, k_n) = q_m · k_n / sqrt(d) - m · |m - n|

# Where m is a head-specific slope hyperparameter
```

Advantages:
- Extremely simple — no position vectors or rotations
- Naturally biases toward local context
- Strong extrapolation to long sequences

Disadvantages:
- Less flexible than RoPE — the bias is hardcoded
- Can't adjust per-layer or per-head behavior

---

## Interactive Notebook

The full notebook with visualizations of RoPE rotations, frequency band analysis, and comparisons between different positional encoding schemes is available [here](https://huggingface.co/datasets/EXD-AI/episode-08-positional-embeddings).

---

## Summary

| Method | Type | Extrapolates? | Used by |
|--------|------|---------------|---------|
| Sinusoidal | Fixed, additive | Yes | Original Transformer |
| Learned | Trainable, additive | No | BERT, GPT-2 |
| RoPE | Fixed, rotary | Yes | Llama, Mistral, Qwen |
| ALiBi | Fixed, bias | Yes | BLOOM, MPT |

RoPE is the current standard — it combines the best of both worlds: relative attention, extrapolation, and no learned parameters. Most models released in 2024-2026 use RoPE with various frequency scaling schemes for long-context support.
