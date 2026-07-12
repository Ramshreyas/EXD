# Ep09 Shownotes — Qwen3.6-35B-A3B Architecture Walkthrough

> Recording guide. Open the diagram at https://huggingface.co/spaces/EXDai/qwen3-architecture. Click each box in the order below and use these notes to explain.

---

## Tab 1: High-Level Architecture (left → right flow)

### 1. Input Tokens
> "This is where it starts. Raw text comes in. The model never sees characters — it sees token IDs. That's integers. The tokenizer handles this conversion, which we'll look at next."

- Mention: Ep07 covered tokenization in detail.
- Numbers: none yet, just the concept.

---

### 2. BPE Tokenizer
> "Byte-Pair Encoding. The Qwen3.6 tokenizer has a vocabulary of 248,320 tokens. That's massive — for comparison, GPT-2 had 50K. This large vocab handles 119 languages and special tokens for images and video."

- **248,320** vocab size
- **BPE** — subword tokenization
- Special tokens: BOS/EOS at 248044, vision tokens 248053–248057
- Connection: Ep07 showed BPE in action on a Qwen tokenizer.

---

### 3. Embedding Lookup
> "Each token ID becomes a 2048-dimensional vector. It's a lookup table — index in, row out. 248,320 rows by 2048 columns — that's 509 million parameters just for the embedding. And it's untied — the output embedding is separate, another 509 million."

- **248,320 × 2048 = 509M params**
- **Untied** — input and output embeddings are separate
- Connection: Ep07 covered embedding spaces. This is the same concept, different model.

---

### 4. mRoPE
> "Multi-resolution Rotary Position Embedding. This is how the model knows token order. Only 25% of each head's dimensions get rotary encoding — the remaining 75% stay position-free for pure content. The three sections [11, 11, 10] encode temporal, height, and width — this is a 3D position encoding, built for multimodal input."

- **25% partial** (64 of 256 dims are rotary)
- **Sections [11, 11, 10]** — T, H, W
- **Interleaved** THWTHW... pattern
- **RoPE θ = 10,000,000**
- Connection: Ep08 covered sinusoidal → learned → RoPE. mRoPE is the 3D extension.

---

### 5. RMSNorm (pre-attention)
> "Root Mean Square normalization. Before every attention and FFN block, we normalize. RMSNorm is simpler than LayerNorm — no mean subtraction, just scaling by root-mean-square. This is the standard pre-norm transformer pattern: normalize, compute, add residual."

- **2048 dims** — applied per-token
- **Pre-norm** architecture — norm before sub-block

---

### 6. Attention
> "The attention mixer. Thirty layers use Gated DeltaNet — a linear recurrence that's O(n) instead of O(n²). Ten layers — every fourth — use full softmax attention with Grouped Query Attention. 16 query heads but only 2 key-value heads — an 8-to-1 ratio that cuts the KV cache by 8x. Head dimension is 256 — double the typical 128 — for more expressive attention."

- **30 GDN + 10 Full** — hybrid design
- **GQA 16:2** — 8× KV cache reduction
- **Head dim 256** — unusual, double the standard
- **QK-Norm** — RMSNorm per head
- **Output gate** — built into q_proj's doubled output
- Connection: We'll deep-dive both attention types in Tab 2.

---

### 7. RMSNorm (pre-FFN)
> "Second normalization, same structure. This comes after the attention residual and before the MoE FFN. Standard Pre-LN transformer."

---

### 8. MoE FFN
> "This is the big one. Every layer has a Mixture-of-Experts feed-forward network. A learned router — just a linear layer from 2048 to 256 — picks the top 8 experts out of 256 for each token. Each expert is a SwiGLU — SiLU-gated linear layers. There's also a shared expert that's always active, gated by a learned scalar. So you get the shared expert plus your top-8 routed experts."

- **256 experts total, 8 active per token** — only 3.1% of expert params used
- **Router**: Linear(2048 → 256), softmax, top-8, renormalize
- **Expert**: SwiGLU, intermediate 512, 3.15M params each
- **Shared expert**: same structure, scalar gate, always active
- **Load balancing loss**: coefficient 0.001
- Connection: This is why 35B total params but only 3B active. The experts are sparse.

---

### 9. Final RMSNorm
> "Last normalization. Applied to the output of the 40th transform block."

---

### 10. LM Head
> "Language Model head. Projects from 2048 dimensions back to the full 248,320 vocabulary. Untied from the input embedding — separate parameter matrix, another 509 million parameters. This gives us raw logits over every possible token."

- **2048 → 248,320** — 509M params
- **Untied** — separate from input embedding

---

### 11. Next Token
> "Logits go through softmax, then we sample — could be greedy argmax, top-p, temperature. The sampled token gets fed back as input for the next step. This is autoregressive generation."

---

### 12. MTP Head (optional)
> "Multi-Token Prediction. An extra layer that predicts not just the next token but the one after that too — token t+2 from the same hidden state. This enables self-speculative decoding. The model acts as its own draft model, producing two tokens per forward pass instead of one."

- **1 MTP layer**, no dedicated embeddings
- Self-speculative decoding connection to Ep05
- Used in production inference but optional

---

## Tab 2: Transform Block Detail (click "Transform Block Detail" tab)

### 13. RMSNorm (pre-attention)
> "Same as before. Normalize the 2048-dimensional hidden state. Input shape is batch × sequence × 2048."

---

### 14. Full Attention
> "Ten layers use this — every fourth layer. The Q projection is special — it outputs double the expected size. The first half is the actual query, the second half becomes a gate. After attention, multiply the output by sigmoid of that gate before the output projection. Key and value are 2 heads each — that's the GQA compression. And QK-Norm applies RMSNorm per head before RoPE, which stabilizes training."

- **Q: 16 hd × 256 × 2** — doubled for gate
- **K: 2 hd × 256**
- **V: 2 hd × 256**
- **GQA 16:2** — KV cache 8× smaller
- **QK-Norm** — RMSNorm per head
- **mRoPE** — 25% partial (64/256 dims)
- **Gate**: sigmoid(gate) × attn_output, built into q_proj

---

### 15. Gated DeltaNet
> "Thirty layers use this. Not Mamba — Gated DeltaNet is a different linear attention mechanism. Instead of computing a full attention matrix, it runs a recurrent state update. The state S_t is updated with a gated outer product of key and value, then multiplied by the query. Key heads: 16 times 128 dims. Value heads: 32 times 128 dims. A small conv1d with kernel 4 provides local context before the recurrence. Two modes: recurrent for single-token decode, chunked for parallel prefill. All O(n) time."

- **Key: 16 hd × 128**
- **Value: 32 hd × 128**
- **Conv1d**: kernel=4, depthwise causal
- **Delta rule**: S_t = (1-β)S_{t-1} + β v k^T, output = S_t q
- **O(n) time, O(1) memory** per layer
- **Output gate**: z projection → RMSNorm-gated

---

### 16. O-projection + Residual
> "Attention output — 4096 dimensions — gets projected back to 2048. Then the residual connection adds the original input. This skip connection is what makes deep networks trainable — it provides a direct identity path so gradients don't vanish."

- **O-proj**: 4096 → 2048
- **x = x + attn_output** (residual)

---

### 17. RMSNorm (pre-MoE)
> "Second normalization. Prepares the representation for the expert routing."

---

### 18. Router
> "The router is just a learned linear layer: 2048 to 256, one weight per expert. Softmax over all 256 experts, then we take the top 8. Those 8 weights are renormalized to sum to 1. The router is trained with an auxiliary loss that penalizes uneven expert usage — this prevents all tokens from routing to the same few experts."

- **Linear(2048 → 256)**
- **Softmax → top-8 → renormalize**
- **Load balancing loss**: coeff 0.001

---

### 19. Expert SwiGLU ×8
> "Each selected expert is a tiny feed-forward network. Gate and up projections are merged into a single tensor — 256 experts × (512+512) × 2048. The input is projected, split into gate and up, gate goes through SiLU, multiplied elementwise with up, then down-projected back to 2048. Eight of these run per token, their outputs weighted by the router scores and summed."

- **SwiGLU**: SiLU(xW_gate) ⊙ (xW_up) × W_down
- **Merged gate_up_proj**: [256, 1024, 2048]
- **down_proj**: [256, 2048, 512]
- **3.15M params per expert**
- **8 × 3.15M = 25.2M active expert params per token**

---

### 20. Shared Expert
> "Always active, same SwiGLU structure. But with a twist — a learned scalar gate per token. Linear from 2048 to 1, through sigmoid, multiplied by the expert output. This lets the model decide how much shared knowledge to mix in. The shared expert handles common patterns so the routed experts can specialize."

- **Same SwiGLU** (2048 → 512 → 2048)
- **Scalar gate**: sigmoid(Linear(x, 1))
- **3.15M params**

---

### 21. + Residual
> "Final residual. The MoE output gets added to the input that came into the FFN sub-block. Together with the attention residual, this completes the Pre-LN transformer pattern: norm → compute → add. The block output is the same shape it came in — batch × sequence × 2048."

- **x = x + moe_output**
- **Output shape**: [batch, seq, 2048] — same as input

---

## Recording Flow

```
OPEN DIAGRAM
  → "This is the Qwen3.6-35B-A3B architecture."
  → Explain split layout (left=diagram, right=annotations)

TAB 1: HIGH-LEVEL
  → Walk left → right: Input → Transform → Output
  → Click 1 → 2 → 3 → 4 (input pipeline)
  → "We covered these in Ep07 and Ep08."
  → Click 5 → 6 → 7 → 8 (transform block)
  → "This is the core — let's zoom in."
  → Click 9 → 10 → 11 → 12 (output)
  → "That's the full pipeline."

SWITCH TO TAB 2
  → "Now let's look inside one transform block."
  → Click 13 → 14 (Full Attention detail)
  → "Ten layers use this. Note the doubled Q projection..."
  → Click 15 (Gated DeltaNet)
  → "Thirty layers use this. It's a linear recurrence..."
  → Click 16 (O-proj + residual)
  → Click 17 → 18 → 19 → 20 → 21 (MoE FFN)
  → "The router, experts, shared expert. This is the MoE."
  → "256 experts, only 8 fire per token. That's sparsity."

WRAP UP
  → "Next episode: we step through the actual PyTorch code with forward hooks."
  → "We'll trace a token through this exact pipeline layer by layer."
```

## Key Numbers Sheet (quick reference)

| Parameter | Value |
|-----------|-------|
| Total params | ~35B |
| Active per token | ~3B |
| Hidden dim | 2048 |
| Layers | 40 (30 GDN + 10 Full) |
| Q heads / KV heads | 16 / 2 (GQA 8:1) |
| Head dim | 256 |
| Total experts | 256 |
| Active experts | 8 |
| Expert params each | 3.15M |
| Shared expert | 3.15M |
| Embedding params | 509M (untied) |
| LM Head params | 509M (untied) |
| Context length | 262,144 |
| Vocab size | 248,320 |
| RoPE theta | 10,000,000 |
| Partial rotary | 25% (64/256) |
| mRoPE sections | [11, 11, 10] |
| Conv1d kernel | 4 |
| GDN key heads | 16 × 128 |
| GDN value heads | 32 × 128 |
| Load balance coeff | 0.001 |
| MTP layers | 1 |
