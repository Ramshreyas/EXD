# Qwen3.6-35B-A3B Architecture

> Source of truth: Hugging Face [`config.json`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json)
> Reference: [Qwen3 Technical Report (arXiv:2505.09388)](2505.09388v1.pdf) — covers original Qwen3 family; Qwen3.6 has significant architectural evolution.

---

## 1. Model Card

| Property | Value |
|----------|-------|
| **Architecture** | Mixture-of-Experts (MoE) + Hybrid Attention |
| **Total params** | ~35B (34.2B text + vision encoder + MTP head) |
| **Activated params per token** | ~3B |
| **Hidden dim** (`hidden_size`) | 2048 |
| **Layers** (`num_hidden_layers`) | 40 |
| **Q heads / KV heads** | 16 / 2 (GQA, ratio 8:1) |
| **Head dim** (`head_dim`) | 256 |
| **Total experts** (`num_experts`) | 256 |
| **Active experts per token** (`num_experts_per_tok`) | 8 |
| **Shared expert** | ✅ Yes (intermediate size 512) |
| **Expert intermediate size** | 512 (SwiGLU: gate, up, down) |
| **Attention type** | Hybrid: 30 linear (Gated DeltaNet) + 10 full attention |
| **Full attention interval** | Every 4th layer (indices 3, 7, 11, ..., 39) |
| **Position encoding** | mRoPE (3D, multi-resolution, `[11, 11, 10]` sections) |
| **RoPE theta** | 10,000,000 |
| **Partial rotary factor** | 0.25 |
| **Context length** | 262,144 (256K tokens) |
| **Activation** | SiLU (SwiGLU in FFN) |
| **Normalization** | RMSNorm (pre-norm), QK-Norm on attention |
| **Vocab size** | 248,320 |
| **Tied embeddings** | No |
| **MTP** | 1 layer (Multi-Token Prediction) |
| **Attention output gate** | Yes (`attn_output_gate: true`) |
| **Multi-modal** | Vision encoder (27-layer ViT, 1152 hidden, patch_size=16, temporal_patch_size=2) |

---

## 2. High-Level Block Diagram

```
Input text
    │
    ▼
┌─────────────────────┐
│   Tokenizer (BPE)   │  vocab_size = 248,320
│   token → ID        │
└─────────┬───────────┘
          │ token_ids.shape = [batch, seq_len]
          ▼
┌─────────────────────┐
│   Embedding Layer   │  lookup table: hidden_size = 2048
└─────────┬───────────┘
          │ embeddings.shape = [batch, seq_len, 2048]
          ▼
┌─────────────────────┐
│   mRoPE (3D RoPE)   │  partial_rotary_factor = 0.25
│   sections: [11,11,10]│  applied to first 25% of head dim
└─────────┬───────────┘
          │
          ▼  × 40 layers (hybrid pattern)
    ┌─────────────────┐
    │  Hybrid Layer i  │
    │                  │
    │  ┌───────────┐   │
    │  │  RMSNorm   │   │  pre-attention norm
    │  └─────┬─────┘   │
    │        │         │
    │  ┌─────┴─────┐   │
    │  │ ATTENTION │   │  see §3A or §3B
    │  │  (varies) │   │
    │  └─────┬─────┘   │
    │        │         │
    │  ┌─────┴─────┐   │
    │  │  + input  │   │  residual connection
    │  └─────┬─────┘   │
    │        │         │
    │  ┌─────┴─────┐   │
    │  │  RMSNorm   │   │  pre-FFN norm
    │  └─────┬─────┘   │
    │        │         │
    │  ┌─────┴─────┐   │
    │  │  MoE FFN  │   │  256 experts, top-8 + shared
    │  │  + Router │   │
    │  └─────┬─────┘   │
    │        │         │
    │  ┌─────┴─────┐   │
    │  │  + input  │   │  residual connection
    │  └───────────┘   │
    └─────────────────┘
          │
          ▼
┌─────────────────────┐
│   Final RMSNorm     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   LM Head (linear)  │  hidden_size → vocab_size (untied)
└─────────┬───────────┘
          │ logits.shape = [batch, seq_len, vocab_size]
          ▼
┌─────────────────────┐
│   MTP Head (+1 tok) │  optional: predict next+1 token
└─────────────────────┘
```

---

## 3. The 40 Hybrid Layers

Attention type alternates in a repeating pattern of **4**:

| Layer indices | Attention type | Count |
|---|---|---|
| 0, 1, 2 | Linear (Gated DeltaNet) | 3 |
| **3** | **Full (softmax) attention** | **1** |
| 4, 5, 6 | Linear (Gated DeltaNet) | 3 |
| **7** | **Full (softmax) attention** | **1** |
| ... | ... repeats every 4 layers ... | ... |
| 36, 37, 38 | Linear (Gated DeltaNet) | 3 |
| **39** | **Full (softmax) attention** | **1** |

**Summary**: 30 linear attention layers, 10 full attention layers (at positions 3, 7, 11, 15, 19, 23, 27, 31, 35, 39).

This is the **hybrid attention** design — full softmax attention for global context mixing, Gated DeltaNet linear attention for efficient recurrent processing.

---

## 3A. Full Attention Block (10 layers)

```
Input: x.shape = [batch, seq_len, 2048]
  │
  ├── Q + Gate Projection (combined)
  │   q_proj: Linear(2048 → 16×256×2=8192)     double output!
  │   Split: [query_states, gate] = chunk(output, 2, dim=-1)
  │   query_states.shape = [batch, seq, 16, 256]
  │   gate.shape = [batch, seq, 4096]
  │   No bias (attention_bias=false)
  │
  ├── K Projection
  │   k_proj: Linear(2048 → 2×256=512)          num_key_value_heads=2
  │
  ├── V Projection
  │   v_proj: Linear(2048 → 2×256=512)
  │
  ├── QK-Norm (RMSNorm per head)
  │   Applied to query_states and key_states on head_dim
  │   Prevents attention entropy collapse
  │
  ├── mRoPE (partial: 25% of head_dim = 64 dims)
  │   Applied to Q and K. Multi-resolution (3D).
  │   Sections: [11, 11, 10] → interleaved T/H/W
  │
  ├── GQA Attention
  │   16 Q heads share 2 KV heads (8:1 grouping)
  │   attn = softmax(Q @ K^T / sqrt(256)) @ V
  │
  ├── Attention output gate (built into q_proj)
  │   attn_output = attn_output * sigmoid(gate)    ← element-wise
  │   No separate gate layer — gate from q_proj's 2× output
  │
  ├── Output projection
  │   o_proj: Linear(4096 → 2048)
  │
  └── Residual: x = x + attn_output
```

### Why GQA 8:1?
Standard multi-head attention would have 16 KV heads. GQA compresses to 2 KV heads shared across 8 Q heads each. This reduces KV cache size by **8×** — critical for 256K context length.

### Why head_dim=256?
Contemporary models typically use head_dim=128. Doubling to 256 means each attention head has more expressive capacity. Combined with only 16 Q heads, the total attention computation is `16 × 256 = 4096` which matches `hidden_size`.

### Why partial RoPE (25%)?
Only the first 64 of 256 head dimensions get rotary encoding. The remaining 192 dims are left as "position-free" — they can encode content information without position interference. This is a known technique for long-context models (cf. Llama 3's 50% RoPE).

---

## 3B. Linear Attention Block — Gated DeltaNet (30 layers)

> NOT Mamba SSM. This is **Gated DeltaNet**, a different linear attention mechanism using a gated delta rule with recurrent and chunked variants.

```
Input: x.shape = [batch, seq_len, 2048]
  │
  ├── Combined Projections
  │   in_proj_qkvz: Linear(2048 → key_dim×2 + value_dim×2)
  │     key_dim = 16×128 = 2048, value_dim = 32×128 = 4096
  │     Total: 2×2048 + 2×4096 = 12288
  │   in_proj_ba:   Linear(2048 → 2×32 = 64)
  │     b, a = beta and alpha for the delta rule
  │
  ├── Split: [query, key, value, z] and [b, a]
  │   query.shape = [batch, seq, 16 kv_head_groups, 128]
  │   key.shape   = [batch, seq, 16 kv_head_groups, 128]
  │   value.shape = [batch, seq, 32 value_heads, 128]
  │   z.shape     = [batch, seq, 32 value_heads, 128]  (gate for output)
  │   b.shape     = [batch, seq, 32]   (beta — gating for delta rule)
  │   a.shape     = [batch, seq, 32]   (alpha — input for discretization)
  │
  ├── 1D Causal Convolution (on q, k, v only, not z/ba)
  │   conv1d: depthwise Conv1d(kernel_size=4, groups=conv_dim)
  │   conv_dim = key_dim×2 + value_dim = 8192
  │   Applied with SiLU activation
  │   Gives each token local context before the recurrence
  │
  ├── Delta Rule Parameters
  │   beta = sigmoid(b)        ← selective gating
  │   g = -exp(A_log) * softplus(a + dt_bias)  ← discretization
  │   A_log: learned per-value-head (init uniform(0,16))
  │   dt_bias: learned per-value-head (init = 1)
  │   mamba_ssm_dtype: float32 (numerical stability)
  │
  ├── Gated Delta Rule Recurrence
  │   Two implementations:
  │   ├── Single-token decode: recurrent_gated_delta_rule
  │   │   State: S_t = (1 - β_t)⊙S_{t-1} + β_t⊙v_t⊙k_t^T
  │   │   Output: o_t = S_t @ q_t
  │   │
  │   └── Prefill/chunked: chunk_gated_delta_rule
  │       Chunk-wise parallel formulation
  │
  │   Both use QK L2-norm for numerical stability
  │   O(n) time, O(1) memory per layer state
  │
  ├── Gated Output (via z projection)
  │   output = RMSNorm(core_attn_out) * gate(z)
  │   z acts as a learned per-head output gate
  │
  ├── Output projection
  │   out_proj: Linear(4096 → 2048)
  │
  └── Residual: x = x + delta_output
```

### Why Gated DeltaNet?
Full softmax attention is O(n²) — prohibitive at 256K context. Gated DeltaNet uses a **linear recurrence** (O(n) time, O(1) memory per layer) for efficient local-to-medium-range context processing. The 10 full attention layers handle long-range global mixing.

This follows the **hybrid linear + attention** design pattern (pioneered by Jamba, used in Samba, Griffin, etc.): interleave efficient recurrent layers with sparse full-attention layers.

---

## 3C. MoE FFN Block (all 40 layers)

Every layer — regardless of attention type — uses the same MoE FFN structure:

```
Input: x.shape = [batch, seq_len, 2048]
  │
  ├── Router (learned gating)
  │   Linear(2048 → 256)                 num_experts=256
  │   Softmax over expert logits
  │   Keep top-8 experts (num_experts_per_tok=8)
  │   Normalize top-8 weights to sum to 1
  │
  ├── Shared Expert (always active, but gated)
  │   SwiGLU FFN: gate=Linear(2048→512), up=Linear(2048→512), down=Linear(512→2048)
  │   intermediate_size=512 (shared_expert_intermediate_size)
  │   shared_expert_gate: Linear(2048 → 1) → sigmoid
  │   shared_output = gate_value × shared_expert(x)
  │   (Learned scalar per token controlling expert contribution)
  │
  ├── Top-8 Selected Experts
  │   Each expert is a SwiGLU FFN with merged gate+up projection
  │   moe_intermediate_size=512
  │   gate_up_proj: [256 experts, 1024, 2048]  (gate + up merged)
  │   down_proj:    [256 experts, 2048, 512]
  │
  │   For each selected expert e:
  │     gate, up = chunk(x @ gate_up_proj[e], 2, dim=-1)
  │     hidden_e = SiLU(gate) * up
  │     output_e = hidden_e @ down_proj[e]
  │
  │   output = Σ( weight_e × output_e ) for e in top-8
  │
  ├── Combine: gated_shared_output + routed_expert_output
  │
  ├── Router aux loss (for load balancing)
  │   coefficient = 0.001
  │   Encourages uniform expert utilization across the batch
  │
  └── Residual: x = x + ff_output
```

### Key numbers
- **256 experts** → the model has 256 independent SwiGLU FFNs
- **8 activated per token** → only 8/256 = 3.125% of expert params are used per token
- **3.15M params per expert** → each expert is small (2048 → 512 → 2048 with 3 matrices)
- **Total expert params**: 256 × 3.15M = **806M** (0.8B)
- **Shared expert**: another 3.15M, always active regardless of routing

### Why 256 experts?
The original Qwen3 paper used 128 experts for 30B-A3B. Qwen3.6 doubles to 256. With only 8 active, this is extreme fine-grained expert segmentation (Dai et al., 2024). Each expert becomes more specialized, and the router has finer-grained control over which knowledge is activated.

### Why shared expert + routed experts?
The shared expert handles "common knowledge" that every token needs, while the routed experts handle specialized knowledge. This prevents the router from wasting capacity on common patterns. (Qwen3 paper explicitly removed shared experts; Qwen3.6 added them back.)

---

## 4. Output Head

```
Final hidden → RMSNorm → Linear(2048 → 248320) → logits
```

- **Untied embeddings** (`tie_word_embeddings: false`)
  - The embedding layer and LM head are separate parameter matrices
  - LM head has 248320 × 2048 ≈ **509M params** (same as embedding)
  - Untied = more parameters but more expressive capacity

---

## 5. MTP Head (Multi-Token Prediction)

```
MTP layer at the output:
  Predicts token_{t+1} and token_{t+2} in the same forward pass
  ├── 1 MTP hidden layer (mtp_num_hidden_layers=1)
  ├── No dedicated embeddings (mtp_use_dedicated_embeddings=false)
  └── Enables speculative decoding without a separate draft model
```

Instead of just predicting the next token, the model can output **two tokens** per forward pass. The second token prediction is used as draft tokens for speculative decoding (connecting to Ep05). This is the "eagle" / "self-speculation" pattern — the model speculates about itself.

---

## 6. Vision Encoder

```
Image / Video → Qwen3.6-35B-A3B is multimodal:
  │
  ├── ViT Encoder (27 layers)
  │   depth=27, hidden_size=1152, num_heads=16
  │   patch_size=16 (image patches)
  │   temporal_patch_size=2 (video frame merging)
  │   activation: gelu_pytorch_tanh
  │   intermediate_size=4304
  │   num_position_embeddings=2304 (supports high-res)
  │
  ├── Projection: 1152 → 2048 (out_hidden_size)
  │   Projects vision tokens into text embedding space
  │
  └── Special tokens: vision_start=248053, vision_end=248054
      image_token=248056, video_token=248057
```

The encoded visual tokens are interleaved with text tokens at the input, then processed by the full 40-layer hybrid transformer. mRoPE handles the 3D positional encoding for spatial (x, y) and temporal (frame) dimensions.

---

## 7. Key Differences from Qwen3 Paper

The paper describes Qwen3-30B-A3B (the predecessor). Qwen3.6-35B-A3B has these architectural changes:

| Aspect | Qwen3-30B-A3B (paper) | Qwen3.6-35B-A3B (actual) |
|--------|----------------------|--------------------------|
| **Total params** | 30B | ~35B |
| **Layers** | 48 | 40 |
| **Q heads / KV heads** | 32 / 4 | 16 / 2 |
| **Head dim** | 128 (typical) | **256** (double) |
| **Total experts** | 128 | **256** |
| **Shared expert** | ❌ "excludes shared experts" | ✅ Yes, added back |
| **Attention type** | Full attention only (48 layers) | **Hybrid**: 30 linear + 10 full |
| **Context length** | 128K | **262K** |
| **RoPE** | Standard RoPE | **mRoPE** (3D, multi-res) |
| **Partial RoPE** | Not mentioned | 25% |
| **MTP** | ❌ | ✅ 1 layer |
| **Vision** | ❌ Text-only | ✅ ViT encoder |
| **Attention gate** | ❌ | ✅ `attn_output_gate` |
| **Vocab size** | 151,669 | **248,320** |

---

## 8. Forward Pass Summary (Text-Only)

```
Step 1: Tokenize
  "What is attention?" → [271, 65825, 374, 14751, 30]
  (5 tokens)

Step 2: Embed
  lookup → [5, 2048] float tensor

Step 3: Apply mRoPE
  Add 3D position encoding to Q/K (25% of head dim)

Step 4: Process through 40 hybrid layers
  Layer 0  (linear attn): Gated DeltaNet + MoE + residual
  Layer 1  (linear attn): Gated DeltaNet + MoE + residual
  Layer 2  (linear attn): Gated DeltaNet + MoE + residual
  Layer 3  (full attn):   QKV + GQA + MoE + residual
  Layer 4  (linear attn): Gated DeltaNet + MoE + residual
  ... pattern repeats ...
  Layer 39 (full attn):   QKV + GQA + MoE + residual

Step 5: Final RMSNorm

Step 6: LM Head
  hidden → logits [5, 248320]

Step 7: Sample next token
  argmax or top-p sampling

Step 8 (optional): MTP Head
  Predict token_{t+2} from same hidden state
```

---

## 9. What We Already Know (Connecting to Ep07-08)

| Concept | Covered in | Connects to |
|---------|-----------|-------------|
| BPE Tokenization | Ep07 | Vocab of 248,320 tokens |
| Embedding lookup | Ep07 | Token → 2048-dim vector |
| RoPE | Ep08 | mRoPE is the 3D extension |
| Position encoding | Ep08 | 25% partial RoPE (design choice) |

The next layer to peel back: **how the hybrid attention works** (full attention vs Gated DeltaNet), **how the MoE router assigns experts**, and **what each expert learns**.
