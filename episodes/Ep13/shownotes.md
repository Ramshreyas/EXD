# Ep13 Shownotes — The Attention Mechanism: Inside the Score

> Recording guide. Open `attention_mechanism.ipynb` in VS Code (connected to atom's
> jupyterlab kernel). Walk through the notebook live, feeding real text through
> the actual Qwen3.6-35B-A3B model.
>
> **Prerequisite:** Ep12 covered Q/K/V/gate projections. This episode is what
> happens when Q and K meet: scores, mask, softmax, weighted sum, gate, output.

---

## Recording Flow

### 0. Setup (Cells "Setup" through "Find the First Full-Attention Layer")
> "Ep12 ended with Q, K, V, and a gate waiting at the door of an attention layer.
> Ep13 is what happens when they meet."

- Load the model with `attn_implementation="eager"` — "we need the real attention
  weights, so no flash attention."
- Show the layer discovery: full-attention layers are `[3, 7, 11, ..., 39]` — 10 of 40.
- "Ep12's article said 4 of 40. The config actually lists 10, every 4th layer.
  The notebook counts them — no hardcoding. We'll focus on full attention only."

### Section 1 — Run the Model Up to Layer 3 (Cells "Run the Model..." through "Recap")
> "Attention needs real hidden states, not raw embeddings. Layers 0-2 have already
> enriched the sentence before the first full-attention layer sees it."

- Forward pass with `output_hidden_states=True` and `output_attentions=True`
- Point out `hidden_states[3]` = the input to layer 3 (7 tokens × 2048 dims)
- Recap table: Q (16×256), gate (16×256), K (2×256), V (2×256)
- "Same three matrices as Ep12 — now measured on the real layer input, for all
  7 tokens at once."

### Section 2 — QK Norm (Cell "qk_norm")
> "The dot product measures direction AND volume. Loud heads would dominate."

- Show norms before vs after `q_norm`/`k_norm`: the spread collapses
- "V is deliberately not normalized — magnitude is the model's content knob."
- Tie back to Ep11's length-bias problem and Ep12's 'who's loud' plot

### Section 3 — RoPE (Cells "rope" through "rope_rotation")
> "Q and K are normalized but position-blind. Position enters HERE."

- Compute cos/sin the same way the model does (3 mRoPE axes, identical for text)
- Implement `rotate_half` + `apply_rope` in 10 lines — no magic
- The reveal: dims 0-63 change, dims 64-255 are EXACTLY zero change
- "This is Ep10's mRoPE, live, inside the layer. Rotation is how word order
  becomes visible to the score."

### Section 4 — The Score Matrix (Cell "score_matrix")
> "Q and K meet. This is the heart of the transformer."

- GQA expansion: repeat K/V 8× (heads 0-7 → KV head 0, heads 8-15 → KV head 1)
- `Q @ Kᵀ / √256` — walk through one cell: 'how much does token i want to look
  at token j?'
- Show 4 heads: different questions, different score landscapes
- "The /√256 scaling keeps softmax from saturating. 256-dim dot products have
  variance ~256."

### Section 5 — The Causal Mask (Cell "causal_mask")
> "Token i may only see tokens 0…i. The future is invisible."

- Build the −inf upper triangle, add to scores
- "This is why every attention heatmap has a black upper-right triangle."

### Section 6 — Softmax (Cell "softmax")
> "Each row becomes a budget."

- One row before/after: raw scores → probabilities summing to 1
- "−inf becomes exactly 0. The winner takes the largest share, not everything."
- Mention float32 for stability

### Section 7 — The Weighted Sum (Cell "weighted_sum")
> "Context is a blend, not a choice."

- `probs @ V` → 16 context vectors of 256 dims
- Show 'rug's recipe: a convex mix of all previous tokens' values
- "Attention never copies. It mixes. That's where multi-head expressivity
  comes from — 16 interpretations of the same sentence."

### Section 8 — The Gate (Cell "gate")
> "Qwen3.6's signature: a volume knob per head, per dimension."

- `× sigmoid(gate)` — gate → 0 silences, gate → 1 passes through
- Show the learned gate pattern: some heads trusted, others damped
- "The model decides per token, per head, in real time, how much to trust."

### Section 9 — Output Projection (Cell "o_proj")
> "4096 dims of opinion squeeze back into the 2048-dim residual stream."

- W_O: 2048 × 4096 — "the project-down from Ep11"
- "2048 in, 2048 out. The layer is done."

### Section 10 — Verification (Cell "verify")
> "Now the moment of truth: our manual math vs the model's own module."

- Run `attn(...)` on the same inputs, same mask, same cos/sin
- Difference heatmaps: weights and output — both ~0 (bf16 rounding)
- THE line: "We reconstructed the attention mechanism from first principles.
  Nothing hidden, no magic."

### Section 11 — What Attention Learned (Cells "attention_heads" through "pronoun")
> "The mechanism is generic. The patterns are learned."

- Second sentence with a pronoun: "The cat sat on the rug because it was tired."
- 4×4 grid of 16 heads at layer 3 — each head is a specialist
- The payoff: where 'it' looks across layers 3 → 39
- "Deep layers converge on 'cat'. The model learned pronoun resolution by gradient
  descent — we're watching it happen."
- Tease the omission: we left 'mat' out of the sentence on purpose. Ask the model
  what comes after 'The cat sat on the ___' — it says 'mat', and its attention
  lands on 'cat' and 'sat' (rhyme). That's the hook for generation.

### Section 12 — The Cost (Cell "cost")
> "Every token pairs with every previous token. That's O(n²)."

- FLOPs table for the 7-token case: QKᵀ + weighted sum dominate
- Log-log plot: full attention vs projections, 32 → 16K tokens
- "This is why only 10 of 40 layers do full attention. The other 30 use
  GatedDeltaNet — O(n). That's Ep14."

---

## Recording Notes

1. **The verification is the episode.** Cells score_matrix → verify are the core.
   Pause between each step. Build the anticipation — the diff heatmaps are the reveal.
2. **Procedural narration**: for every step say what it does, THEN why. "We divide
   by √256 — because un-scaled dot products saturate softmax."
3. **Physical analogies**: score = "how much does this token want to look at that
   one?"; softmax = "budget"; weighted sum = "mixing a smoothie, not picking a
   fruit"; gate = "volume knob".
4. **The pronoun section is the emotional payoff.** After all the math, watching
   "it" find "cat" in the deeper layers lands hard.
5. **Model loading takes ~4 min.** Use it to recap Ep12 and set up the pipeline
   story: "we're about to watch the score happen on real vectors."
6. **Screen share**: full-screen notebook. The 4×4 head grid and the pronoun
   chart are the visuals to linger on.

---

## Connections

| Ep | Connection |
|----|-----------|
| Ep07 | Tokenization & embeddings — the vectors entering the layer |
| Ep08 | Positional embeddings intro — why RoPE exists in the first place |
| Ep09 | Architecture guided tour — the attention layer in the diagram |
| Ep10 | mRoPE — the exact rotation applied in Step 2 of this episode |
| Ep11 | Linear algebra toolkit — dot products, scaling, project down |
| Ep12 | Q, K, V, gate — the inputs this episode's pipeline consumes |
| Ep14 (planned) | GatedDeltaNet — why 30 of 40 layers skip full attention |

---

## Key Numbers

| Parameter | Value |
|-----------|-------|
| Hidden size | 2048 |
| Num layers | 40 |
| Full-attention layers | 10 (every 4th: 3, 7, 11, …, 39) |
| Linear-attention layers | 30 |
| Q heads | 16 |
| KV heads | 2 |
| GQA ratio | 8:1 |
| Head dim | 256 |
| QK Norm | RMSNorm per head, eps 1e-6 |
| RoPE rotary dims | 64 of 256 (partial_rotary_factor 0.25) |
| RoPE theta | 10,000,000 |
| Scaling | 1/√256 = 1/16 |
| Mask | additive, −inf above diagonal |
| Softmax | float32 |
| Gate | sigmoid, applied after weighted sum, before W_O |
| W_O | 2048 × 4096 |
