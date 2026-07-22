# Ep12 Shownotes — Attention Heads: Inside the Qwen3.6 Transformer

> Recording guide. Open `attention_heads.ipynb` in VS Code (connected to atom's
> jupyterlab kernel). Walk through the notebook live, feeding real text through
> the actual Qwen3.6-35B-A3B model.
>
> **Prerequisite:** Ep11 covers the linear algebra toolkit (dot products, matrix×vector,
> projection up/down, rotation). This episode assumes viewers know what a projection is
> and why Q is bigger than K.

---

## Recording Flow

### 0. Setup (Cells "Setup" through "Identify Layers")
> "We're loading the full 35B model. 130GB unified memory on the GB10 — it fits
> comfortably. We use eager attention so we can extract the real weight matrices."

- Show model config: 2048 hidden, 40 layers, 16 Q heads, 2 KV heads (GQA 8:1)
- Reveal the layer pattern: only 4 full-attention layers out of 40
- "10% of layers do classic attention. The other 90% use a fast linear approximation."

### Section 1 — Input & Forward Pass (Cells "Pick an Input Text" through "Forward Pass")
> "A 13-token sentence: 'The cat sat on the mat because it was tired.' Watch as the
> model processes it through all 40 layers."

- Show tokenization: how the sentence breaks into subword tokens
- Run the forward pass with `output_attentions=True`
- Point out: we get attention weights from ALL layers, including linear attention ones

### Section 2 — Attention Heatmaps (Cell "plot_heatmaps")
> "Each row is one full-attention layer. Each cell (i,j) is how much token j
> influences token i."

- Walk through the 4-layer heatmap
- Point out: causal mask (upper triangle is zero — future tokens are invisible)
- Show how patterns change across layers
- "Early layers often show strong diagonal attention. Deeper layers spread out."

### Section 3 — Head-by-Head (Cell "head_by_head")
> "16 heads, each looking at different things."

- 4×4 grid — scan across and call out interesting patterns
- Some heads focus on the previous token (sub-diagonal), others look at sentence-start
- One head might track the period, another "the", another the clause boundary
- "They're specialists. Each head learns a different attention pattern."

### Section 4 — Manual Step-Through Setup (Cells "Stepping Through" through "grab_weights")
> "Heatmaps show the result. Now we open the machine."

- Show the Q/K/V/O projection dimensions:
  - Q: 2048 → 8192 (16 heads × 256 dim × 2 for gate!)
  - K: 2048 → 512 (2 KV heads × 256 dim)
  - V: 2048 → 512
  - O: 4096 → 2048
- "Q is 16× bigger than K or V. That's GQA — Grouped Query Attention."

### Section 5 — Run Up to Target Layer (Cell "run_to_target")
> "We can't just look at raw embeddings. We need to run through all preceding
> layers because each one transforms the hidden states."

- Run the loop: embed → layer 0 → layer 1 → ... → layer N-1
- "Now we're at the input to the first full-attention layer. These hidden states
> carry all the context from the preceding linear-attention layers."

### Section 6 — Q, K, V Computation (Cell "compute_qkv")
> "Here's where it happens. The hidden states hit four weight matrices simultaneously."

- Show the Q projection with the *2 factor
- Split into query and gate — "the gate decides how much to trust this head"
- QK Norm: "RMSNorm on Q and K, applied per-head. This stabilizes training."
- V projection: no norm, no frills

### Section 7 — RoPE Application (Cell "apply_rope")
> "Before scores, we rotate. This is the mRoPE from Ep10, happening live."

- Show Q and K being rotated
- Verify: first 64 dims change, last 192 stay identical
- "Position information enters here, through rotation. Without RoPE, attention
> wouldn't know word order."

### Section 8 — THE BIG MOMENT: Attention Scores (Cells "attention_scores" through "weighted_sum")
> "Q × K^T divided by sqrt(d_k). This is the heart of the transformer."

- Walk through each step:
  1. Repeat KV heads 8× (GQA expansion)
  2. QK^T / √256 = matmul + scaling
  3. Causal mask: future tokens → -inf
  4. Softmax: -inf → 0, rest → probabilities
  5. Weighted sum with V
  6. Gate: multiply by sigmoid(gate)
  7. Output projection: back to 2048
- "Each of these 16 heads just voted on what information to pass forward."

### Section 9 — Verification (Cell "compare_manual")
> "Our manual computation should match the model's output bit-for-bit."

- Side-by-side: model vs manual attention weights
- Difference heatmap: should be all zeros
- "This IS what the model does. Nothing hidden, no magic."

### Section 10 — GQA Deep Dive (Cell "gqa_viz")
> "8 Q heads share KV head 0. The other 8 share KV head 1."

- 2×8 grid: Q heads grouped by KV head
- "Same K and V, different Q — yet each head learns different patterns."
- "GQA saves 8× memory on K/V cache while keeping most of multi-head's expressivity."

### Section 11 — Layer Evolution (Cell "layer_evolution")
> "How does attention change from layer 3 to layer 39?"

- Bar charts: attention distribution shift across layers
- "Early layers: focus on nearby tokens. Deep layers: focus on semantic connections."
- Point out: "it" attending to "cat" in deeper layers (pronoun resolution!)

### Section 12 — The Gate (Cell "gate_analysis")
> "The sigmoid gate — Qwen3.5's unique contribution."

- Mean gate per head: which heads does the model trust more?
- Token × head gate heatmap: some tokens are gated harder
- "This is learned. The model decides per-head, per-token how much attention matters."

### Section 13 — Full vs Linear (Cell "full_vs_linear")
> "Why only 10% full attention?"

- Show the two module types side by side
- "Linear attention: O(n) instead of O(n²). No softmax, no explicit weight matrix."
- "The model uses full attention at regular intervals as 'synchronization points'"
- Pattern: linear, linear, linear, FULL, linear, linear, linear, FULL...

---

## Recording Notes

1. **Live coding feel**: Don't just execute — narrate each cell while it runs. The
   model loading takes time; use it to explain what we're about to do.
2. **Physical analogies**: Q/K/V = "what am I looking for" / "what do I contain" /
   "what information do I carry." This is the standard explanation but it actually
   works when you're watching it happen.
3. **Screen share**: Full-screen notebook. The heatmaps are the payoff.
4. **Pacing**: The manual step-through section (cells compute_qkv through
   weighted_sum) is dense. Pause between each step. These 5 cells are the core of
   the episode.
5. **Big reveal**: Cell "compare_manual" — when the difference is ~0, that's the
   moment. "We've reconstructed the exact attention computation from first principles."

---

## Connections

| Ep | Connection |
|----|-----------|
| Ep07 | Tokenization & embeddings — the vectors entering attention |
| Ep08 | Positional embeddings intro — why we need RoPE in attention |
| Ep09 | Architecture guided tour — the attention layer in the diagram |
| Ep10 | mRoPE — the exact rotation happening inside Qwen3.5Attention.forward |
| Ep11 | Linear algebra toolkit — dot products, projections, matrix×vector |
| Ep13 (planned) | Deep dive on one attention mechanism component |

---

## Key Numbers

| Parameter | Value |
|-----------|-------|
| Hidden size | 2048 |
| Num layers | 40 |
| Full-attention layers | 4 (every 4th) |
| Linear-attention layers | 36 |
| Q heads | 16 |
| KV heads | 2 |
| GQA ratio | 8:1 |
| Head dim | 256 |
| Q projection | 2048 → 8192 (heads × dim × 2 for gate) |
| K projection | 2048 → 512 |
| V projection | 2048 → 512 |
| O projection | 4096 → 2048 |
| RoPE rotary dims | 64 (25% of head_dim) |
| RoPE theta | 10,000,000 |
