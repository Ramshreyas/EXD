# Ep10 Shownotes — mRoPE: The Geometry of Position

> Recording guide. Open `mrope_exploration.ipynb` in VS Code. Walk through the notebook,
> explaining each visualization as it appears. This is a "watch the math happen" episode.

---

## Recording Flow

### 0. Setup (Cell 1–3)
> "We're not loading the full 35B model. Just the tokenizer and the embedding matrix —
> extracted directly from the safetensors shard. 509 million parameters, ~1GB."

- Show vocab size, embedding shape
- Explain the 8 pseudo-heads arrangement (2048 ÷ 256 = 8)

### Section 1 — Why Rotate? (Cells 4–5)
> "Before the math, the intuition. Take a single 2D vector. Apply a rotation where the
> angle is position × frequency. At position 0: unrotated. At position 50: almost a
> full circle."

- **Cell 4**: The 5-panel rotation visualization
  - Point out: same vector, different positions → different angles
  - The arc shows how much it's spun
- **Cell 5**: Dot product test
  - "p=0,q=5 gives the same dot product as p=100,q=105"
  - This is the key property: relative distance, not absolute position

### Section 2 — Frequency Ladder (Cells 6–7)
> "Real embeddings have 256 dims per head, 64 get rotated = 32 pairs. Each pair
> spins at a different frequency."

- **Cell 6**: Frequency construction. Show theta=10M, highest/lowest frequencies
- **Cell 7**: The two-panel frequency visualization
  - Left: full spectrum. Point out the log scale — huge range.
  - Right: cos(angle) vs position for 3 pairs
    - Fast pair (red): completes a cycle every ~3 positions
    - Slow pair (green): barely moves in 200 positions
  - "This is the frequency ladder. Fast = local grammar. Slow = paragraph-level structure."

### Section 3 — Real Embeddings, Rotated (Cells 8–11)
> "Now the real thing. We take the actual Qwen embeddings, reshape into pseudo-heads,
> and apply RoPE."

- **Cell 8**: The `apply_rope_1d` function — walk through it briefly
  - Outer product for angles, cos/sin, split rotary/pass-through, rotate pairs, recombine
- **Cell 9**: Show numbers changing
  - dims 0–5 change with position
  - dims 64+ NEVER change (that's the 75% position-free zone)
- **Cell 10–11**: **THE BIG VISUALIZATION**
  - PCA of 5 words at 100 positions each
  - Star = position 0, lines trace orbits
  - "Each word has its own neighborhood. Position pushes it around within that neighborhood."
  - "These aren't perfect circles because it's a 2D projection of 32 simultaneous rotations."

### Section 4 — Multi-Resolution (Cells 12)
> "mRoPE divides the 32 pairs into 3 sections: [11 T, 11 H, 10 W]"

- The bar chart showing frequency per section
- The interleave labels: THWTHWTHW on the top axis
- "For text, T=H=W, so it's just standard RoPE. For images, each coordinate is independent."

### Section 5 — 3D Trajectory (Cells 13–14)
> "Let's watch one word's rotary subspace move through 3D PCA space."

- **Cell 13**: Build the trajectory
- **Cell 14**: The 3D plot
  - Color gradient = position
  - "It's a complex quasi-periodic path — 32 incommensurate frequencies mean it never exactly repeats."
  - "But it stays bounded — rotation preserves vector norm."

### Section 6 — The Property Demonstrated (Cell 15)
> "The most important property: dot products depend only on relative distance."

- Show the table: all rows with same |p-q| have identical dots
- "p=0→5, p=100→105, p=1000→1005 — all give the same number"
- **Translation invariance for attention scores**

### Section 7 — The 75/25 Split (Cells 16–17)
> "Three lines on one plot: rotary dims, position-free dims, combined."

- Blue line (position-free): flat at 1.0 — "the model always knows it's 'cat'"
- Red line (rotary): oscillates wildly — "encodes how far from reference"
- Green line (combined): high baseline + position wiggle
- "This separation is brilliant engineering. Semantics in one subspace, position in another."

### Section 8 — Theta Comparison (Cell 18)
> "θ=10M for Qwen vs 500K for Llama 3 vs 10K for the original paper."

- Wavelength comparison: Qwen's slowest pair wraps at ~10M positions
- Context window lines: 262K for Qwen vs 128K for Llama
- "Qwen's θ is 20× higher than Llama's. That's how you go from 128K to 262K context."

## Recording Notes

1. **Pacing**: This is dense. Pause after each visualization. Let it land.
2. **Hand gestures**: The rotation metaphor benefits from physical "twisting" motions on camera.
3. **Screen share**: Full-screen the notebook. Scroll slowly through cells.
4. **Code skip**: Don't read every line of `apply_rope_1d`. Explain the idea, let them read the code.
5. **Big moment**: The PCA orbit plot (Cell 11) is the payoff. Dwell on it.

## Connections

| Ep | Connection |
|----|-----------|
| Ep07 | Tokenization & embeddings — these are the vectors we're rotating |
| Ep08 | Positional embeddings intro — this IS Qwen's actual position encoding |
| Ep09 | Architecture guided tour — mRoPE is the "25% partial RoPE" in the diagram |
| Ep05 | Speculative decoding — MTP head discussed, next token prediction |
| Ep11 (planned) | Attention — where the relative distance property actually gets used |

## Key Numbers

| Parameter | Value |
|-----------|-------|
| Embedding dims | 2048 |
| Pseudo-heads | 8 (2048 ÷ 256) |
| Head dim | 256 |
| Rotary dims | 64 (25% of 256) |
| Frequency pairs | 32 |
| RoPE theta | 10,000,000 |
| mRoPE sections | [11, 11, 10] |
| Position-free dims | 192 per head (75%) |
| Embedding params | 509M |
