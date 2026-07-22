# Ep11 Shownotes — Linear Algebra for Transformers

> Recording guide. Open `linear_algebra.ipynb` in VS Code (runs locally, no GPU needed —
> just numpy and matplotlib). Keep `vector_toolkit.html` open in a browser for interactive
> demos. End with `projections.html` as the bridge to Ep12.

---

## Recording Flow

### 0. Why This Episode Exists
> "Between rotation (Ep10) and attention heads (Ep12), there's a gap. We need a shared
> vocabulary for the math. This episode fills that gap."

- We aren't loading any models. No GPU needed. Pure linear algebra.
- "Every operation in a transformer is one of four things: dot product, matrix×vector,
> projection up, or projection down."

### Section 1 — Vectors Are Just Lists (Cells "Vectors Intro" + "Vectors Plot")
> "A token embedding is 2048 numbers. But let's start with 2 numbers we can draw."

- Show the 2D vectors: cat = [0.8, 0.3], dog = [0.6, 0.7]
- Plot them as arrows from origin
- "The math doesn't care about the dimension. What we do in 2D generalizes to 2048D."

### Section 2 — Rotation Recap (Cells "Rotation" + "All Angles")
> "You already know this from Ep10. Rotation is a matrix multiply that preserves length."

- Show the rotation matrix, apply to cat, verify norm unchanged
- 12-position rotation plot — the circle
- "This is what mRoPE does to Q and K. Only 64 of 256 dims per head."

### Section 3 — Dot Product (Cells "Dot Product" + "Heatmap")
> "The dot product is how transformers decide which tokens relate to which."

- Compute cat·dog, cat·cat, dog·dog by hand
- Notice: cat·cat = 0.73 but dog·dog = 0.85 — why?
- **THE HEATMAP**: "This grid IS the QK^T attention score matrix"
  - Point to cells: red = related, blue = unrelated
  - "But something's wrong here..."

### Section 4 — Normalization (Cells "Normalization" + "Norm Heatmap")
> "dog·dog > cat·cat — but they should both be 1.0 for self-similarity."

- Show that `a·a = |a|²` — the lengths differ
- Normalize to unit vectors: now all self-dots = 1.0
- "This is WHY Qwen3.5 uses QK Norm — RMSNorm on Q and K"
- Normalized heatmap: clean diagonal, direction-only comparisons
- "Without normalization, loud vectors dominate attention. With it, only direction matters."

### Section 5 — Matrix × Vector (Cells "Matrix×Vec" + "Shape Rule")
> "A matrix is a learned transformation. Multiply by a vector, get a new vector."

- Walk through W @ x step by step: each output element is a weighted sum
- Shape rule: (m×n) @ (n,) → (m,)
- Show the projection shape table — this is why W_Q is 8192×2048

### Section 6 — Projecting Up (Cells "Project Up" + "Analogy")
> "When the matrix has more rows than columns, you get MORE dimensions out."

- 2D cat → 5D via W_up
- "Same math as Q projection: 2048D → 8192D"

### Section 7 — Projecting Down (Cell "Project Down")
> "When the matrix has fewer rows than columns, you get FEWER dimensions out."

- 5D → 2D via W_down. Output ≠ original — compression loses information
- "Same as O projection: 4096D → 2048D. Model learns what to keep."

### Section 8 — Summary & Bridge (Cells "Summary" + "Bridge")
> "Every operation in the transformer is one of these."

- Read the summary table: rotation, dot product, normalize, project up, project down
- "Plus two nonlinearities: softmax and sigmoid. That's it."
- Open projections.html: same operations, real Qwen3.6 shapes
- Tease Ep12: "Next episode we watch this happen inside the 70GB model."

---

## Recording Notes

1. **This is a chalk-talk episode.** Less screen share, more explanation. The toolkit
   is for live demos, not slides.
2. **Use the toolkit extensively.** Each section has a corresponding tab — switch to it,
   drag things around, let the visual do the work.
3. **Bridge at the end is critical.** The last 2 minutes should connect everything to
   `projections.html` and tease Ep12. "Next episode: we watch this happen inside a 70GB model."
4. **No GPU, no model.** This episode runs entirely on Yoneda. The notebook uses numpy only.
   Makes it accessible to viewers who don't have a GB10.
5. **Pacing**: Sections 1-3 are the core. Section 4-5 are quick. Section 6 is a recap.
   Section 7-8 are the payoff.

---

## Connections

| Ep | Connection |
|----|-----------|
| Ep07 | Tokenization — the vectors we're manipulating |
| Ep08 | Positional embeddings — why we need rotation |
| Ep10 | mRoPE — rotation, now understood as matrix multiply |
| Ep12 (planned) | Attention heads — all four operations in action |
| Ep09 | Architecture tour — where each projection lives |

---

## Key Numbers

| Concept | Toy | Real (Qwen3.6) |
|---------|-----|-----------------|
| Hidden dim | 2D/3D | 2048D |
| Q projection | 3D → 5D | 2048D → 8192D |
| K projection | 3D → 2D | 2048D → 512D |
| V projection | 3D → 2D | 2048D → 512D |
| O projection | 5D → 3D | 4096D → 2048D |
| Rotation angle | 30° | position × frequency |
