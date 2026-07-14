# MRoPE

**Multi-Resolution Rotary Position Embedding in Qwen3.6-35B-A3B**

> 📓 **Companion notebook:** [`EXDai/mrope-exploration`](https://huggingface.co/datasets/EXDai/mrope-exploration) —
> download and run every code snippet yourself with real Qwen3.6 embeddings.

---

## 1. The Problem: Position Must Exist

A token embedding is a vector — a point in $R^{2048}$. The embedding for `␣cat`
is the same vector regardless of where it appears in a sentence. This is correct:
"cat" means cat. But it creates a problem.

Consider:

```
the dog bit the man
the man bit the dog
```

Identical tokens, opposite meanings. Without position information, the model sees the
same multiset of vectors in both sentences and cannot distinguish them.

Transformers solve this by injecting **position information** into the token
representations. The question is: *what form should that information take?*

---

### 1.1 Absolute vs. Relative Position

The naive approach: attach each token to its absolute index i.

$$x'_i = x_i + p_i \quad\text{where}\quad p_i = f(i)$$

If the model sees "the dog" at positions $(0, 1)$ during training, it learns that the
pattern lives at absolute coordinates $(0, 1)$. At inference, what happens if the same
pattern appears at $(100, 101)$? The model has never seen those coordinates — it must
generalize from a training distribution.

Now consider what happens when we shift text by inserting whitespace:

```
        the dog barked       ← positions (8, 9, 10)
the dog barked               ← positions (0, 1, 2)
```

The absolute positions are completely different: $(8,9,10)$ vs $(0,1,2)$. But the
**meaning** of the sequence — the relationship between tokens — is identical. An
absolute position encoding forces the model to learn "$\texttt{the}$ at position 8
followed by $\texttt{dog}$ at position 9" as completely separate from
"$\texttt{the}$ at position 0 followed by $\texttt{dog}$ at position 1." This is
wasteful: the model re-learns the same linguistic pattern at every possible
coordinate.

What matters for meaning is **relative position** — the offset between tokens — not
where the sequence happens to sit in the document.

---

### 1.2 What We Want

Return to the whitespace example:

```
        the dog barked       ← positions (8, 9, 10)
the dog barked               ← positions (0, 1, 2)
```

In both versions, `the` and `dog` are adjacent —
$q-p = 1$ in both cases. 

Put mathematically, if position encoding is doing its job, the dot product
between these two tokens (which drives attention, as we'll see later) should be the same regardless of
whether the pair sits at $(0,1)$ or $(8,9)$:

$$\langle\,\texttt{the}_R,\; \texttt{dog}_R\,\rangle_{(0,1)} \;=\; \langle\,\texttt{the}_R,\; \texttt{dog}_R\,\rangle_{(8,9)}$$

More generally, a good position encoding should satisfy:

$$\langle\,R_p(a),\; R_q(b)\,\rangle = f(a, b,\; q-p)$$

The dot product between two rotated tokens should depend on:

1. **Who the tokens are** — $a$ and $b$, their semantic embeddings
2. **How far apart they are** — the signed relative offset $q-p$
3. **Nothing else** — not on $p$ or $q$ individually

This is **translation invariance** for relative position. Whether a pair of tokens
appears at $(0, 1)$ or $(1000, 1001)$, the attention score between them is the same.

Among the family of transformations that satisfy this property, **rotation** is the
natural choice. It preserves vector norms (semantic information isn't distorted) and
has the group property $R_p^T R_q = R_{q-p}$, which is exactly what delivers
translation invariance.

---

### 1.3 Rotation by Example

Let's make this concrete. Start with a simple vector in 2D:

$$a = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$

A 2D rotation matrix rotates a vector counterclockwise by angle $\theta$:

$$R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

Multiply them:

$$R_\theta \, a = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} \cos\theta \\ \sin\theta \end{bmatrix}$$

Let's evaluate at specific angles:

$$\theta = 0: \quad \cos(0) = 1,\; \sin(0) = 0 \quad\Rightarrow\quad R_0 a = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$

$$\theta = \tfrac{\pi}{2}: \quad \cos(\tfrac{\pi}{2}) = 0,\; \sin(\tfrac{\pi}{2}) = 1 \quad\Rightarrow\quad R_{\pi/2} a = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

$$\theta = \pi: \quad \cos(\pi) = -1,\; \sin(\pi) = 0 \quad\Rightarrow\quad R_\pi a = \begin{bmatrix} -1 \\ 0 \end{bmatrix}$$

$$\theta = \tfrac{3\pi}{2}: \quad \cos(\tfrac{3\pi}{2}) = 0,\; \sin(\tfrac{3\pi}{2}) = -1 \quad\Rightarrow\quad R_{3\pi/2} a = \begin{bmatrix} 0 \\ -1 \end{bmatrix}$$

$$\theta = 2\pi: \quad \cos(2\pi) = 1,\; \sin(2\pi) = 0 \quad\Rightarrow\quad R_{2\pi} a = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$

The vector traces the unit circle: right → up → left → down → back to right.
At every step, $\|R_\theta a\| = \sqrt{\cos^2\theta + \sin^2\theta} = 1$ —
**rotation preserves length**, so values stay bounded no matter how many times
the vector is rotated.

In the plane:

```
                  ↑ y
                  │
                  ● [0,1]  θ=π/2
                  │
                  │
  ───[-1,0] ●─────┼─────● [1,0]── → x
       θ=π        │      θ=0
                  │
                  ● [0,-1]  θ=3π/2
                  │
```

The single vector $a = [1,0]$ spins through these four positions as $\theta$
increases. All four have the same length (1) — rotation is an **isometry**:
it moves the vector without stretching or shrinking. This matters in practice
because values stay bounded (no vanishing or exploding), even after being
rotated repeatedly across dozens of transformer layers.

---

### 1.4 Translation Invariance, Demonstrated

Now set the rotation angle proportional to position: $\theta_p = p \cdot \omega$.
Choose $\omega = \pi/2$ (90° per position step) for clean arithmetic.

Consider the sentence:

```
and the cat and the dog and the cat
      ^^^                     ^^^
    (the,cat)              (the,cat)
  positions (1,2)       positions (7,8)
```

For demonstration, assign simple 2D vectors to these tokens:

$$a = \texttt{the} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \qquad
  b = \texttt{cat} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

To get the rotated vectors for the first occurrence — `the` at position 1 and
`cat` at position 2 — we rotate each by its position angle:

$$\theta_1 = 1 \cdot \tfrac{\pi}{2} = 90°,\qquad
  \theta_2 = 2 \cdot \tfrac{\pi}{2} = 180°$$

$$R_1 a = R_{90°}\,a = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

$$R_2 b = R_{180°}\,b = \begin{bmatrix} -1 & 0 \\ 0 & -1 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ -1 \end{bmatrix}$$

Their dot product: $(0)(0) + (1)(-1) = -1$.

More generally, for any two vectors $a$ and $b$, the dot product after rotation depends
only on the relative offset $q-p$:

$$(R_p a) \cdot (R_q b) = \sin\!\big((q-p) \cdot \tfrac{\pi}{2}\big)$$

Now let's compute the second occurrence — `the` at position 7 and `cat` at
position 8 — same offset $q-p = 1$, different positions:

$$\theta_7 = 7 \cdot \tfrac{\pi}{2} = \tfrac{7\pi}{2} \equiv 270°,\qquad
  \theta_8 = 8 \cdot \tfrac{\pi}{2} = 4\pi \equiv 0°$$

For 270°: $\cos(270°) = 0$, $\sin(270°) = -1$, so the rotation matrix is
$R_{270°} = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}$.
For 0°: $\cos(0°) = 1$, $\sin(0°) = 0$, giving $R_{0°} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$.

Now apply them:

$$R_7 a = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ -1 \end{bmatrix}$$

$$R_8 b = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

$$(R_7 a) \cdot (R_8 b) = (0)(0) + (-1)(1) = -1$$

Same dot product: $-1$. Despite being 6 positions to the right, the attention
score is identical.

For comparison, a different offset. For $q-p = 2$ (180° apart):

$$(R_0 a) \cdot (R_2 b) = \sin(\pi) = 0$$

**Summary:** the dot product depends on $q-p$, not on $(p,q)$. Shift both tokens
by any amount — as long as the offset stays the same, the attention score stays
the same. This is what we set out to achieve in §1.2.

---

### 1.5 From Toy to Real: What Changes in Actual RoPE

The toy example uses a **single frequency** $\omega = \pi/2$ on a **single 2D vector**.
Now let's build up to what Qwen3.6 actually does, step by step.

**Step 1 — 2048 dimensions.** A token embedding is a vector in $R^{2048}$,
not $R^2$. But we don't rotate the whole thing as one block. Instead, we 
split it into smaller subspaces, and only rotate them (actually, part of them).

**Step 2 — Split into 256-dimensional subspaces.** $2048 \div 256 = 8$. Each 256-D
chunk is called a *pseudo-head* (the actual model has 16 real attention heads, but
the embedding splits cleanly into 8 groups of 256). We'll apply the rotation
independently within each pseudo-head.

**Step 3 — Within each head, only 64 of the 256 dims rotate.** The remaining 192
are *position-free* — they stay exactly as they are, carrying pure semantic
information. The 64 rotary dimensions pair up to form **32 frequency pairs**:
$(d_0, d_1), (d_2, d_3), \ldots, (d_{62}, d_{63})$. Each pair rotates like our toy
2D vector, but at its own frequency.

**Step 4 — A different frequency for each pair.** Instead of one $\omega$, we
need 32 of them:

$$\omega_i = \frac{1}{10\,000\,000\,^{\,2i / 64}}, \qquad i = 0, 1, \ldots, 31$$

Pair 0 (fast): $\omega_0 = 1$ — completes a full rotation every $2\pi$ positions.
Pair 31 (slow): $\omega_{31} \approx 10^{-7}$ — barely moves over thousands of
positions. Together they form a **frequency ladder** that captures positional
relationships at every scale, from adjacent words to paragraph-spanning patterns.

**Step 5 — 3D multi-resolution (the "m" in mRoPE).** Qwen3.6 is multimodal (text
+ images + video). So those 32 frequency pairs are split across three spatial
dimensions: 11 for temporal (T), 11 for height (H), 10 for width (W), interleaved
as THWTHWTHW... For text-only input, T = H = W = position, so mRoPE reduces to
standard RoPE. For images, each visual token has distinct $(t, h, w)$ coordinates,
giving the model native 3D spatial awareness.

We'll build each of these steps in detail in the sections ahead.

---

## 2. Step 1 & 2 — Loading Real Embeddings

Let's load the actual embedding matrix from Qwen3.6-35B-A3B and pull out the
vector for the token `␣dog` (token ID 5388).

```python
from transformers import AutoTokenizer
from safetensors import safe_open

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-35B-A3B", trust_remote_code=True)

# Load embedding weights directly from the safetensors shard (~1GB)
with safe_open("model-00001-of-00026.safetensors", framework="pt") as f:
    embed_weight = f.get_tensor("model.language_model.embed_tokens.weight")

dog_id = tok.encode(" dog")[0]   # 5388 — mid-sentence token
dog_emb = embed_weight[dog_id]    # shape: [2048]
```

```
Embedding shape: [2048]
First 6 values: [0.0192, -0.0052, -0.0234, -0.0090, -0.0107, -0.0101]
```

Now split into 8 subspaces of 256 dimensions each:

```python
HEAD_DIM = 256
NUM_HEADS = 2048 // 256  # = 8

dog_heads = dog_emb.reshape(NUM_HEADS, HEAD_DIM)  # [8, 256]
```

```
Shape after reshape: [8, 256]
Head 0, first 6 dims: [ 0.0192, -0.0052, -0.0234, -0.0090, -0.0107, -0.0101]
Head 1, first 6 dims: [ 0.0105, -0.0009, -0.0081, -0.0000,  0.0075, -0.0045]
```

Each of the 8 pseudo-heads is an independent 256-dimensional vector. The rotation
will be applied identically within each head, but the heads themselves are separate
subspaces — head 0 and head 1 contain different features of "dog."

---

## 3. Step 3 — Rotary vs. Position-Free Dimensions

Within each 256-D head, only the first 64 dimensions participate in the rotation.
The remaining 192 stay exactly as they are, regardless of position.

```python
ROTARY_DIM = int(HEAD_DIM * 0.25)  # = 64

rotary_part = dog_heads[:, :ROTARY_DIM]          # [8, 64]  — these will rotate
free_part   = dog_heads[:, ROTARY_DIM:]           # [8, 192] — these stay frozen
```

```
Rotary part shape:      [8, 64]   (8 heads × 64 rotary dims)
Position-free part:     [8, 192]  (8 heads × 192 frozen dims)

Rotary, head 0, dims 0–3:  [ 0.0192, -0.0052, -0.0234, -0.0090]
Free,   head 0, dims 0–3:  [-0.0045, -0.0317,  0.0139, -0.0030]
```

The 64 rotary dimensions pair up to form **32 frequency pairs**:

$$\text{pair } 0: (d_0, d_1),\; \text{pair } 1: (d_2, d_3),\; \ldots,\; \text{pair } 31: (d_{62}, d_{63})$$

Each pair behaves like our toy 2D vector from §1.4 — it gets rotated by an angle
$p \cdot \omega_i$ that depends on position. The 192 frozen dimensions carry
"what the token IS" — semantics that don't change with position.

---

## 4. Step 4 — The Frequency Ladder

Each of the 32 pairs gets its own frequency $\omega_i$. The formula:

$$\omega_i = \frac{1}{\theta^{\,2i / d}}, \qquad \theta = 10\,000\,000,\; d = 64$$

```python
ROPE_THETA = 10_000_000
i = torch.arange(0, 32, dtype=torch.float32)
freqs = 1.0 / (ROPE_THETA ** (2 * i / 64))

# Wavelength = how many positions for a full 360° rotation
wavelengths = 2 * torch.pi / freqs
```

```
Pair  0: ω = 1.000000,        λ = 6.3 positions        (fast — local grammar)
Pair 10: ω = 0.006494,        λ = 967.6 positions      (medium — sentence structure)
Pair 20: ω = 0.00004217,      λ = 148,998 positions    (slow — paragraph context)
Pair 31: ω = 0.000000165482,  λ = 38,000,000 positions (glacial — document scale)
```

The fastest pair completes a full rotation every ~6 tokens — perfect for detecting
adjacent-word relationships. The slowest pair has barely moved after 262,144 tokens
(Qwen3.6's full context window). Together, the 32 frequencies form a **frequency
ladder** that captures positional structure at every scale.

---

## 5. Putting It Together — Rotating a Real Embedding

We now have all the pieces. Let's rotate the `␣dog` embedding at position 0
(no rotation) and position 5:

```python
def apply_rope(x_heads, positions, freqs):
    """x_heads: [batch, seq_len, num_heads, head_dim]"""
    angles = torch.outer(positions.float(), freqs)
    cos, sin = torch.cos(angles), torch.sin(angles)

    # Split rotary and position-free parts
    rotary_dim = len(freqs) * 2
    x_rot  = x_heads[..., :rotary_dim]
    x_pass = x_heads[..., rotary_dim:]

    # Reshape into pairs: [..., seq, heads, pairs, 2]
    x_rot_pairs = x_rot.reshape(*x_rot.shape[:-1], -1, 2)
    cos = cos[None, :, None, :]  # broadcast over batch and heads
    sin = sin[None, :, None, :]

    # Apply rotation to each pair: (x1, x2) → (x1·cos - x2·sin, x1·sin + x2·cos)
    out = torch.zeros_like(x_rot_pairs)
    out[..., 0] = x_rot_pairs[..., 0] * cos - x_rot_pairs[..., 1] * sin
    out[..., 1] = x_rot_pairs[..., 0] * sin + x_rot_pairs[..., 1] * cos

    return torch.cat([out.reshape(*x_rot.shape), x_pass], dim=-1)

# Apply rotation
dog_h = dog_heads.unsqueeze(0).unsqueeze(0)  # [1, 1, 8, 256]
rot0 = apply_rope(dog_h, torch.tensor([0]), freqs)  # position 0
rot5 = apply_rope(dog_h, torch.tensor([5]), freqs)  # position 5
```

```
Head 0, rotary dims 0–3 at pos 0:  [ 0.0192, -0.0052, -0.0234, -0.0090]
Head 0, rotary dims 0–3 at pos 5:  [ 0.0004, -0.0199,  0.0243,  0.0061]  ← rotated!

Head 0, free dims 64–67 at pos 0:  [-0.0045, -0.0317,  0.0139, -0.0030]
Head 0, free dims 64–67 at pos 5:  [-0.0045, -0.0317,  0.0139, -0.0030]  ← unchanged!
```

The rotary dimensions shift with position. The position-free dimensions stay frozen.
This is the 25/75 split in action.

---

## 6. Translation Invariance — The Payoff

Finally, let's verify that the same token pair at the same relative offset produces
the same dot product, regardless of where it sits in the sentence. We use the
sentence from §1.2:

```
and the cat and the dog and the cat
     ^^^                      ^^^
  (the,cat)                (the,cat)
  positions (1,2)        positions (7,8)
```

```python
sentence = "and the cat and the dog and the cat"
token_ids = tok.encode(sentence)

def pair_dot(p, q):
    # Apply RoPE to tokens at positions p and q, return their dot product
    positions = torch.tensor([p, q])
    emb = embed_weight[token_ids[[p, q]]]  # [2, 2048]
    emb_h = emb.reshape(2, NUM_HEADS, HEAD_DIM).unsqueeze(0)
    rotated = apply_rope(emb_h, positions, freqs).reshape(2, -1)
    return torch.dot(rotated[0], rotated[1]).item()
```

```
Pair             positions    q-p      dot product
--------------------------------------------------
(the,cat) early   (1,2)       1        0.015581
(the,cat) late    (7,8)       1        0.015581     ← identical!
(the,dog)         (4,5)       1        0.015196     ← different tokens

(the,cat) early vs late diff: 9.31e-10  ≈ zero
(the,cat) vs (the,dog) diff:  0.000384  ← different words
```

Same token pair, same $q-p$, different absolute positions → identical dot product
to 10 decimal places. Different token pair at the same offset → different dot
product. This is exactly the property we asked for in §1.2: the attention score
between two tokens depends on **who they are** and **how far apart they are** —
not on where they sit in the sequence.

---

**Next Episode: Building the full mRoPE implementation — the 3D sections, interleaving,
and what changes for multimodal input.**
