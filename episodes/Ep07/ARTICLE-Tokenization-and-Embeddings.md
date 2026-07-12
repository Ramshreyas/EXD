# Tokenization & Embeddings from First Principles

How does a sentence become a list of vectors — and what does that space look like?

---

## The Full Pipeline

Before diving into mechanics, let's see the whole chain working end-to-end on live hardware.

### Text → Token IDs

Qwen's tokenizer uses BPE (Byte-Pair Encoding) with a vocabulary of 248,044 tokens:

```python
from transformers import AutoTokenizer
qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-35B-A3B")

prompt = "Gentrification is a complex and multifaceted process"
input_ids = qwen_tok.encode(prompt)
# → [271, 65825, 374, 264, 7846, 323, 10275, 1630, 3645]
```

These integers are what gets sent to the LLM. Inside the model, each integer becomes a 2048-dimensional vector via a lookup table.

### Tokens → LLM (Live Inference)

The token IDs are sent to the vLLM server on the GPU machine:

```python
API_BASE = "http://<hostname>.local:8000/v1"
MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"

def query_model(prompt, max_tokens=100):
    response = requests.post(
        f"{API_BASE}/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
    )
    return response.json()
```

The model processes input through 40 transformer layers of attention and feed-forward computation, then produces output token IDs one at a time.

---

## Section 1: From Text to Tokens

### The Goal of Tokenization — Vocabulary Compression

BPE starts with individual bytes and iteratively merges the most frequent pairs. Over ~100,000 merge steps, common subwords like "ing", "tion", "er" become single tokens.

```python
# Compare how different words split
words = ["transformer", "revolutionized", "NLP", "embeddings", "attention"]
for w in words:
    tokens = qwen_tok.tokenize(w)
    print(f"{w:20s} → {tokens}")
```

Output:
```
transformer           → ['transform', 'er']
revolutionized        → ['revolution', 'ized']
NLP                   → ['NLP']
embeddings            → ['embedding', 's']
attention             → ['attention']
```

### How BPE Actually Works

Build a miniature BPE from scratch:

```python
# Training corpus
corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are animals",
]
```

1. Start with individual characters as tokens
2. Count all adjacent pairs: ("t", "h"), ("h", "e"), etc.
3. Merge the most frequent pair into a new token
4. Repeat until desired vocabulary size

The algorithm doesn't know that "er" means "comparative" or "est" means "superlative" — it's purely statistical. It just learns that certain character sequences co-occur frequently enough to justify a dedicated token.

---

## Section 2: Tokens Are Vectors

### The Embedding Matrix

Every token ID maps to a vector via a lookup table. Shape: `(vocab_size, d_model)`.

For Qwen 3.6 35B: **248,320 rows × 2,048 columns** = ~1.9 billion parameters (just for the embedding layer).

```python
# Load Qwen's embedding matrix from the model weights
shard_path = hf_hub_download("Qwen/Qwen3.6-35B-A3B", "model-00001-of-00010.safetensors")
weights = safe_open(shard_path, framework="pt", device="cpu")

# The embedding layer is stored under "model.embed_tokens.weight"
embed_weight = weights.get_tensor("model.embed_tokens.weight")
print(f"Embedding shape: {embed_weight.shape}")
# → torch.Size([248320, 2048])
```

### A Closer Look

Pick a few tokens and examine their vectors:

```python
sample_tokens = ["the", "complex", "process", "city", "and", "."]
sample_ids = qwen_tok.convert_tokens_to_ids(sample_tokens)
sample_vectors = embed_weight[sample_ids]
```

Each token is a point in a **2048-dimensional** space. The model doesn't see "the" as a word — it sees a specific coordinate in this learned space. Tokens that appear in similar contexts end up near each other.

---

## Section 3: Exploring the Embedding Space

### t-SNE Visualization

Sample 3,000 tokens and project them into 2D:

```python
from sklearn.manifold import TSNE

n_sample = 3000
indices = rng.choice(embed_weight.shape[0], n_sample, replace=False)
sample_vecs = embed_weight[indices].float().numpy()

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
coords = tsne.fit_transform(sample_vecs)
```

Clear clusters emerge:
- Space-prefixed tokens (start of word) form tight groups
- Subword pieces scatter more widely
- Punctuation clusters in its own region
- Digits form an extremely tight cluster

### Nearest Neighbors

Find tokens that live closest to each other in the embedding space:

```python
def nearest_neighbors(token_str, k=8):
    tid = qwen_tok.convert_tokens_to_ids(token_str)
    vec = embed_weight[tid]
    sims = torch.cosine_similarity(vec.unsqueeze(0), embed_weight, dim=1)
    topk = sims.topk(k + 1)  # +1 because self is always #1
    for score, idx in zip(topk.values[1:], topk.indices[1:]):
        print(f"  {qwen_tok.decode([idx]):15s}  {score:.4f}")
```

**Key insight:** The embedding space captures **distributional similarity** — tokens that appear in similar contexts have similar vectors. This is why "walk" is closer to "run" than to "the", and why digits cluster together.

---

## Interactive Notebook

The full interactive notebook with all visualizations, BPE from scratch, nearest neighbor searches, and t-SNE plots is available [here](https://huggingface.co/datasets/EXDai/episode-07-tokenization).

---

## Summary

1. **The full pipeline**: Text → Token IDs → Embedding vectors → Transformer → Output
2. **BPE tokenization**: Statistical subword merging, not linguistic rules
3. **The embedding matrix**: 248K × 2048 — the model's entire vocabulary as vectors
4. **Embedding space**: Distributional similarity creates meaningful clusters
5. **Nearest neighbors**: Cosine similarity reveals semantic relationships between tokens
