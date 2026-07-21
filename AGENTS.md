# EXD — Self-Directed PhD in AI

EXD is an open learning log for deep AI engineering mastery. Work backwards from high-level concepts to fundamentals — not bottom-up from theory.

## Machines

| Name | Role | Access |
|------|------|--------|
| **Yoneda** | Desktop (where pi runs, where EXD lives) | — |
| **atom** | Gigabyte AI TOP (DGX Spark variant) — runs models, training, inference | `ssh atom` |

Work locally on Yoneda in VS Code. Keep a plain SSH terminal open to atom for running commands.

Repo: `github.com/Ramshreyas/EXD`

## Directory Layout

- **`/home/ramshreyas/Documents/Dev/EXD/`** (Yoneda) — Git repo root. All notebooks, code, documentation, experiments.
- **`EXD/projects/`** — Atom-bound code: serve harness, configs, benchmarks. This is what gets synced to atom.
- **`~/EXD/`** (atom) — Git clone of this repo. Keep atom clean — only this clone. Source of truth is Yoneda.
- **`EXD/episodes/`** — YouTube follow-along materials: code, docs, and other assets for the video series accompanying this project.
- **`~/EXD/projects/serve/`** (atom) — vLLM inference harness. Config-driven: `./scripts/up.sh <config-name>`

## Focus Areas

- Model architectures (transformers, attention variants, MoE, etc.)
- Model fine-tuning (LoRA, QLoRA, full fine-tune, RLHF/DPO)
- Inference optimization (quantization, KV cache, speculative decoding, compilation)
- Work backwards: start with a working high-level thing, then progressively peel back layers to understand internals

## Teaching Approach

### Paper-First Architecture Deep Dives

Each architecture episode starts with a research paper as the map. The paper provides the complete picture at altitude; then we descend into the machinery.

### HF Spaces as the Primary Explanation Surface

Since Ep09 onward, each episode ships with an interactive HF Space as its main deliverable. The Space is the visual anchor — diagrams, interactive knobs, live demos. Local code/notebooks on Yoneda support it, and SSH connections to atom are reserved for live inference demos.

### The Guided Tour → Deep Dive Pipeline

Episode format evolves into two tiers:

| Tier | Pattern | Purpose |
|------|---------|---------|
| **Guided Tour** (odd-numbered, ~Ep09, Ep13, ...) | One episode covering an entire paper at altitude | Build the mental map; ship a central interactive reference Space |
| **Deep Dive** (even-numbered, ~Ep10-12, ...) | One episode per component from the tour | Peel back one layer; extend the central Space with per-component interactive modules |

This directly enforces "work backwards." The guided tour shows the whole machine running. Deep dives yank out individual gears and hold them up to the light.

## Episodes

| # | Status | Title | Notes |
|---|--------|-------|-------|
| 01 | ✅ Complete | Intro to EXD | Project philosophy, learning approach, what EXD is and isn't |
| 02 | ✅ Complete | Setup Overview | Machines (Yoneda + atom), directory layout, workflow, tooling |
| 03 | ✅ Complete | Inference Benchmarking (Intro) | Prefill vs decode mental model, vLLM bench, llama-benchy setup |
| 04 | ✅ Complete | Performance Tuning | Concurrency & batching, vLLM flag sweeps (GPU mem, scheduler, MTP depth), long-context prefill, prefix caching |
| 05 | ✅ Complete | Speculative Decoding | Draft-verify loop, MTP depth tradeoffs, interactive simulator Space |
| 06 | ⬜ Planned | *(gap / placeholder)* | — |
| 07 | ✅ Complete | Tokenization & Embeddings | BPE, token→vector lookup, embedding spaces, qwen tokenizer internals |
| 08 | ✅ Complete | Positional Embeddings | Sinusoidal → learned → RoPE, how attention knows token order |
| 09 | ✅ Complete | Qwen 3 Architecture — A Guided Tour | Qwen3 technical report as map; interactive HF Space architecture diagram; sets up deep-dive pipeline for Ep10+ |
| 10 | ✅ Complete | mRoPE — The Geometry of Position | Deep dive on multi-resolution RoPE; build from scratch, rotate real Qwen embeddings, visual PCA orbits, translation invariance demo |

## Workflow

1. Write code, notebooks, and docs locally on **Yoneda** in VS Code
2. Commit and push to GitHub: `git push`
3. Pull on **atom**: `ssh atom 'cd ~/EXD && git pull'`
4. Run experiments on atom via the SSH terminal
5. Results, analysis, and notes come back to EXD on Yoneda
