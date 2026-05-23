# EXD — Self-Directed PhD in AI

EXD is an open learning log for deep AI engineering mastery. Work backwards from high-level concepts to fundamentals — not bottom-up from theory.

## Machines

| Name | Role | Access |
|------|------|--------|
| **Yoneda** | Desktop (where pi runs, where EXD lives) | — |
| **atom** | Gigabyte AI TOP (DGX Spark variant) — runs models, training, inference | `ssh atom` as `ramshreyas` |

A remote session to atom is typically open in another terminal. When running inside a VS Code terminal, `ssh atom` works directly.

## Directory Layout

- **`/home/ramshreyas/Documents/Dev/EXD/`** (Yoneda) — All notebooks, code, documentation, experiments. This is the project root.
- **`~/project/`** (atom) — Project code that needs to run on atom. Keep atom clean — only files required for execution live here. Source of truth for code lives on Yoneda in EXD.
- **`~/project/serve/`** (atom) — Minimal harness for running models using config files. Inspect it to understand how models are served.

## Focus Areas

- Model fine-tuning (LoRA, QLoRA, full fine-tune, RLHF/DPO)
- Model architectures (transformers, attention variants, MoE, etc.)
- Inference optimization (quantization, KV cache, speculative decoding, compilation)
- Work backwards: start with a working high-level thing, then progressively peel back layers to understand internals

## Workflow

1. Write code, notebooks, and docs on **Yoneda** in the EXD directory
2. When something needs to run on the GPU, push relevant code to **atom** (`~/project/`)
3. Run experiments on atom; keep it minimal and clean
4. Results, analysis, and notes come back to EXD on Yoneda
