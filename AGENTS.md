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

- Model fine-tuning (LoRA, QLoRA, full fine-tune, RLHF/DPO)
- Model architectures (transformers, attention variants, MoE, etc.)
- Inference optimization (quantization, KV cache, speculative decoding, compilation)
- Work backwards: start with a working high-level thing, then progressively peel back layers to understand internals

## Episodes

| # | Status | Title | Notes |
|---|--------|-------|-------|
| 01 | ✅ Complete | Intro to EXD | Project philosophy, learning approach, what EXD is and isn't |
| 02 | ✅ Complete | Setup Overview | Machines (Yoneda + atom), directory layout, workflow, tooling |
| 03 | ✅ Complete | Inference Benchmarking (Intro) | Prefill vs decode mental model, vLLM bench, llama-benchy setup |
| 04 | ✅ Complete | Performance Tuning | Concurrency & batching, vLLM flag sweeps (GPU mem, scheduler, MTP depth), long-context prefill, prefix caching |

## Workflow

1. Write code, notebooks, and docs locally on **Yoneda** in VS Code
2. Commit and push to GitHub: `git push`
3. Pull on **atom**: `ssh atom 'cd ~/EXD && git pull'`
4. Run experiments on atom via the SSH terminal
5. Results, analysis, and notes come back to EXD on Yoneda
