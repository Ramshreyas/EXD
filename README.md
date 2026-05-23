# EXD — Self-Directed PhD in AI

An open learning log for deep AI engineering mastery. Work backwards from high-level concepts to fundamentals — not bottom-up from theory.

📺 **Follow along:** [YouTube playlist](https://www.youtube.com/playlist?list=PLU1ov53-rMvMrzzRR3rMJVm7pXCoHy2Ik) · [Episode 1: Project Overview](https://youtu.be/mUFNk2yOblc)

## What is this?

EXD is my personal curriculum for going deep on AI engineering. It's not a course — it's a live learning log. I pick a working system, then progressively peel back layers until I understand the internals. Notebooks, benchmarks, configs, and notes all live here.

## Focus Areas

- **Model fine-tuning** — LoRA, QLoRA, full fine-tune, RLHF/DPO
- **Model architectures** — transformers, attention variants, MoE, hybrid models
- **Inference optimization** — quantization, KV cache, speculative decoding, compilation

## Setup

The repo assumes you have two things:

1. **A development machine** — where you write code, notebooks, and docs (VS Code or similar)
2. **A GPU machine** — where models run (a local server, DGX Spark, or cloud VPS with GPUs)

The code you want on the GPU machine lives in `project/`. Sync it however you like — `rsync`, `git pull`, or just clone the whole repo on both machines.

### Quick start

```bash
git clone git@github.com:Ramshreyas/EXD.git
cd EXD
```

If you have a GPU machine with Docker and vLLM:

```bash
# Sync the serve harness to your GPU machine, then:
cd project/serve
./scripts/up.sh          # lists available model configs
./scripts/up.sh qwen2.5-7b  # launch a model
./scripts/test.sh        # smoke test the endpoint
```

## What's in here

```
├── AGENTS.md           # Project context for AI coding agents (pi, Claude Code, etc.)
├── project/            # Code that runs on the GPU machine
│   └── serve/          # vLLM inference harness (config-driven Docker launcher)
│       ├── configs/    # One .env file per model recipe
│       ├── scripts/    # up, down, logs, test
│       ├── parsers/    # Custom reasoning parsers
│       └── benchmarks/ # llama-benchy runbooks + results
├── notebooks/          # Experiments, explorations, notes (future)
└── docs/               # Deeper dives and write-ups (future)
```

## Philosophy

- **Work backwards** — start with a working system, then dig into why and how
- **Everything is documented** — benchmarks, decisions, dead ends included
- **One GPU machine, kept clean** — only runtime files, no IDE cruft
- **Use AI coding agents** — pi, Claude Code, or Cursor with AGENTS.md for context
