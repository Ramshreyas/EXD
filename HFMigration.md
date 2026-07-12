# EXD Hugging Face Organization Migration Plan

This document serves as a high-level blueprint for a coding agent to scaffold the transition of the EXD (Experimental Doctorate) project from a local Git/YouTube-only structure to a unified Hugging Face Organization.

The objective is to establish the infrastructure for a centralized hub hosting articles, datasets, models, and interactive Spaces, executing the migration incrementally episode by episode.

---

## Status (2026-06-24)

### Live on HF

| Asset | Type | Location |
|-------|------|----------|
| **Org Profile** | Space (org card) | `EXDai/README` |
| **Ep3 Simulator** | Static Space | `EXDai/inference-simulator` |
| **Ep4 Simulator** | Static Space | `EXDai/inference-simulator-v2` |
| **Ep5 Simulator** | Static Space | `EXDai/speculative-decoding-simulator` |

### Article Strategy

HF Articles on **organizations** (like `EXD-AI`) require a Team plan. **Individual accounts** (like `EXDai`) support articles without a paid plan — EXDai is an individual account, so articles work natively.

The strategy: publish episode articles directly on `EXDai` as HF Blog posts (accessible at `huggingface.co/EXDai`). These serve as the narrated walkthrough, with links to the interactive Spaces for hands-on exploration.

Episodes with articles:

- `huggingface.co/EXDai` — Ep09: Qwen3.6-35B-A3B Architecture Overview

---

## Phase 1: Organization Initialization & Authentication

### Step 1.1: Environment & CLI Setup ✅
- [x] `huggingface_hub` v1.13.0 installed
- [x] HF_TOKEN configured in `.env`
- [x] Auth verified — token saved to `~/.cache/hf/token`
- [x] Git credential helper set to `store`

### Step 1.2: Namespace Scaffolding ✅
- [x] Org `EXDai` created at `huggingface.co/EXDai`
- [x] Org card Space created at `EXDai/README` — links to GitHub, YouTube, episodes

---

## Phase 2: Core Hub Infrastructure Setup

### Step 2.1: The Interactive Hub (Spaces) ✅
- [x] `EXDai/inference-simulator` — Ep3 pipeline visualization
- [x] `EXDai/inference-simulator-v2` — Ep4 extended version
- [x] `EXDai/speculative-decoding-simulator` — Ep5 spec decode explorer
- [ ] `EXDai/tokenizer-visualizer` — Ep7 notebook conversion (Gradio/Streamlit)

### Step 2.2: The Artifact Hub (Models & Datasets)
- [ ] Create `EXDai/benchmark-results` dataset — performance data and metric sweeps
- [ ] Create `EXDai/vllm-configs` dataset — vLLM configuration profiles

---

## Phase 3: Incremental Episode Migration Framework

For each episode, execute the following:

### 1. Articles → GitHub Links
Articles live on GitHub (no Team plan for HF Blog). Each episode's row in the org card links to the markdown file on GitHub.

### 2. Port Interactivity (Spaces Deployment) ✅ (Ep3-5)
- Identify local UI/UX assets from `episodes/Ep{N}/` (e.g., `inference-simulator.html`)
- Push to a Static Space under the org namespace

### 3. Upload Artifacts (Datasets)
- Check for episode-specific telemetry, environment configurations, or benchmarking text logs
- Push to the org's dataset repos via `huggingface_hub`

---

## Phase 4: Automation & Continuous Integration

- [ ] Draft `scripts/hf_push_util.py` — reusable Python utility that abstracts the HF API for quick uploads