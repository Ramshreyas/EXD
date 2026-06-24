# EXD Hugging Face Organization Migration Plan

This document serves as a high-level blueprint for a coding agent to scaffold the transition of the EXD (Experimental Doctorate) project from a local Git/YouTube-only structure to a unified Hugging Face Organization.

The objective is to establish the infrastructure for a centralized hub hosting articles, datasets, models, and interactive Spaces, executing the migration incrementally episode by episode.

---

## Status (2026-06-24)

### Live on HF

| Asset | Type | Location |
|-------|------|----------|
| **Org Profile** | Space (org card) | `EXD-AI/README` |
| **Ep3 Simulator** | Static Space | `EXD-AI/inference-simulator` |
| **Ep4 Simulator** | Static Space | `EXD-AI/inference-simulator-v2` |
| **Ep5 Simulator** | Static Space | `EXD-AI/speculative-decoding-simulator` |

### Article Strategy

HF Articles require a Team plan. All episode write-ups are linked directly from the org card to their GitHub source:

- `github.com/Ramshreyas/EXD/blob/main/episodes/Ep{N}/`

---

## Phase 1: Organization Initialization & Authentication

### Step 1.1: Environment & CLI Setup ✅
- [x] `huggingface_hub` v1.13.0 installed
- [x] HF_TOKEN configured in `.env`
- [x] Auth verified — token saved to `~/.cache/hf/token`
- [x] Git credential helper set to `store`

### Step 1.2: Namespace Scaffolding ✅
- [x] Org `EXD-AI` created at `huggingface.co/EXD-AI`
- [x] Org card Space created at `EXD-AI/README` — links to GitHub, YouTube, episodes

---

## Phase 2: Core Hub Infrastructure Setup

### Step 2.1: The Interactive Hub (Spaces) ✅
- [x] `EXD-AI/inference-simulator` — Ep3 pipeline visualization
- [x] `EXD-AI/inference-simulator-v2` — Ep4 extended version
- [x] `EXD-AI/speculative-decoding-simulator` — Ep5 spec decode explorer
- [ ] `EXD-AI/tokenizer-visualizer` — Ep7 notebook conversion (Gradio/Streamlit)

### Step 2.2: The Artifact Hub (Models & Datasets)
- [ ] Create `EXD-AI/benchmark-results` dataset — performance data and metric sweeps
- [ ] Create `EXD-AI/vllm-configs` dataset — vLLM configuration profiles

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