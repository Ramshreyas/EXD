# serve — local OpenAI-compatible inference (vLLM on GB10)

Config-driven launcher. One generic script, one `.env` file per model.

## Layout
```
scripts/{up,down,logs,test}.sh
configs/<name>.env          # one per model recipe
configs/active              # symlink → currently-running config
parsers/                    # model-specific assets mounted into the container
```

## Run
    ./scripts/up.sh <config-name>     # e.g. qwen2.5-7b, nemotron-3-super
    ./scripts/up.sh                   # lists available configs
    ./scripts/down.sh
    ./scripts/logs.sh
    ./scripts/test.sh                 # uses configs/active
    ./scripts/test.sh <model-id>      # explicit override

Container name is always `vllm`. Restart policy: unless-stopped.

## Endpoint
    http://localhost:8000/v1          # OpenAI-compatible
    http://localhost:8000/metrics     # Prometheus

From laptop: VS Code Remote-SSH auto-forwards port 8000 — just hit
`http://localhost:8000` on the laptop. Manual fallback: `ssh -L 8000:localhost:8000 atom`.
Any API key string works.

## Available configs
| Name                | Model                                                 | Image                              | Notes |
|---------------------|-------------------------------------------------------|------------------------------------|-------|
| `qwen2.5-7b`        | Qwen/Qwen2.5-7B-Instruct                              | nvcr.io/nvidia/vllm:26.04-py3      | bf16, 8k ctx, fast |
| `qwen3.6-27b`       | Qwen/Qwen3.6-27B                                      | vllm/vllm-openai:v0.20.0           | bf16, 128k ctx, multimodal, reasoning+tools+MTP |
| `qwen3.6-35b-a3b`   | Qwen/Qwen3.6-35B-A3B                                  | vllm/vllm-openai:v0.20.0           | optimized bf16 MoE default, 128k ctx, text-only, reasoning+tools+MTP2 |
| `qwen3.6-35b-a3b-baseline` | Qwen/Qwen3.6-35B-A3B                           | vllm/vllm-openai:v0.20.0           | initial 0.80 memory-util recipe, preserved for comparison |
| `qwen3.6-35b-a3b-throughput` | Qwen/Qwen3.6-35B-A3B                         | vllm/vllm-openai:v0.20.0           | higher batch window for aggregate throughput; heavier cold start |
| `qwen3.6-35b-a3b-mtp3` | Qwen/Qwen3.6-35B-A3B                              | vllm/vllm-openai:v0.20.0           | MTP3 profile for batch-heavy c4 throughput tests |
| `nemotron-3-super`  | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4        | vllm/vllm-openai:v0.20.0           | NVFP4, 256k ctx, hybrid Mamba+MoE+MTP, reasoning ON/OFF |

## Adding a new model
1. Drop a new file in `configs/<name>.env` defining:
   - `MODEL`, `SERVED_NAME`, `IMAGE`, `GPU_MEM_UTIL`
   - `EXTRA_DOCKER_ARGS=( ... )` — extra `-e` env vars, `-v` mounts, etc.
   - `EXTRA_VLLM_ARGS=( ... )` — extra `vllm serve` flags
   - `CMD_PREFIX=()` if the image's entrypoint already runs `vllm serve`
     (e.g. `vllm/vllm-openai`); otherwise omit (defaults to `(vllm serve)`).
2. `./scripts/up.sh <name>`. That's it.

Per-model assets (reasoning parsers, custom code) live in `parsers/` and are
mounted into the container via `EXTRA_DOCKER_ARGS`.

## HF auth
If `~/.hf_token` exists (KEY=VALUE format containing `HF_TOKEN=hf_...`),
`up.sh` automatically passes it via `--env-file`. Required for gated models.

## Qwen3.6-27B notes
- Initial recipe uses multimodal serving at 128k context. The model card lists
  262k native context, but 128k is a safer first target on Spark's shared
  128 GB memory. If stable, try `--max-model-len 262144` later.
- Reasoning is ON by default and uses vLLM's built-in `qwen3` parser. To disable
  thinking per request:
  `extra_body={"chat_template_kwargs": {"enable_thinking": false}}`
- For agent loops that should retain earlier thinking traces:
  `extra_body={"chat_template_kwargs": {"preserve_thinking": true}}`
- Tool calling is enabled with `--enable-auto-tool-choice` and
  `--tool-call-parser qwen3_coder`.
- MTP speculative decoding is enabled with
  `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`.
- Recommended sampling from the model card:
  - General thinking: `temperature=1.0`, `top_p=0.95`, `top_k=20`
  - Precise coding: `temperature=0.6`, `top_p=0.95`, `top_k=20`
  - Non-thinking: `temperature=0.7`, `top_p=0.80`, `top_k=20`,
    `presence_penalty=1.5`
- If boot fails from memory pressure, first lower `--max-model-len` to `65536`.
  If MTP fails, remove only `--speculative-config` and keep reasoning/tool
  parsing intact. If multimodal memory is the blocker, add `--language-model-only`
  as a text-only fallback.

## Qwen3.6-35B-A3B notes
- The model card's vLLM recipe recommends `vllm>=0.19.0`, 262k native context,
  `--reasoning-parser qwen3`, tool use via `--enable-auto-tool-choice` and
  `--tool-call-parser qwen3_coder`, and MTP via
  `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`.
- Optimized Spark default uses 128k context, `--language-model-only`,
  `GPU_MEM_UTIL=0.85`, `--max-num-seqs 16`, `--max-num-batched-tokens 8192`,
  and MTP2. This was selected from llama-benchy short-suite results as the best
  balanced interactive/concurrent profile.
- The initial Spark recipe is preserved as `qwen3.6-35b-a3b-baseline`.
- If stable and memory allows, try removing `--language-model-only` for
  image/video input.
- The model is 35B total / 3B active MoE in bf16. If boot fails from memory
  pressure, first lower `--max-model-len` to `65536`. If MTP fails, remove only
  `--speculative-config` and keep reasoning/tool parsing intact.
- For contexts beyond 262k, the card recommends YaRN via
  `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, `--hf-overrides` rope parameters, and a
  larger `--max-model-len`; do this only for dedicated long-context runs.

## Nemotron-3-Super notes
- Reasoning ON by default. To disable per-request:
  `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`
- Recommended sampling: `temperature=1.0, top_p=0.95` for all tasks.
- First boot downloads ~75 GiB (~10 min) then loads 17 shards (~8 min) +
  torch.compile + CUDA graph capture. Subsequent boots reuse the HF cache and
  start in ~10–11 min.
- `GPU_MEM_UTIL=0.80` on Spark's 128 GB unified memory. Higher (e.g. 0.90 from
  the NVIDIA cookbook, written for discrete VRAM) starves the host and causes
  swap pressure.
- Required env vars (set in `EXTRA_DOCKER_ARGS`):
  `VLLM_NVFP4_GEMM_BACKEND=marlin`, `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`,
  `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm`, `VLLM_USE_FLASHINFER_MOE_FP4=0`.
- Custom reasoning parser at `parsers/super_v3_reasoning_parser.py` is
  mounted into `/app/` and registered via `--reasoning-parser-plugin`.
- `vllm/vllm-openai` image's entrypoint already runs `vllm serve`, hence
  `CMD_PREFIX=()` in this config.

### MTP speculative decoding (enabled)
`--speculative_config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'`
is on by default. Cold-start request after `up.sh` is slow (~45 s, graph
capture for the spec path); steady-state is what counts.

### Single-stream perf (500-token essay, reasoning ON, GB10)
| Config                   | Time   | Throughput  | Spec accept |
|--------------------------|--------|-------------|-------------|
| Baseline (no MTP)        | 31.5 s | ~15.9 tok/s | —           |
| MTP, num_spec_tokens=3   | 21.8 s | ~22.9 tok/s | 52% (1.57/3)|

Live spec metrics: `curl -s localhost:8000/metrics | grep spec_decode`.

### Deferred / future tuning
- 1 M context: add `--max-model-len 1000000` (replace `262144`). Will reduce
  KV cache headroom — re-check `GPU_MEM_UTIL`.
- Try `--max-num-batched-tokens 4096` (vLLM warns 2048 is suboptimal with
  spec decoding).
- Sweep `num_speculative_tokens` ∈ {2,3,4} for best wall-clock.
- Concurrency benchmark from `/metrics` at QPS 1/2/4.
