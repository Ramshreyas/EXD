# atom — projects root

Host: GIGABYTE AI TOP ATOM (DGX Spark-class, NVIDIA GB10, 128 GB unified mem, arm64 Ubuntu 24.04).
SSH alias from laptop: `atom` (passwordless).

## Conventions
- One directory per project under `~/projects/`.
- Shared resources live outside projects: `~/models`, `~/datasets`, `~/cache/{hf,torch,uv,pip}`.
- GPU work runs in NGC containers (Blackwell-aware). Default base image:
  `nvcr.io/nvidia/pytorch:26.04-py3` (PyTorch 2.12, CUDA 13.2 fwd-compat).
- Non-GPU / utility / agent code uses `uv` envs in `<project>/.venv`.
- `~/bin/dgpu` — wrapper for interactive GPU containers. `dgpu --` drops you in
  /workspace=cwd with all caches and ~/models mounted.

## Projects
| Path                | Purpose                                  | Status |
|---------------------|------------------------------------------|--------|
| `serve/`            | OpenAI-compatible inference (vLLM)       | live   |

## Quick resume checklist (new session)
1. `groups | grep -q docker || echo "kill .vscode-server and reconnect"`
2. `docker ps`  — see what's running (e.g. `vllm` container)
3. `cat ~/projects/<name>/README.md` for project-specific run commands
4. Repo memory `/memories/atom-spark.md` has hardware/quirk notes
