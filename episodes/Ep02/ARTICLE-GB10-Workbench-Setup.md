# GB10 Workbench Setup

Setting up a two-machine AI workbench: a desktop for authoring, and a GPU machine for running models.

---

## The Two-Machine Picture

```
Desktop (Yoneda)               GPU Box (atom)
Ubuntu, VS Code          ←→    NVIDIA Linux, Docker, vLLM
   client                       server, port 8000
```

The GPU machine is reachable via mDNS. A short SSH config alias makes it seamless:

```
Host atom
    HostName <hostname>.local
    User <username>
    ServerAliveInterval 60
```

So `ssh atom` just works.

---

## The GPU Machine

**Hardware:** Gigabyte AI TOP (DGX Spark variant) with NVIDIA Grace-Blackwell Superchip. Unified memory — a single 128 GB pool shared between CPU and GPU. No PCIe bottleneck, no VRAM vs RAM boundaries.

**Software stack:**
- NVIDIA Linux (custom kernel with Grace-Blackwell drivers)
- Docker for all GPU workloads
- vLLM runs inside Docker with `--gpus all --ipc=host`

---

## Pulling a Model

Models land in `~/cache/hf/` and `~/models/`. Docker containers mount these paths so models persist across restarts.

```bash
# Install HF CLI
curl -LsSf https://hf.co/cli/install.sh | bash

# Authenticate
hf auth login

# Pull a model
hf download Qwen/Qwen2.5-7B-Instruct
```

---

## Serving with vLLM

Quick one-shot Docker command:

```bash
docker run -d --name vllm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 \
  -v ~/cache/hf:/root/.cache/huggingface \
  -v ~/models:/models \
  nvcr.io/nvidia/vllm:26.04-py3 \
  vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 --max-model-len 8192
```

Check it's up:

```bash
docker logs -f vllm
# Look for: "Uvicorn running on http://0.0.0.0:8000"
```

Smoke test:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct",
       "messages":[{"role":"user","content":"Hi"}],
       "max_tokens":20}'
```

---

## Config-Driven Harness

Managing raw Docker flags per model gets tedious. A config-driven harness wraps each model recipe in a `.env` file:

```bash
# Start: one command, any model
./scripts/up.sh qwen2.5-7b

# Stop
./scripts/down.sh

# View logs
./scripts/logs.sh

# Smoke test
./scripts/test.sh
```

What happens under the hood:
- Pulls the Docker image if not cached
- Starts a container named `vllm` with `--gpus all`, port `8000`
- Mounts `~/cache/hf` and `~/models` for model access
- Model loads into unified GPU memory

---

## Inference from the Desktop

The GPU machine exposes vLLM on port 8000. Three ways to reach it:

**Option A — Direct via mDNS:**

```bash
curl -s http://<hostname>.local:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Explain backpropagation in one sentence."}],
    "max_tokens": 80
  }'
```

**Option B — SSH tunnel:**

```bash
ssh -N -L 8000:localhost:8000 atom
# Then use localhost:8000
```

**Option C — VS Code Remote-SSH:** Port 8000 forwards automatically.

---

## What This Enables

This is the loop for everything that follows: pick a model, serve it, query it, benchmark it. The workbench is now ready for deep-dives into inference optimization, fine-tuning, and model internals.
