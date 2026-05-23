# Ep. 2 — Setup & First Inference

> I call my GPU machine **atom** and my desktop **Yoneda**. Swap in your own
> hostnames and paths — the rest is the same.

---

## 1. The two-machine picture

```
Yoneda (desktop)          atom (GPU box)
Ubuntu, VS Code     ←→    NVIDIA Linux, Docker, vLLM
   client                  server, port 8000
```

atom is on the local network and reachable via mDNS. Rather than typing the
full mDNS hostname every time, `~/.ssh/config` maps a short alias:

```ssh-config
Host atom
    HostName <mdns-hostname>.local
    User <your-username>
    ServerAliveInterval 60
```

So `ssh atom` just works. No IP hunting, no `/etc/hosts` edits.

```bash
# yoneda
cat ~/.ssh/config 
```

---

## 2. Yoneda — the client

This is where everything is authored: code, docs, this markdown file. Keep
a terminal open here — all client-side commands run from this box.

Project layout (this repo):

```bash
# yoneda
tree 
```

Quick sanity check that atom is alive:

```bash
# yoneda
ssh atom echo ok
```

---

## 3. atom — the GPU machine (deep dive)

### Hardware

Gigabyte AI TOP (DGX Spark variant):
- NVIDIA Grace-Hopper Superchip
- **Unified memory**: single 128 GB pool shared between CPU and GPU
  — no PCIe bottleneck, CPU and GPU see the same physical memory
- This is why we can run large models without worrying about VRAM vs. RAM
  boundaries

### Software stack

- **NVIDIA Linux** (custom kernel with Grace-Hopper drivers)
- **Docker** as the runtime for all GPU workloads — clean isolation,
  reproducible environments
- **vLLM** runs inside Docker with `--gpus all --ipc=host`

SSH in to poke around:

```bash
# yoneda
ssh atom
```

Once on atom, check what's under the hood:

```bash
# atom
nvidia-smi
```

Note: Memory reads "Not Supported" — that's expected. Grace-Hopper uses
**unified memory**, so the GPU doesn't have its own VRAM. The system RAM
*is* GPU memory:

```bash
# atom
free -h
```

Other specs:

```bash
# atom
lscpu | head -30        # ARM-based Grace CPU
uname -a                # NVIDIA custom kernel
docker ps               # what's running
```

---

## 4. Pulling a model

Models land in `~/cache/hf/` (HuggingFace cache) and `~/models/` on atom.
The Docker container mounts these so models persist across restarts.

Install the HF CLI if you haven't:

```bash
# atom
curl -LsSf https://hf.co/cli/install.sh | bash
```

Authenticate (needed for gated models like Llama, Nemotron):

```bash
# atom
hf auth login
```

Check what's already cached:

```bash
# atom
hf cache ls
```

Pull a model:

```bash
# atom
hf download Qwen/Qwen2.5-7B-Instruct
```

---

## 5. Serving with vLLM

Quickest way to get vLLM running — a single Docker command:

```bash
# atom
docker run -d --name vllm-test --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 \
  -v ~/cache/hf:/root/.cache/huggingface \
  -v ~/models:/models \
  -e HF_HOME=/root/.cache/huggingface \
  nvcr.io/nvidia/vllm:26.04-py3 \
  vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --served-model-name Qwen/Qwen2.5-7B-Instruct \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 --max-model-len 8192
```

Check it's up (first boot takes a minute or two while the model loads):

```bash
# atom
docker logs -f vllm-test
# look for: "Uvicorn running on http://0.0.0.0:8000"
# then Ctrl+C — the model is ready
```

Smoke test:

```bash
# atom
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct",
       "messages":[{"role":"user","content":"Hi"}],
       "max_tokens":20}' | python3 -m json.tool
```

Stop it:

```bash
# atom
docker rm -f vllm-test
```

That works, but managing flags per model gets tedious fast. Instead, the
repo has a config-driven harness at `projects/serve/`:

```bash
# atom
cd ~/EXD
tree
```

Each `.env` file is a model recipe. Here's the 7B config:

```bash
# atom
cat projects/serve/configs/qwen2.5-7b.env
```

Same thing as the raw Docker command, but now just:

```bash
# atom
./scripts/up.sh qwen2.5-7b
```

What happens:
- Pulls the Docker image if not cached (first time only)
- Starts a container named `vllm` with `--gpus all`, port `8000`
- Mounts `~/cache/hf` and `~/models` so the model is visible inside the container
- Model loads into GPU memory (unified, so it "just fits")
- GPU memory utilization is capped at `0.85` by default

Wait for it to come up:

```bash
# atom
docker logs -f vllm
# look for: "Uvicorn running on http://0.0.0.0:8000"
# Ctrl+C once you see it
```

Same smoke test, same endpoint:

```bash
# atom
./scripts/test.sh
```

Stop it:

```bash
# atom
./scripts/down.sh
```

For the video, keep it running — we'll hit it from Yoneda next.

---

## 6. Inference from Yoneda

atom exposes vLLM on port 8000. Since the container binds `0.0.0.0` and
atom is on the local network, we can hit it directly via mDNS.

### Option A: Direct via mDNS

Find your GPU machine's mDNS hostname:

```bash
# yoneda
ssh -G atom | grep hostname
```

Hit it:

```bash
# yoneda
curl -s http://aitopatom-0a62.local:8000/v1/models | python3 -m json.tool
```

```bash
# yoneda
curl -s http://aitopatom-0a62.local:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Explain backpropagation in one sentence."}],
    "max_tokens": 80,
    "temperature": 0.2
  }' | python3 -m json.tool
```

### Option B: SSH tunnel

If you prefer to keep everything on `localhost`:

```bash
# yoneda (separate terminal, keep it open)
ssh -N -L 8000:localhost:8000 atom
# -N: tunnel only, no remote shell. It'll just sit there — that's correct.
```

Then use `localhost:8000` in the same curl commands above.

### Option C: VS Code Remote-SSH (auto-forwards)

If you use VS Code's Remote-SSH to work on atom, port 8000 is forwarded
automatically — just hit `localhost:8000`.

---

## 7. What we just did

```
atom:   pulled a model → served it with vLLM on port 8000
Yoneda: tunnelled in → sent chat requests → measured throughput
```

This is the loop for everything that follows: pick a model, serve it, query
it, benchmark it. Next episode: we dig into what the model actually *does*
— embeddings, attention, and the forward pass.
