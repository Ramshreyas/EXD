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

```bash
# atom
cd projects/serve
ls configs/
```

Each `.env` file is a model recipe: image, model ID, GPU budget, extra flags.
Same thing as above, but now just:

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

atom exposes vLLM on port 8000. The container binds to `0.0.0.0:8000`, but
we don't expose it to the wider network — we tunnel.

### Option A: SSH tunnel (manual, works everywhere)

```bash
# yoneda (separate terminal, keep it open)
ssh -L 8000:localhost:8000 atom
```

Now `localhost:8000` on Yoneda forwards to atom's port 8000. In another
Yoneda terminal:

```bash
# yoneda
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

```bash
# yoneda
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Explain backpropagation in one sentence."}],
    "max_tokens": 80,
    "temperature": 0.2
  }' | python3 -m json.tool
```

### Option B: VS Code Remote-SSH (auto-forwards)

If you use VS Code's Remote-SSH to work on atom, port 8000 is forwarded
automatically. Just hit `localhost:8000` from Yoneda — no manual tunnel.

### Option C: Direct (if you trust your LAN)

Since the container binds `0.0.0.0`, you can hit the mDNS hostname directly
(find yours with `ssh -G atom | grep hostname`):

```bash
# yoneda
curl -s http://<mdns-hostname>.local:8000/v1/models | python3 -m json.tool
```

Pick the one that fits your setup. For the rest of this episode I'll assume
the SSH tunnel.

---

## 7. Quick benchmark

A lightweight latency/throughput check using the OpenAI-compatible API.
Save this as `bench.py` on Yoneda:

```python
# yoneda: bench.py
import time, requests

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-7B-Instruct"

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Write a 200-word essay about GPUs."}],
    "max_tokens": 256,
    "temperature": 0.0,
}

start = time.monotonic()
resp = requests.post(URL, json=payload)
resp.raise_for_status()
elapsed = time.monotonic() - start

body = resp.json()
choice = body["choices"][0]
ttft = None  # time to first token — requires streaming, see below
completion_tokens = body["usage"]["completion_tokens"]
total_tokens = body["usage"]["total_tokens"]

print(f"Status:         {resp.status_code}")
print(f"Time (wall):    {elapsed:.2f} s")
print(f"Prompt tokens:  {body['usage']['prompt_tokens']}")
print(f"Completion:     {completion_tokens} tokens")
print(f"Throughput:     {completion_tokens / elapsed:.1f} tok/s")
print(f"---")
print(choice["message"]["content"][:200])
```

Run it:

```bash
# yoneda
python3 bench.py
```

For proper TTFT (time-to-first-token) and per-token latency, use streaming:

```python
# yoneda: bench_stream.py
import time, requests, json

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-7B-Instruct"

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Write a 200-word essay about GPUs."}],
    "max_tokens": 256,
    "temperature": 0.0,
    "stream": True,
}

tokens = []
ttft = None
start = time.monotonic()

with requests.post(URL, json=payload, stream=True) as resp:
    resp.raise_for_status()
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if line == "data: [DONE]":
            break
        if line.startswith("data: "):
            chunk = json.loads(line[6:])
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                if ttft is None:
                    ttft = time.monotonic() - start
                tokens.append(content)

elapsed = time.monotonic() - start

print(f"TTFT:           {ttft*1000:.0f} ms")
print(f"Total time:     {elapsed:.2f} s")
print(f"Tokens:         {len(tokens)}")
print(f"Throughput:     {len(tokens) / elapsed:.1f} tok/s")
print(f"Tok/tok (gen):  {(elapsed - ttft) / len(tokens) * 1000:.0f} ms avg")
```

```bash
# yoneda
python3 bench_stream.py
```

---

## 8. What we just did

```
atom:   pulled a model → served it with vLLM on port 8000
Yoneda: tunnelled in → sent chat requests → measured throughput
```

This is the loop for everything that follows: pick a model, serve it, query
it, benchmark it. Next episode: we dig into what the model actually *does*
— embeddings, attention, and the forward pass.
