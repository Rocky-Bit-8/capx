# Molmo Pointing Service

CaP-X calls Molmo through an OpenAI-compatible local endpoint:

```text
http://127.0.0.1:8122/v1/chat/completions
```

It is deliberately a separate, long-running process. Do not add it to every task YAML: loading an 8B VLM at each task start is slow and competes for GPU memory with SAM3 and Contact-GraspNet.

## One-time install

Use an isolated serving environment. The current vLLM release requires a newer
PyTorch than CaP-X, so installing it into `.setup-venv` would replace the
tested CUDA 12.8 / PyTorch 2.8 stack. This does not download model weights.

```bash
cd /home/rocky/Code/cap-x-main && uv venv .molmo-venv --python 3.10 && UV_CACHE_DIR=/tmp/capx-uv-cache uv pip install --python .molmo-venv/bin/python vllm
```

If the package download is inconvenient, download the matching wheel from:

```text
https://pypi.org/project/vllm/#files
```

The Molmo model files are from the official repository:

```text
https://huggingface.co/allenai/Molmo2-8B
```

The already downloaded model in this workspace is located at:

```text
/home/rocky/Code/cap-x-main/molmo/allenai--Molmo2-8B/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b
```

For a fresh download, the repository may instead be placed at
`/home/rocky/Code/cap-x-main/models/Molmo2-8B`.

With the Hugging Face CLI already in the environment, the direct command is:

```bash
cd /home/rocky/Code/cap-x-main && mkdir -p models && ./.setup-venv/bin/hf download allenai/Molmo2-8B --local-dir models/Molmo2-8B
```

The FP16 checkpoint is roughly 16 GB; leave additional GPU memory for runtime cache, SAM3, and Contact-GraspNet.

## Start service

Start this once in a dedicated terminal before the CaP-X Web UI or evaluation command:

```bash
cd /home/rocky/Code/cap-x-main && .molmo-venv/bin/python -m vllm.entrypoints.openai.api_server --model /home/rocky/Code/cap-x-main/molmo/allenai--Molmo2-8B/e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b --served-model-name allenai/Molmo2-8B --host 127.0.0.1 --port 8122 --dtype bfloat16 --gpu-memory-utilization 0.45 --max-model-len 4096 --trust-remote-code
```

`0.45` keeps GPU headroom for the existing perception services. Raise it only when Molmo is the only GPU workload.

## Check service

```bash
curl -fsS http://127.0.0.1:8122/v1/models
```

When it returns JSON containing `allenai/Molmo2-8B`, CaP-X's `point_prompt_molmo(...)` API is available again. A refused connection means the Molmo process is not running; it is independent of the CaP-X Web UI process.
