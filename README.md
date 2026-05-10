# Grimoire

Self-hosted AI inference infrastructure for multi-GPU llama.cpp serving.

## Architecture

```
┌──────────────┐    HTTPS/v1      ┌──────────────────┐
│   OpenCode   │ ───────────────► │ chat.lost.plus   │
│   (local)    │                  │ (Cloudflare Tunnel)│
└──────────────┘                  └────────┬─────────┘
                                           │
                                 ┌─────────▼──────────┐
                                 │  grimoire container │ :9001
                                 │  (gateway + models) │
                                 └──┬──────┬──────┬───┘
                                    │      │      │
                         ┌──────────▼─┐ ┌──▼────┐ ┌▼────────────┐
                         │  GPU 0     │ │ GPU 1 │ │ GPU N       │
                         │  model A   │ │ model │ │ model Z     │
                         └────────────┘ └───────┘ └─────────────┘
```

## Features

- **Multi-GPU** — Run multiple models simultaneously, one per GPU
- **Dynamic GPU allocation** — Free GPU preferred, oldest non-pinned model evicted when all GPUs busy
- **Pinned models** — Fix specific models to specific GPUs via `fixed` section
- **Model registry** — JSON-based registry with per-model settings
- **Model ingestion** — Download and register new models via CLI or API
- **OpenAI-compatible API** — Standard `/v1/chat/completions` with automatic routing

## Usage

```bash
# Start with a specific model
docker run --gpus all -p 9001:9001 -v /path/to/models:/models \
  ghcr.io/lpfchan/grimoire:latest --model qwen-3.6-27B

# List registered models
docker exec grimoire grimoire list

# Ingest a new model
docker exec grimoire grimoire ingest --alias "my-model" --url "https://..."

# Pin a model to GPU 1
docker exec grimoire grimoire pin gemma-4-31B 1

# Unpin
docker exec grimoire grimoire unpin gemma-4-31B

# Switch models via API
curl -X POST http://localhost:9001/switch/qwen-3.6-27B

# Chat completions
curl -X POST http://localhost:9001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen-3.6-27B", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Model Registry

`/etc/grimoire/models.json`:

```json
{
  "models": {
    "qwen-3.6-27B": {
      "file": "gguf/Qwen3.6-27B-UD-Q4_K_XL.gguf",
      "mmproj": "gguf/Qwen3.6-27B-mmproj-BF16.gguf",
      "ctx-size": 262144,
      "cache-type-k": "turbo4",
      "cache-type-v": "turbo4"
    }
  },
  "fixed": {
    "gemma-4-31B": 1
  }
}
```

- `models` — model definitions (no GPU assignment)
- `fixed` — model alias → GPU ID (pinned, never evicted)
- Models not in `fixed` use dynamic LRU allocation

## Building

```bash
docker build -t grimoire:latest .
```
