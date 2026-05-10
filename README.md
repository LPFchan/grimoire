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

- **Multi-GPU support** - Run multiple models simultaneously, one per GPU
- **Model registry** - JSON-based registry with model metadata, GPU assignment, and per-model settings
- **Model ingestion** - Download and register new models via CLI or API
- **OpenAI-compatible API** - Standard `/v1/chat/completions` endpoint with model routing
- **Model switching** - Start/stop models on demand via `/switch/{model_name}`
- **Health monitoring** - Built-in health checks and status endpoints

## Usage

```bash
# Start with a specific model
docker run --gpus all -p 9001:9001 -v /path/to/models:/models \
  ghcr.io/lpfchan/grimoire:latest --model qwen-3.6-27B

# List registered models
docker exec grimoire grimoire list

# Ingest a new model
docker exec grimoire grimoire ingest --alias "my-model" --url "https://..." --gpu 0

# Switch models via API
curl -X POST http://localhost:9001/switch/qwen-3.6-27B

# Chat completions
curl -X POST http://localhost:9001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen-3.6-27B", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Model Registry

Models are defined in `/etc/grimoire/models.json`:

```json
{
  "qwen-3.6-27B": {
    "file": "gguf/Qwen3.6-27B-UD-Q4_K_XL.gguf",
    "mmproj": "gguf/Qwen3.6-27B-mmproj-BF16.gguf",
    "ctx-size": 262144,
    "gpu": 0,
    "cache-type-k": "turbo4",
    "cache-type-v": "turbo4"
  },
  "gemma-4-31B": {
    "file": "gguf/gemma-4-31B-it-UD-Q4_K_XL.gguf",
    "mmproj": "gguf/mmproj-BF16.gguf",
    "ctx-size": 131072,
    "gpu": 1,
    "cache-type-k": "turbo4",
    "cache-type-v": "turbo4"
  }
}
```

## Building

```bash
docker build -t grimoire:latest .
```

## Development

```bash
# Install dependencies
pip install -e .

# Run gateway locally
python -m grimoire.entrypoint --port 9001

# CLI commands
grimoire list
grimoire ingest --alias "test" --url "https://..." --gpu 0
grimoire status
```
