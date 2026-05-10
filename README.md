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
- **Canonical model switcher** — Built-in web UI and API for loading/switching models
- **Server-side history** — Per-api-key conversation history stored in SQLite
- **Token/cost tally** — Per-api-key and global token/cost accounting with legacy import
- **Pinned models** — Fix specific models to specific GPUs via `fixed` section
- **Model registry** — JSON-based registry with per-model settings
- **Safe model ingestion** — Download and register HTTPS models via CLI or authenticated API
- **Protected API** — `/v1/*`, history, stats, and management endpoints require API/admin auth
- **OpenAI-compatible API** — Standard `/v1/chat/completions` with automatic routing

## Usage

```bash
# Start with a specific model
docker run --name grimoire --gpus all -p 9001:9001 \
  -e GRIMOIRE_API_KEY="change-me" \
  -v /path/to/models:/models \
  -v /home/yeowool/templates:/templates:ro \
  -v /home/yeowool/structured-cot/grammars:/etc/grimoire/grammars:ro \
  -v grimoire-state:/var/lib/grimoire \
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
curl -X POST http://localhost:9001/switch/qwen-3.6-27B \
  -H "Authorization: Bearer change-me"

# Chat completions
curl -X POST http://localhost:9001/v1/chat/completions \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen-3.6-27B", "messages": [{"role": "user", "content": "Hello"}]}'
```

API endpoints require `Authorization: Bearer ...` or `X-Grimoire-Token` when `GRIMOIRE_API_KEY` is set.
Management endpoints use `GRIMOIRE_ADMIN_TOKEN` if set, otherwise the API key.

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

## Ingest Safety

`/ingest` and `grimoire ingest` use HTTPS by default, reject private/non-routable hosts, write downloads atomically, and enforce a maximum size.

Environment controls:

- `GRIMOIRE_INGEST_MAX_BYTES` — maximum download size, default 80 GiB
- `GRIMOIRE_ALLOW_HTTP_INGEST=1` — allow plain HTTP URLs
- `GRIMOIRE_ALLOW_PRIVATE_INGEST=1` — allow private or loopback targets

## Stats Migration

Legacy `/home/yeowool/token-stats.json` can be imported once into the new SQLite tally and then appended from there:

```bash
docker run --name grimoire --gpus all -p 9001:9001 \
  -e GRIMOIRE_API_KEY="change-me" \
  -e GRIMOIRE_LEGACY_STATS_PATH=/tmp/token-stats.json \
  -v /home/yeowool/models:/models \
  -v /home/yeowool/templates:/templates:ro \
  -v /home/yeowool/structured-cot/grammars:/etc/grimoire/grammars:ro \
  -v /home/yeowool/token-stats.json:/tmp/token-stats.json:ro \
  -v grimoire-state:/var/lib/grimoire \
  ghcr.io/lpfchan/grimoire:latest
```

The import is idempotent per source path. New usage is appended to `/var/lib/grimoire/usage.sqlite3`.

Endpoints:

- `GET /stats` — current API key totals
- `GET /stats/global` — global totals, admin auth required

## Building

```bash
docker build -t grimoire:latest .
```
