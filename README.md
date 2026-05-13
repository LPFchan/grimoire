# Grimoire

Self-hosted AI inference infrastructure for multi-GPU llama.cpp + DFlash serving.

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
                         │  llama     │ │ dflash│ │ llama       │
                         │  model A   │ │ model │ │ model Z     │
                         └────────────┘ └───────┘ └─────────────┘
```

## Features

- **Multi-GPU** — Run multiple models simultaneously, one per GPU
- **Dual backends** — llama.cpp (HTTP) + DFlash (stdin/stdout) managed by the same gateway
- **DFlash speculative decoding** — DDTree + PFlash for ~3.4x faster decode on RTX 3090
- **Prefix cache** — LRU KV snapshot cache with disk persistence for repeated prompts
- **Dynamic GPU allocation** — Free GPU preferred, oldest non-pinned model evicted when all GPUs busy
- **Canonical model switcher** — Built-in web UI and API for loading/switching models
- **Server-side history** — Per-api-key conversation tree (branches, currNode pointer, fork chains) stored in SQLite, served to the bundled webui via the same `/history` API contract its DatabaseService used to hit IndexedDB
- **Token/cost tally** — Per-api-key and global token/cost accounting with legacy import
- **Pinned models** — Fix specific models to specific GPUs via `fixed` section
- **Model registry** — JSON-based registry with per-model settings
- **Safe model ingestion** — Download and register HTTPS models via CLI or authenticated API
- **Protected API** — `/v1/*`, history, stats, and management endpoints require API/admin auth
- **OpenAI-compatible API** — Standard `/v1/chat/completions` with automatic routing
- **Built-in chat UI** — Stock llama.cpp SvelteKit webui served at `/`, talking to grimoire as a router-mode backend (no DOM injection, no fork patches)

## Usage

```bash
# Start with a specific model
docker run --name grimoire --gpus all -p 9001:9001 \
  -e GRIMOIRE_API_KEY="change-me" \
  -v /path/to/models:/models \
  -v /home/yeowool/templates:/templates:ro \
  -v /home/yeowool/structured-cot/grammars:/etc/grimoire/grammars:ro \
  -v grimoire-state:/var/lib/grimoire \
  grimoire:local --model qwen-3.6-27B

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

API endpoints require `Authorization: Bearer ...` or `X-Grimoire-Token`; set `GRIMOIRE_API_KEY` or the legacy-compatible `GATEWAY_API_KEY` before exposing the gateway.
Unauthenticated local-only development requires the explicit opt-in `GRIMOIRE_ALLOW_ANONYMOUS=1`.
Management endpoints use `GRIMOIRE_ADMIN_TOKEN` if set, otherwise the API key.

## Model Registry

The mutable registry is stored at `/var/lib/grimoire/models.json` by default so it is persisted by the state volume. The image ships a seed registry at `/etc/grimoire/models.json`; if the state registry does not exist yet, Grimoire reads the seed and writes future changes to `/var/lib/grimoire/models.json`.

`/var/lib/grimoire/models.json`:

```json
{
  "models": {
    "qwen-3.6-27B": {
      "backend": "llama",
      "file": "gguf/Qwen3.6-27B-UD-Q4_K_XL.gguf",
      "mmproj": "gguf/Qwen3.6-27B-mmproj-BF16.gguf",
      "ctx-size": 262144,
      "cache-type-k": "turbo4",
      "cache-type-v": "turbo4"
    },
    "dflash-qwen-27B": {
      "backend": "dflash",
      "target": "gguf/Qwen3.6-27B-Q4_K_M.gguf",
      "draft": "dflash/Qwen3.6-27B-DFlash/model.safetensors",
      "drafter": "gguf/Qwen3-0.6B-BF16.gguf",
      "ctx-size": 262144,
      "budget": 22,
      "prefix-cache-slots": 4
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
- **Backends**: `backend: "llama"` (default, HTTP) or `backend: "dflash"` (stdin/stdout protocol)

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
  grimoire:local
```

The import is idempotent per source path. New usage is appended to `/var/lib/grimoire/usage.sqlite3`.

Endpoints:

- `GET /stats` — current API key totals
- `GET /stats/global` — global totals, admin auth required

## Chat UI

The image bundles the stock llama.cpp SvelteKit webui (built from
`tools/server/webui` in the same llama.cpp ref the runtime uses) and serves it
at `https://chat.lost.plus/`. Grimoire implements the same router-mode API
contract the webui already speaks: `GET /props`, `GET /props?model=<id>`,
`GET /v1/models` (with `status.value`), `POST /models/load`, `POST /models/unload`.

On first load the webui prompts for the API key, which it sends as
`Authorization: Bearer ...` on every authenticated request. The key is the
same `GRIMOIRE_API_KEY` / legacy `GATEWAY_API_KEY` used by OpenCode. After a
successful POST to `/login`, grimoire writes the key into the webui's
`LlamaCppWebui.config` localStorage entry so users only authenticate once.

To override where the webui assets are served from (e.g. for development),
set `GRIMOIRE_WEBUI_DIR` to a directory containing `index.html`.

### Server-side conversation history

The webui's upstream conversation store is Dexie/IndexedDB (browser-local).
This repo ships `patches/grimoire-webui-history.patch`, applied during the
webui build stage, that swaps the `DatabaseService` Dexie backend for HTTP
calls to grimoire's tree-aware `/history` endpoints. The patch is scoped to
one file (`tools/server/webui/src/lib/services/database.service.ts`) and
preserves the existing static method signatures so chat/conversations stores
keep working without changes.

Endpoints the patched DatabaseService hits:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/history` | List conversations (sidebar) |
| POST | `/history` | Create or upsert conversation |
| GET | `/history/{id}` | Read one conversation with its message tree |
| PATCH | `/history/{id}` | Partial conversation update (`updateConversation`) |
| DELETE | `/history/{id}?with_forks=true` | Delete conversation; optionally cascade through forks |
| POST | `/history/{id}/messages` | Create a message branch under `parent_id` |
| PATCH | `/history/messages/{id}` | Update a message (resolves convId server-side) |
| DELETE | `/history/messages/{id}` | Delete a single message |
| DELETE | `/history/{id}/messages/{id}?cascade=true` | Cascade delete a subtree |
| POST | `/history/{id}/fork` | Fork conversation at a given message |
| POST | `/history/import` | Bulk-import conversations in the webui's exported shape |

Conversations and message trees are stored per `user_hash` (sha256 of the
API key), so two users with different keys see disjoint history.

## Building

```bash
# Clone with submodules (required for DFlash build)
git clone --recursive <grimoire-repo> ~/grimoire

# Build (includes llama.cpp + DFlash compilation, ~90 min first build)
cd ~/grimoire
docker compose build

# Update DFlash submodule later
git submodule update --remote dflash
docker compose build
```

## Running As A Service

A systemd unit ships at `etc/grimoire.service`. To install:

```bash
sudo install -d /etc/grimoire
sudo install -m 600 /dev/stdin /etc/grimoire/grimoire.env <<'EOF'
GRIMOIRE_API_KEY=change-me
GRIMOIRE_ADMIN_TOKEN=change-me
GRIMOIRE_LEGACY_STATS_PATH=/var/lib/grimoire/token-stats.json
EOF
sudo install -m 644 etc/grimoire.service /etc/systemd/system/grimoire.service
sudo systemctl daemon-reload
sudo systemctl enable --now grimoire.service
journalctl -u grimoire.service -f
```

The unit uses `--log-driver=journald --log-opt tag=grimoire`, so all gateway and
child llama-server output goes through `journalctl -t grimoire` for the same
post-mortem grep parity as the legacy `eastself-*` units.

The unit also runs `docker run --rm`, so each restart starts from a clean
container. State (model registry, history, usage) is held in the
`grimoire-state` named volume.
