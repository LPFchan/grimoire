# Grimoire

Multi-GPU inference gateway: llama.cpp + DFlash speculative decoding behind an OpenAI-compatible `/v1` API.

```
client ──/v1──► chat.lost.plus (CF Tunnel) ──► grimoire :9001 ──┬── GPU 0: llama model A
                                                                 ├── GPU 1: dflash model
                                                                 └── GPU N: llama model Z
```

## Quick Start

```bash
docker run --name grimoire --gpus all -p 9001:9001 \
  -e GRIMOIRE_API_KEY="change-me" \
  -v /path/to/models:/models \
  -v grimoire-state:/var/lib/grimoire \
  grimoire:local --model qwen-3.6-27B
```

CLI management:

```bash
docker exec grimoire grimoire list                    # registered models
docker exec grimoire grimoire ingest --alias X --url Y  # download + register
docker exec grimoire grimoire pin gemma-4-31B 1        # pin to GPU
docker exec grimoire grimoire unpin gemma-4-31B        # release
curl -X POST http://localhost:9001/switch/qwen-3.6-27B -H "Authorization: Bearer $KEY"
```

## Auth

| Header | Env var | Scope |
| --- | --- | --- |
| `Authorization: Bearer ...` or `X-Grimoire-Token` | `GRIMOIRE_API_KEY` (or legacy `GATEWAY_API_KEY`) | `/v1/*`, history, stats |
| Admin auth | `GRIMOIRE_ADMIN_TOKEN` (falls back to API key) | Management endpoints |
| `GRIMOIRE_ALLOW_ANONYMOUS=1` | — | Local dev, no auth |

## Model Registry

Seed at `/etc/grimoire/models.json`, persisted to `/var/lib/grimoire/models.json`:

```json
{
  "models": {
    "qwen-3.6-27B": {
      "file": "gguf/Qwen3.6-27B-Q4_K_M.gguf",
      "gpu-ids": [0, 1],
      "ctx-size": 262144,
      "cache-type-k": "turbo4",
      "cache-type-v": "turbo4",
      "extra-args": ["--tensor-split", "1,1"]
    },
    "dflash-qwen3.6-27B": {
      "file": "gguf/Qwen3.6-27B-Q4_K_M.gguf",
      "draft": "gguf/dflash-draft-3.6-q8_0.gguf",
      "speculative-type": "dflash",
      "spec-dflash-cross-ctx": 1024,
      "ctx-size": 184832,
      "cache-type-k": "q8_0",
      "cache-type-v": "q8_0",
      "fa-window": 2048,
      "budget": 18,
      "kv-cache-disk-dir": "/var/lib/grimoire/kv_cache/dflash-qwen3.6-27B",
      "kv-cache-ram-budget-mb": 512,
      "kv-cache-disk-budget-mb": 2048,
      "kv-cache-ttl-hours": 168
    }
  },
  "fixed": {
    "qwen-3.6-27B": 0
  }
}
```

- `models` — model definitions; `gpu-ids` optionally assigns one ordinary llama-server process to an ordered list of physical GPUs
- The first `gpu-ids` member is the primary physical GPU reported by the backward-compatible `gpu` field. llama.cpp defaults to layer splitting; `extra-args` may set `--tensor-split` proportions in the same visible-device order.
- `gpu-ids` is initially incompatible with `cpu-only`, `vram-budget-mib`, PFlash, park/unpark, and native DFlash models
- `fixed` — alias → GPU ID (pinned, never evicted); for a model with `gpu-ids`, the fixed ID must equal the first member
- `predict` — maximum generated tokens passed to llama-server; `-1` means no separate output cap, so generation is limited by the model context and stop conditions

Temporary runtime controls are available through admin-authenticated POST requests. They live only in manager memory, never change the registry or preset state, and clear on restart:

```bash
curl -X POST "$GRIMOIRE_ORIGIN/models/qwen/clone" \
  -H "Authorization: Bearer $GRIMOIRE_API_KEY" -H "Content-Type: application/json" \
  -d '{"gpu_ids":[0,1],"tensor_split":[1,1]}'
curl -X POST "$GRIMOIRE_ORIGIN/models/qwen/declone" -H "Authorization: Bearer $GRIMOIRE_API_KEY"
curl -X POST "$GRIMOIRE_ORIGIN/models/qwen/pin" \
  -H "Authorization: Bearer $GRIMOIRE_API_KEY" -H "Content-Type: application/json" -d '{"gpu":0}'
curl -X POST "$GRIMOIRE_ORIGIN/models/qwen/unpin" -H "Authorization: Bearer $GRIMOIRE_API_KEY"
```

`clone` runs one llama-server process sharded across the ordered GPUs; it does not create a replica. Clone/declone reload active models with rollback on failure. Pin reloads only when residency must move; unpin changes eviction protection without moving a running model. `/status` keeps `gpu`/`gpus` for actual residency and reports requested placement, placement/pin sources, and runtime overrides separately. Locked presets clear runtime overrides and reconcile target models; manual-control presets retain them but enforce their GPU mask.
- Dynamic allocation: free GPU preferred, oldest non-pinned evicted when all busy
- All models use `backend: "llama"` (Bee `llama-server` HTTP) — the legacy `backend: "dflash"` daemon path is retired

### Prompt Cache Reuse

Grimoire applies `reasoning_effort` and Muse Glimmer's `reasoning_strength` from model `--chat-template-kwargs` to each request. Model aliases that produce the same llama-server command share one running process, so changing reasoning level keeps the existing KV cache. Other template defaults are merged into one startup argument.

The web UI pre-encodes the final conversation branch after each response, including reasoning and tool messages. This keeps the next user turn aligned with the serialized prompt already in the server cache. The production Compose profile reserves 4 GiB per llama-server process for its RAM prompt cache through `LLAMA_ARG_CACHE_RAM`.

### KV Cache Store (Content-Hash)

Content-hash KV caching replaces the legacy snapshot daemon. On every decode, the gateway saves a content-addressed KV persist to RAM (tmpfs) with async disk mirroring. On the next request with the same system prompt prefix, the cached KV is restored directly — no daemon round-trip.

| Key | Description |
| --- | --- |
| `kv-cache-disk-dir` | Persistent KV cache directory |
| `kv-cache-ram-budget-mb` | Max tmpfs usage before LRU eviction |
| `kv-cache-disk-budget-mb` | Max disk usage before LRU eviction |
| `kv-cache-ttl-hours` | Entry TTL in hours (default 168 = 7 days) |

Content-hash deduplication means two conversations with the same system prompt share a single cached KV entry, skipping re-prefill entirely.

### DFlash Speculative Decoding

DFlash uses Bee's native `--spec-type dflash` with a GGUF draft model (`dflash-draft-3.6-q8_0.gguf`). The canary serves text-only chat with `ctx-size=60000`, `cache-type-k=q8_0`, `cache-type-v=q8_0`, and `fa-window=2048`.

### PFlash Compression

Prompt split on `len(prompt_ids) >= prefill_threshold`:

```
[ HEAD: system + first user block             ]
[ MIDDLE: compressible blocks at 5%           ]
[ TAIL: protected recent whole blocks         ]
[ TOOLS: protected tool blocks stay exact     ]
```

Head, protected tool blocks, and recent tail blocks stay uncompressed. Compressible middle blocks are scored by the standalone `pflash_daemon` (extracted to `src/grimoire/pflash/`).

**Tuned values**: `max-effective-context=60000`, `prefill-threshold=48000`, `prefill-tail-budget=16000`, `prefill-keep-ratio=0.05`, `cache-type-k=q8_0`, `cache-type-v=q8_0`, `fa-window=2048`, `budget=18`.

## Building

```bash
git clone --recursive <repo> ~/grimoire
cd ~/grimoire
docker compose build        # ~90 min first build (llama.cpp + DFlash)

# update dflash submodule:
git submodule update --remote dflash
docker compose build
```

## Systemd

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
journalctl -t grimoire -f
```

`docker run --rm` per restart. State in `grimoire-state` volume. Logs via `--log-driver=journald`.

## Ingest

HTTPS-only, rejects private hosts, atomic writes, size-limited.

| Env | Default | Effect |
| --- | --- | --- |
| `GRIMOIRE_INGEST_MAX_BYTES` | 80 GiB | Max download size |
| `GRIMOIRE_ALLOW_HTTP_INGEST=1` | off | Allow plain HTTP |
| `GRIMOIRE_ALLOW_PRIVATE_INGEST=1` | off | Allow private/loopback |

## Stats

`GET /stats` — per-key totals. `GET /stats/global` — admin auth required.

Legacy import (`GRIMOIRE_LEGACY_STATS_PATH=/path/to/token-stats.json`) is idempotent. Appended to `/var/lib/grimoire/usage.sqlite3`.

## Dashboard

Its own site at `https://dash.lost.plus/`, built from `dash/` (see `dash/README.md`).
Static files; it reads `GET /stats/dashboard` and `PUT /stats/card-order` from the
browser and holds no state. Runs as the `dash` compose service on host port 9002.

Because it is on a different host from the API, `GRIMOIRE_CORS_ORIGINS` must name
its origin or the browser will not release the response. Set it to
`https://dash.lost.plus`; never `*` — `/stats` serves private usage and cost data.

### Telemetry storage

`state/telemetry.sqlite3` keeps raw samples every 5s plus a per-minute rollup
(`system_rollup`, storing sum and count). Windows an hour or wider read the
rollup; shorter windows read raw. The dashboard only draws 60 points, so this
costs no visible resolution and takes the lifetime window from ~15s of scanning
to under a second. Payloads are built on a worker thread and cached for a
fraction of one bin, so `/stats` can never stall the chat event loop.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GRIMOIRE_TELEMETRY_INTERVAL_S` | `5` | Sampling interval |
| `GRIMOIRE_TELEMETRY_RETENTION_DAYS` | `0` (keep all); set to `30` in the fleet | Prunes **raw** samples only; rollups are never pruned, so lifetime graphs survive. Pruning is clamped to the rollup watermark, so it can never drop an unfolded sample |
| `GRIMOIRE_CORS_ORIGINS` | empty | Comma-separated browser origins allowed to read `/stats` cross-site |

First start after adding the rollup backfills it in chunks (~21s for a 1.3 GB,
18M-row database) while the sampler keeps running.

## Chat UI

Forked llama.cpp SvelteKit webui at `https://chat.lost.plus/` (submodule at `webui/`). Router-mode API: `GET /props`, `GET /v1/models`, `POST /models/load`, `POST /models/unload`.

The dashboard used to live here as `src/routes/dashboard/`. It moved to its own
site (see above) to keep this fork closer to upstream — DEC-20260830-001.

`GET /v1/models` includes registry capabilities plus `input_modalities` and
the configured native reasoning advertisement. Multimodal aliases advertise
`["text", "image"]`; aliases without a configured reasoning kwarg advertise
reasoning as unsupported. Native alias levels are kept verbatim.

`GRIMOIRE_WEBUI_DIR` overrides asset path.

### Server-Side History

The `webui/src/lib/services/database.service.ts` replaces IndexedDB with HTTP to `/history` (tree-aware: branches, forks, cascade delete). Per `user_hash` (SHA-256 of API key).

| Method | Path |
| --- | --- |
| `GET` | `/history` — list conversations |
| `POST` | `/history` — create/upsert |
| `GET` | `/history/{id}` — conversation + tree |
| `PATCH` | `/history/{id}` — update |
| `DELETE` | `/history/{id}?with_forks=true` — delete + cascade forks |
| `POST` | `/history/{id}/messages` — create branch |
| `PATCH` | `/history/messages/{id}` — update message |
| `DELETE` | `/history/messages/{id}` — delete message |
| `DELETE` | `/history/{id}/messages/{id}?cascade=true` — cascade delete |
| `POST` | `/history/{id}/fork` — fork at message |
| `POST` | `/history/import` — bulk import |

## Drop-In Cutover

Replaces legacy `eastself-gateway` on port 9001. Same Cloudflare tunnel target, same API key, same `/v1` path. Stop legacy services before enabling grimoire (port/GPU conflict).

Mounts: `/home/yeowool/models`, `/home/yeowool/templates`, `/home/yeowool/structured-cot/grammars`.
