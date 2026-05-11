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
      "ctx-size": 262144,
      "cache-type-k": "turbo4",
      "cache-type-v": "turbo4"
    },
    "dflash-qwen-27B": {
      "backend": "dflash",
      "target": "gguf/Qwen3.6-27B-Q4_K_M.gguf",
      "draft": "dflash/Qwen3.6-27B-DFlash/model.safetensors",
      "drafter": "gguf/Qwen3-0.6B-BF16.gguf",
      "tokenizer": "tokenizers/qwen3.6-27B",
      "ctx-size": 262144,
      "budget": 22,
      "cache-type-k": "tq3_0",
      "cache-type-v": "tq3_0",
      "prefix-cache-slots": 2,
      "session-kv-slots": 2,
      "prefill-threshold": 95000,
      "prefill-keep-ratio": 0.10,
      "prefill-tail-budget": 76500,
      "prefill-compression": "auto"
    }
  },
  "fixed": {}
}
```

- `models` — definitions, no GPU assignment
- `fixed` — alias → GPU ID (pinned, never evicted)
- Dynamic allocation: free GPU preferred, oldest non-pinned evicted when all busy
- `backend: "llama"` (default, HTTP) vs `backend: "dflash"` (stdin/stdout)

### DFlash Settings

| Key | Description |
| --- | --- |
| `target` | Main model GGUF (16 GB, always resident) |
| `draft` | DFlash speculative draft (3.5 GB, parked except during verify) |
| `drafter` | Compression scorer GGUF (1.2 GB, parked except during compression) |
| `tokenizer` | Local tokenizer dir |
| `budget` | DDTree page pool (22 = 262K ctx) |
| `prefix-cache-slots` | VRAM slots for prefix cache snapshots |
| `session-kv-slots` | VRAM slots for per-session KV snapshots |
| `prefill-threshold` | Token count to trigger PFlash compression |
| `prefill-keep-ratio` | Middle keep fraction (0.10 = 10%) |
| `prefill-tail-budget` | Protected tail tokens (uncompressed) |
| `prefill-compression` | `"auto"` or `"never"` |

## DFlash Speculative Decoding

~3.4x decode speedup on RTX 3090 via DDTree + PFlash. Target model **never parked** — only draft/drafter cycle.

### VRAM Budget (24 GB RTX 3090)

```
target weights:                  16.0 GB  (resident)
active KV (tq3_0, 25KB/token):   variable
session KV snapshots (×2):        variable  (mirrors active)
prefix cache (15K head):          0.375 GB  (fixed)
───────────────────────────────────────────────
remaining budget:                 8.0 GB
```

```
vram = 16.0 + effective_tokens × 25000 × 3 / 1e9 + 0.375  ≤ 23.5 GB
```

`effective_tokens = head(15K) + middle_raw × keep_ratio + tail_budget`

### PFlash Compression

Prompt split on `len(prompt_ids) >= prefill_threshold`:

```
[ HEAD: 15K fixed ── sysprompt + compaction ]
[ MIDDLE: compressed at 10% via drafter      ]
[ TAIL:   protected, uncompressed             ]
```

Head/tail always uncompressed. Middle scored by drafter (loads 1.2 GB, ~2s, parks).

**Tuned values**: threshold=95000, tail=76500 (Opus 4.7). Leaves ~10K gap → 100+ turns between compressions. GPT-5.5 recommends tail=61428 for 200K+ stress safety.

### Session KV

SHA-1 prefix hash stored with each (slot, prefix_len). On restore, current prompt prefix is validated — mismatch evicts stale snapshot (e.g. after in-place message edit).

### SSD Snapshot Swap

LRU snapshots evicted to NVMe as `.dfsn` (key-derived filenames, manifest-tracked). Loaded on demand. SSD: 1.9 GB/s.

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

## Chat UI

Stock llama.cpp SvelteKit webui at `https://chat.lost.plus/`. Router-mode API: `GET /props`, `GET /v1/models`, `POST /models/load`, `POST /models/unload`.

`GRIMOIRE_WEBUI_DIR` overrides asset path.

### Server-Side History

`patches/grimoire-webui-history.patch` swaps IndexedDB for HTTP to `/history` (tree-aware: branches, forks, cascade delete). Per `user_hash` (SHA-256 of API key).

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