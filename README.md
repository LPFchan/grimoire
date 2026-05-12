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
      "max-effective-context": 100000,
      "budget": 22,
      "cache-type-k": "q8_0",
      "cache-type-v": "q8_0",
      "snapshot-mode": "compact-full",
      "snapshot-ram-dir": "/dev/shm/grimoire-snapshots",
      "snapshot-disk-dir": "/var/lib/grimoire/snapshot_swap/dflash-qwen-27B",
      "snapshot-ram-budget-gb": 20,
      "snapshot-staging-slot": 7,
      "prefix-cache-slots": 2,
      "session-kv-slots": 4,
      "prefill-threshold": 48000,
      "prefill-keep-ratio": 0.05,
      "prefill-tail-budget": 16000,
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
| `target` | Main model GGUF (~14.0 GB GPU, always resident) |
| `draft` | DFlash speculative draft (3.3 GB, parked except during verify) |
| `drafter` | Compression scorer GGUF (1.1 GB, parked except during compression) |
| `tokenizer` | Local tokenizer dir |
| `budget` | DDTree page pool (22 = 262K ctx) |
| `max-effective-context` | Hard cap for prompt tokens after PFlash compression |
| `max-raw-ceiling` | Hard cap for raw prompt tokens before compression; defaults to `ctx-size` |
| `snapshot-mode` | Live snapshot format; `compact-full` is the default and only restore path |
| `snapshot-ram-dir` | tmpfs hot path for persisted compact snapshot files |
| `snapshot-disk-dir` | async mirrored disk backup for snapshots |
| `snapshot-ram-budget-gb` | LRU budget for RAM-backed snapshot files |
| `snapshot-staging-slot` | transient daemon slot used for `LOAD_SNAPSHOT` / `RESTORE` / `SNAPSHOT` / `SAVE_SNAPSHOT` |
| `prefix-cache-slots` | max persisted prefix-cache entries |
| `session-kv-slots` | max persisted per-conversation session snapshots |
| `prefill-threshold` | Token count to trigger PFlash compression |
| `prefill-keep-ratio` | Middle keep fraction (0.05 = 5%) |
| `prefill-tail-budget` | Protected tail tokens (uncompressed) |
| `prefill-compression` | `"auto"` or `"never"` |

## DFlash Speculative Decoding

~3.4x decode speedup on RTX 3090 via DDTree + PFlash. Target model **never parked** — only draft/drafter cycle.

### VRAM Budget (24 GB RTX 3090)

**Hard ceiling: 23.5 GB** (leaves 500 MB safety margin).

```
target weights (GPU, no token_embd):   ~14.0 GB  (always resident)
draft model (bf16 safetensors):         ~3.3 GB  (loaded during verify)
rollback cache (DDTree budget=22):      ~1.9 GB  (verify intermediates)
SSM + conv states + target_feat:        ~0.35 GB (fixed)
CUDA workspace + overhead:              ~0.5 GB
─────────────────────────────────────────────────────
total fixed during generation:          ~20.1 GB
remaining for variable KV:              ~3.9 GB
```

**q8_0 KV cost:** 34,816 bytes/token  
(16 full-attention layers × 2 (K+V) × 4 heads × 256 dims × 1.0625 bytes/q8_0 element)

The ~113K figure is the steady-state headroom with only the active cache resident. The 100K `max-effective-context` limit assumes snapshots are persisted as files and no snapshot copy stays resident in a daemon slot between requests.

Compact full snapshots still scale with the used KV prefix, but Grimoire routes them through one transient staging slot and frees that slot immediately after save/load completes.

### PFlash Compression

Prompt split on `len(prompt_ids) >= prefill_threshold`:

```
[ HEAD: system + first user block             ]
[ MIDDLE: compressible blocks at 5%           ]
[ TAIL: protected recent whole blocks         ]
[ TOOLS: protected tool blocks stay exact     ]
```

Head, protected tool blocks, and recent tail blocks stay uncompressed. Compressible middle blocks are scored by the drafter (loads 1.2 GB, ~2s, parks).

When a request includes the `conversation_recall` tool, Grimoire also injects a small DFlash runtime system note. It tells the model that older middle context may be compressed on long prompts and that exact older wording should be recovered with `conversation_recall` instead of assumed to be verbatim.

**Tuned values**: `max-effective-context=100000`, `prefill-threshold=48000`, `prefill-tail-budget=16000`, `prefill-keep-ratio=0.05`, `cache-type-k=q8_0`, `cache-type-v=q8_0`.

### Session KV

SHA-1 prefix hash stored with each `(snapshot_key, prefix_len)`. On restore, the current prompt prefix is validated before loading the compact full snapshot from RAM or disk — mismatch evicts the stale entry.

### Prefix Cache vs Session Snapshots

| | **Prefix cache** | **Session KV** |
|---|---|---|
| **Purpose** | Skip prefill for shared prompt prefixes | Skip prefill on the next turn of the same conversation |
| **Key** | SHA-1 of prefix tokens | stable `conversation_id` hash |
| **Persistence** | compact full snapshot file | compact full snapshot file |
| **Hot path** | `/dev/shm/grimoire-snapshots` | `/dev/shm/grimoire-snapshots` |
| **Cold backup** | `/var/lib/grimoire/snapshot_swap/...` | `/var/lib/grimoire/snapshot_swap/...` |

### Thin Snapshots

The daemon supports two snapshot types:

- **Compact full snapshot** — copies the full hybrid-model state the gateway needs to resume correctly: the used KV prefix `[0, cur_pos)`, all 48 SSM states, all 48 conv states, and the `target_feat` ring buffer. This is the gateway's default and only live restore format.
- **Thin snapshot** — copies **only the used KV range** `[kv_start, kv_end)` and **skips SSM/conv/target_feat entirely**. This saves ~400 MB per snapshot but still costs 34 KiB per *used* token.

Thin snapshots remain available in the daemon protocol, but the gateway does not use them for live session restores because qwen35 resume correctness depends on the non-KV hybrid state too.

### RAM-Backed Compact Snapshots

With q8_0 + DDTree budget=22 + draft loaded, only **~3.9 GB** remains for variable KV. That fits **~113K active tokens** with **zero** snapshots kept in VRAM between requests. To hold 100K context and preserve resumability, snapshots are saved to tmpfs immediately and the daemon slot is freed right away.

Current behavior:

- After generation: `SNAPSHOT` -> `SAVE_SNAPSHOT /dev/shm/...` -> daemon frees the slot
- Prefix cache entries can reuse that just-written compact snapshot file instead of taking a second daemon snapshot
- In the background: RAM copy is mirrored to `/var/lib/grimoire/snapshot_swap/...`
- Before the next turn: `LOAD_SNAPSHOT` from RAM if present, otherwise from disk -> `RESTORE` -> `FREE_SNAPSHOT`

```
After generation:
  SNAPSHOT <staging_slot>               → capture compact full state at cur_pos
  SAVE_SNAPSHOT <staging_slot> <ram_path> → write /dev/shm copy and free the slot
  [async] mirror ram_path -> disk_path  → non-blocking cold-start backup

Before next turn:
  LOAD_SNAPSHOT <staging_slot> <ram_or_disk_path>
  RESTORE <staging_slot> ...
  FREE_SNAPSHOT <staging_slot>
```

RAM restore is the fast path; disk restore is the cold-start fallback.

### Snapshot Store

Snapshots are stored as `.dfsn` files keyed by stable content/conversation hashes. The RAM store is LRU-evicted under the configured tmpfs budget; evicted entries remain recoverable from disk.

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
