# DEC-20260622-002: Multi-process gateway with data-parallel GPU replicas

Opened: 2026-06-22 16-36-35 KST
Recorded by agent: claude

## Metadata

- Area: gateway process topology, `proxy_app.py`, `proxy/routes_table.py`, `model_manager.py`, `entrypoint.py`, `etc/models.json`
- Commit: `8031d6e`
- Related: RSH-20260622-001 (throughput findings), DEC-20260622-001 (co-location allocator)

## Decision

Split the single gateway process into a **control plane** and a **data plane**:

- **Manager (1 process)** owns `ModelManager` (lifecycle, GPU allocation/co-location,
  always-on), admin routes, and chat (`_proxy_chat` — stateful: KV slots, pflash,
  conversation tracking). Binds an internal port (9000). Publishes a route table
  (`/dev/shm/grimoire/routes.json`) on every model change.
- **Proxy workers (N, `uvicorn --workers N`)** are stateless, own the public port
  (9001), round-robin the stateless encoder endpoints (`/v1/embeddings`,
  `/v1/rerank`) across a model's GPU replicas, and forward chat/admin to the
  manager (auth preserved).

Replicas are first-class via `replica_peers` (a model lists sibling data-parallel
copies on other GPUs). The embedder and reranker each get a GPU-0 replica
(always-on, pinned GPU 0), so one model name fans out across both GPUs.

`entrypoint.main()` launches both: manager in-process + proxy workers as a
subprocess. No image rebuild — the container ENTRYPOINT is unchanged.

## Context

A single Python gateway process caps high-RPS throughput at ~150 req/s (one event
loop forwarding each request twice). The GPUs scale 2x with data-parallel replicas
(226 req/s direct), but funneling through one process wasted it (~141 req/s). The
operator runs large embed+rerank jobs that need the GPUs' full aggregate.

## Options considered

1. **Bypass the gateway for the job** — fastest, but loses auth/observability and
   is a one-off.
2. **Multi-process gateway** (chosen) — N stateless proxy workers + 1 stateful
   manager. Transparent 2x under one model name; chat stays correct.
3. **uvicorn --workers N on the whole app** — rejected; `ModelManager` owns
   subprocesses/state and cannot be forked per worker.

## Rationale

Embeddings/rerank are stateless and parallelize across worker processes cleanly;
a 4-worker PoC hit 223.7 req/s. Chat is stateful (per-`ActiveModel` KV slots,
pflash) and stays on the single manager, where per-request overhead amortizes over
long generations anyway. A tmpfs route table is the minimal shared state.

## Consequences

- Production :9001 embeddings: 224 req/s under one model name (was 115 single-GPU /
  150 single-process), both GPUs ~86%.
- Always-on GPU-0 replicas cost ~2.9 GiB/GPU; both GPUs drop 24 -> ~21 GiB free, so
  large-ctx chat models are capped accordingly (operator-accepted; alternative was
  on-demand replicas).
- New failure surface: if the manager dies, models die (as before) and the proxy
  health check fails -> container restart. Stale route table -> a request to a
  just-evicted model can 502 (re-resolve/retry).
- Deployment is now two processes in one container (manager + proxy workers),
  launched by `entrypoint.main()`.
