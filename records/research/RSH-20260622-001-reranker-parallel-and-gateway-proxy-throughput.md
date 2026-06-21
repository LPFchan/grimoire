# RSH-20260622-001: Reranker `--parallel` plateau and the gateway proxy throughput cap

Opened: 2026-06-22 04-26-49 KST
Recorded by agent: claude

## Question

Does raising the reranker's `--parallel` (and scaling `ctx-size` to use GPU 1's
spare VRAM) increase rerank throughput, and where does it plateau?

## Findings

### Bumping `--parallel` does not help — it hurts

`eastself-reranker-0.6B` is a prefill-only encoder (one forward pass per
query/doc pair, no decode loop). Throughput is flat-to-declining as `--parallel`
rises; extra slots fragment batching and inflate KV VRAM and tail latency.

Direct to the llama-server (bypassing the gateway), docs=1, concurrency 32:

| parallel | req/s | note |
| --- | --- | --- |
| 1 | 68.7 | best |
| 8 | 46.8 | worse |

Through the gateway after the proxy fix (below), docs=1, concurrency 32:

| parallel | req/s | p99 |
| --- | --- | --- |
| 1 | 65.0 | 567 ms |
| 4 | 48.7 | 914 ms |
| 8 | 47.1 | 5436 ms |

`ctx-size` only changes allocated KV VRAM (≈0.109 MiB/token f16 for this model);
it has no throughput effect. Scaling ctx with parallel fills VRAM but buys
nothing.

### The embedder confirms the same pattern

`eastself-embedder-0.6B` (also a prefill-only encoder) behaves identically.
Gateway embeddings, batch=1, concurrency 32:

| parallel | req/s |
| --- | --- |
| 1 | 112.1 |
| 4 | 83.9 |
| 8 | 81.9 |

Throughput is GPU-compute-bound at ~115 texts/s and does not improve with input
**batching** either (batch=1 → 112, batch=8 → 117, batch=16 → 117 texts/s);
batching only trades request rate for per-request latency. The gateway fix also
closed the embeddings gap: gateway 112.5 vs direct 115.7 req/s (~3% overhead).

General rule: both always-on encoders are compute-bound at their ceiling
(~115 texts/s embed, ~69 pairs/s rerank); neither benefits from `--parallel`,
input batching, or larger `ctx-size`. Keep both at `parallel=1`.

### The real limiter was the gateway proxy's per-request httpx client

Every proxy path (`proxy_v1`, `/v1/responses`, `_proxy_chat`) created a fresh
`httpx.AsyncClient` per request — new TCP connection, no keepalive — costing
~33 ms/request and capping high-RPS small-request endpoints at ~30 req/s.

Micro-benchmark against the same backend (docs=1, concurrency 32):

- per-request client: **30.2 req/s**
- shared pooled client: **88.8 req/s** (~2.9x)

After switching to a shared, connection-pooled client
(`grimoire/proxy/client.py`), gateway rerank went **29.2 → 64.4 req/s**
(p50 1087 → 476 ms), matching the direct llama-server. Rerank scores,
embeddings, and a streaming chat completion (with speculative decoding) all
stayed correct. (commits `b616a99` allocator context, `4c37298` the fix.)

### Correction to RSH-20260518-006

RSH-20260518-006 concluded "gateway adds 0-60 ms per request (<1%)". That held
for **chat**, where the per-request client setup amortizes over a multi-second
generation. It did **not** hold for high-RPS small requests (rerank/embeddings),
where the per-request connection setup dominates and caps throughput at ~30/s.
Both can be true; the earlier memo only measured the chat regime.

## Rejected paths

- Raising `--parallel` for the reranker — flat-to-negative throughput.
- Raising `ctx-size` for throughput — only fills VRAM.
- Summing declared budgets to enforce VRAM — live `nvidia-smi` free VRAM is
  better ground truth (see DEC-20260622-001).

## Open / follow-ups

- Two off-hot-path per-request clients remain (`entrypoint.py` cors-proxy,
  `proxy/llama.py` pflash KV-save); pooling them is cosmetic, left as-is.
- Reranker kept at `parallel=1 / ctx=2048` (its optimum). A ctx bump for
  long-document headroom is available but unrelated to throughput.
- Benchmark tool: `tests/bench_rerank_parallel.py`.
