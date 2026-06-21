# DEC-20260622-001: GPU co-location via per-model VRAM budget

Opened: 2026-06-22 04-26-49 KST
Recorded by agent: claude

## Metadata

- Area: `model_manager.py` allocator, `registry.py`/`routes/models.py` validation, `etc/models.json`
- Commit: `b616a99`
- Related: RSH-20260519-001 (3090 allocator ceiling 24,096 MiB)

## Decision

Replace the strict one-model-per-GPU allocator rule with a per-GPU VRAM-budget
check, gated by a new optional model field `vram-budget-mib`:

- **Budgeted** model (has `vram-budget-mib`): may co-locate on a GPU when
  `nvidia-smi` free VRAM ≥ its budget, without evicting pinned incumbents.
- **Unbudgeted** model: stays exclusive, never queries `nvidia-smi`, but treats
  budgeted incumbents as non-blocking co-tenants — it prefers an empty GPU, then
  swaps an exclusive incumbent (whole-GPU), and only co-locates over budgeted
  tenants as a last resort.
- A malformed `vram-budget-mib` (bool/string/≤0) is rejected by
  `registry.validate()` so it cannot silently revert a model to exclusive.

The two always-on 0.6B models (`eastself-embedder-0.6B`,
`eastself-reranker-0.6B`) are pinned to GPU 1 with `vram-budget-mib: 1800`.

## Context

Under one-model-per-GPU, the two always-on 0.6B models each occupied an entire
24 GiB 3090 (~1.4 GiB resident each), leaving no room for chat models and
forcing them to evict the small models. They needed to share one GPU and leave
the other free for chat LLMs.

## Options considered

1. **Sum declared budgets per GPU** — rejected; relies on operator estimates and
   can drift from reality.
2. **Live `nvidia-smi` free-VRAM check** (chosen) — ground truth that already
   reflects every resident process, including still-loading models.
3. **Keep one-model-per-GPU, just spread** — rejected; does not free a GPU and
   keeps the eviction churn.

## Rationale

Free VRAM from `nvidia-smi` is strictly better than summing estimates. `start_model`
holds `self._lock`, serializing allocation; a post-eviction re-check (with a 0.5 s
settle) guards the eviction path. Unbudgeted models declare no footprint, so the
conservative tiering (empty → swap exclusive → co-locate last) avoids cramming a
large model onto a small model's GPU and OOMing.

## Consequences

- Both 0.6B models co-locate on GPU 1 (~2.9 GiB total); GPU 0 is free for chat.
- Verified live: a 27B chat model loads on GPU 0 without disturbing the pinned pair.
- A model **explicitly pinned** to a GPU whose spare VRAM is too small will OOM at
  launch (the unbudgeted path has no VRAM gate by design); give such a model a
  `vram-budget-mib` to route it through the budgeted path.
- Co-location relies on the gateway being able to feed both models efficiently;
  see RSH-20260622-001 for the proxy throughput fix done alongside this.
