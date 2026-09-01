# DEC-20260902-001: Deprecate DFlash and PFlash

Opened: 2026-09-02 04-20-00 KST
Recorded by agent: claude-code

## Metadata

- Status: accepted
- Deciders: operator
- Related ids: DEC-20260516-005, DEC-20260517-004, DEC-20260518-001, DEC-20260528-001

## Decision

Remove DFlash and PFlash from the repository. This supersedes
DEC-20260516-005 (port the Bee DFlash pipeline), DEC-20260517-004 (retire the
TheTom DFlash port) and DEC-20260518-001 (dedicated PFlash compressor thread).
Those records stay as written; this one replaces their outcome.

Removal happens in stages, because the two names cover three different things:
code that is dead, code that is live but misnamed, and a vendored toolchain that
other work depends on.

## Context

The MTP migration on 2026-05-21 replaced every served model with an MTP or
NextN variant and, in its own words, retired DFlash and PFlash. The features
were never removed from the code.

Verified on 2026-09-02 against both the tracked seed and the live persisted
registry the gateway actually loads, which are byte identical: 60 models, 8
`mtp`, 6 `nextn`, zero `dflash`, zero `pflash`, and no model setting the
`pflash` boolean. The `_start_pflash_daemon()` path still exists and is
reachable, but is never taken.

What the repository still pays for it:

- a `pflash-build` CUDA stage in the Dockerfile compiling 23 sources
- 254 MB of `/opt/pflash` in the runtime image, including a `pflash_daemon`
  binary that cannot be launched
- Block-Sparse-Attention vendored solely to build that daemon
- `PflashAwarenessPlugin`, which fires only on `speculative-type: pflash`
- five test harnesses pointed at models deleted in May, plus a test asserting
  that they keep pointing there

Three things carry the names but are load-bearing and must be preserved:

- `KVCacheStore`, the prompt cache used on every chat request. `_kv_store()` is
  called outside the PFlash branch.
- `DFLASH_PROTECTED_TOOLS`, which drives protected-block marking during live
  prompt rendering.
- the vendored `llama.cpp` under `src/grimoire/pflash/deps/`, which supplies
  `convert_lora_to_gguf.py` and `gguf-py` to PEFT intake and the tokenizer
  writer. It is 151 MB of the 152 MB footprint.

## Options Considered

### Leave both in place

- Upside: no risk of breaking a serving path
- Downside: every image build compiles a dead CUDA daemon and ships 254 MB of it
- Downside: two retired features keep appearing in the registry schema, the
  model editor UI and the validation code, inviting configurations that cannot work

### Delete everything named dflash or pflash

- Upside: simplest to describe
- Downside: breaks chat, by deleting the prompt cache
- Downside: breaks PEFT intake, by deleting the vendored converter

### Stage the removal, separating dead code from misnamed live code

- Upside: each stage is independently verifiable and revertible
- Upside: the largest win, deleting the CUDA daemon, carries almost no runtime risk
- Downside: more commits, and the tree names both features until the last stage

## Rationale

The staged option is the only one that is both complete and safe. The order runs
from no runtime effect to most: records and dead tests, then the unreachable
CUDA daemon and its build stage, then salvaging the live code out of the
misnamed packages, then relocating the vendored toolchain, then the surfaces
outside `src/`.

An earlier draft of this plan claimed the compression call sites also produced
the block manifest feeding prompt-cache boundaries, which would have made the
salvage delicate. Review found that wrong, and it was confirmed against the
tree: `materialize_blocks` and `_prefix_cache_boundaries` are imported but never
called, and `maybe_compress` is reached only inside `if pcfg and pcfg.enabled:`.
The compression branch lifts out as a unit.

## Consequences

- Image loses roughly 254 MB and the build loses a CUDA compile stage.
- `src/grimoire/pflash/deps/llama.cpp` moves to `vendor/llama.cpp`, where its
  role as intake tooling is visible rather than buried under a dead feature.
- `KVCacheStore` and the prefill helpers move to `src/grimoire/cache/`.
- The `webui` submodule stops offering DFlash and PFlash in the model editor.
  That change and the backend change must be deployed together, or the UI
  offers configurations the backend rejects.
- Model configs may no longer set `speculative-type: dflash` or `pflash`, or the
  `pflash` boolean. Nothing in the registry does today.
- `records/` keeps every DFlash and PFlash record. They are history.
