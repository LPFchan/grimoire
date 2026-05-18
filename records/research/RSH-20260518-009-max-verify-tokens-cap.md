# LLAMA_DFLASH_MAX_VERIFY_TOKENS Cap Blocks DDTree Tree-Mode

Opened: 2026-05-18 19-00-00 KST
Recorded by agent: mimo-v2.5-pro-precision

## Question

Why does DDTree tree-mode (`--spec-branch-budget > 0`) on Bee produce ~3-5% draft acceptance vs ~30% for flat mode? Is it a fundamental algorithm problem or a configuration bottleneck?

## Method

Source-audited Bee's ggml CUDA kernels, tree construction algorithm, and speculative decode dispatch in `Anbeeld/beellama.cpp` HEAD `4db14be`. Compared against Lucebox's published DDTree implementation (`Luce-Org/lucebox-hub`, branch `luce-dflash` commit `b16de6590`) which achieves 129.5 tok/s on the same RTX 3090 + Qwen3.6-27B Q4_K_M hardware.

## Key Finding

**Bee already has all the critical CUDA kernel infrastructure for tree-mode DDTree, and the tree construction algorithm is already present with `chain_seed` (backbone pre-seed) and best-first heap expansion at `common/speculative.cpp:1975-2150`. The bottleneck is a stale compile-time enum:**

```
include/llama.h:1135
    LLAMA_DFLASH_MAX_VERIFY_TOKENS = 25
```

### What the cap controls

The enum sits in the decode dispatch at `src/llama-context.cpp:4722-4732`:

```cpp
dflash_graph_tape_ready =
    ... && ubatch.n_seq_tokens <= LLAMA_DFLASH_MAX_VERIFY_TOKENS;

dflash_graph_hidden_ready =
    ... && !tree_bufs.active && ubatch.n_seq_tokens <= LLAMA_DFLASH_MAX_VERIFY_TOKENS;
```

When the verify batch exceeds 25 tokens (which is virtually any tree with `--spec-branch-budget > 0` beyond the minimum), both `dflash_graph_tape_ready` and `dflash_graph_hidden_ready` become `false`. This forces the decode to fall back to the slow CPU eval-callback path, silently destroying acceptance.

### Why it was never caught

Bee inherited the 25-token cap from `spiritbuun/buun-llama-cpp`, where DFlash was flat/chain-mode only. With `--spec-draft-n-max 16` (the default), 25 tokens provides comfortable headroom (16 draft + 1 bonus + safety margin). Tree-mode was added later (DDTree with `--spec-branch-budget`) but the cap was never revised. Bee's own docs at `docs/quickstart-qwen36-dflash.md` describe tree-mode as "very slow and not included in any recommended configs" — meaning no one ever pushed a large enough tree to hit the cap and notice the silent fallback.

### What already works (does NOT need porting)

| Component | File | Status |
|-----------|------|--------|
| `ggml_gated_delta_net_tree` with `persist_inter` (half) | `ggml/src/ggml-cuda/gated_delta_net.cu:409` | ✅ Present |
| `ggml_ssm_conv_tree` | `ggml/src/ggml-cuda/ssm-conv.cu:156` | ✅ Present |
| Both registered in CUDA backend | `ggml/src/ggml-cuda/ggml-cuda.cu:2951` | ✅ Present |
| Both registered in ggml.c | `ggml/src/ggml.c:1097` | ✅ Present |
| `build_ddtree` with chain_seed backbone | `common/speculative.cpp:1975-2150` | ✅ Present |
| `llama_set_tree_parent_ids()` API | `src/llama-context.cpp:3456` | ✅ Present |
| `tree_bufs.ssm_intermediates` (f16) | `src/llama-context.cpp:3570` | ✅ Present |
| `GGML_DFLASH_MAX_CTX` env-var pattern | `src/models/dflash_draft.cpp:8` | ✅ Precedent exists |

### What's needed

One env-var reader following the existing `GGML_DFLASH_MAX_CTX` pattern (its identical twin), replacing 4 hardcoded enum references:

| Location | Change |
|----------|--------|
| `include/llama.h:1135` | Rename `MAX_VERIFY_TOKENS` → `DEFAULT_VERIFY_TOKENS` |
| `src/llama-context.cpp` (new) | Add `dflash_max_verify_tokens()` reading `GGML_DFLASH_MAX_VERIFY_TOKENS` env var |
| `src/llama-context.cpp:1521` | Replace `LLAMA_DFLASH_MAX_VERIFY_TOKENS` → `dflash_max_verify_tokens()` |
| `src/llama-context.cpp:4724` | Same |
| `src/llama-context.cpp:4732` | Same |
| `src/llama-context.cpp:6427` | Same |

Default stays 25 — zero behavioral change for flat-mode users.

## Expected Impact

| Config | `--spec-draft-n-max` | `--spec-branch-budget` | Total | Before | After |
|--------|---------------------|----------------------|-------|--------|-------|
| Flat baseline | 24 | 0 | 24 | ~70 tok/s | ~70 tok/s (unchanged) |
| Tree under cap | 16 | 8 | 24 | ~70 tok/s | ~80-90 tok/s |
| Tree over cap | 24 | 8 | 32 | ~3-5% acceptance | ~70-80 tok/s |

Sweet spot expected at budget 22 (n_max=16, branch_budget=6) per Lucebox's published sweep on the same RTX 3090 hardware.

## Upstream Potential

The fix can be upstreamed to `spiritbuun/buun-llama-cpp` as a trivial 2-call-site change (spiritbuun has only 2 sites, Bee has 4 because of tree-mode additions). The env-var pattern (`GGML_DFLASH_MAX_CTX`) is already established there. Not applicable to `ggml-org/llama.cpp` which has no DFlash code.

## Related

- `include/llama.h:1135`: enum definition
- `src/llama-context.cpp:4722-4732`: dispatch constraint checks
- `src/llama-context.cpp:1521,6427`: allocate_tape_gpu call sites
- `src/models/dflash_draft.cpp:8-14`: precedent `dflash_max_cross_ctx()` pattern
- `common/speculative.cpp:1975-2150`: existing `build_ddtree` algorithm
- Lucebox DDTree code: `dflash/src/common/ddtree.h` / `ddtree.cpp`
- Luce's kernel commit: `b16de6590` on `Luce-Org/llama.cpp@luce-dflash`
- `.tmp_commit_msg_enable-ddtree-tree-mode-with-spec-dflash-max-slots-measure-5-5-acceptance_20260518-185201.txt`: prior attempt hitting the cap
