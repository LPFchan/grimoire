# PFlash Head/Tail Block Protection: Layout Rule for Triggering Compression (2026-05-18)

Opened: 2026-05-18 18-40-00 KST
Recorded by agent: opencode

## Question

Why does PFlash compression decline ("no compressible blocks found") even when the prompt exceeds the prefill threshold?

## Mechanism

`maybe_compress()` in `prefill.py` calls `_protected_block_indexes()` to determine which prompt blocks should be preserved verbatim. This function:

1. **Head protection**: Protects the first 2 content messages (or 1 if only 1 exists).
2. **Tail protection**: Starts from the last content message and walks backwards, protecting whole message groups until `tail_so_far > tail_budget` (default 16K tokens).
3. **Compressible blocks**: Only unprotected blocks with `token_count >= 256` are compressible. Their total must sum to `>= 1024` tokens for compression to fire.

## The Trap: Single-Message Prompts

A single user message creates ONE prompt block. Head protection covers it (first content message = the only message). Result: `compressible_indexes = {}`, `total_compressible = 0`, compression declines.

## The Fix: 4+ Messages with Large Tail Breaker

To trigger compression, the conversation must have:

```
system (small)    → head (1st content)
user1 (small)     → head (2nd content)
asst1 (small)     → caught by tail loop, but...
user2 (BIG, >16K) → tail loop breaks here (> budget)
```

The LAST content block must exceed `tail_budget` (16K) by itself so the tail loop breaks before reaching earlier blocks. Those earlier blocks become the compressible middle.

Minimum viable layout:

```
system (10 tok)       → head (protected)
user1 (100 tok)       → head (protected)
asst1 (100 tok)       → compressible middle (>=256 tok each)
user2 (>16000 tok)    → tail budget breaker (protected)
```

The `test_pflash_pipeline.py` `build_multi_turn_prompt()` function with `TURNS >= 2` and large enough total chars naturally produces this layout.

## The `_default_prompt_blocks` Escalation

When no explicit blocks are provided (non-Qwen models or malformed messages), `_default_prompt_blocks` creates a SINGLE block for the entire prompt. `_is_default_prompt_block` returns True, causing `_protected_block_indexes` to return an empty set (no protection). This SHOULD make the entire block compressible — but this path also skips the per-message boundaries that the protection logic uses, meaning the "no compressible blocks" error from the head/tail trap only applies when explicit blocks ARE provided.

## Implications

- PFlash compression tests must use multi-turn conversations (4+ messages) with the last message large enough to exhaust the 16K tail budget
- Single-message prompts WILL decline compression (head protection blocks them)
- The tail budget default (16K) means the last message should be >16K tokens to force the tail loop to break
