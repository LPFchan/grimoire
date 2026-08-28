# RSH-20260828-001: Qwen3.8 Long-Context Crash and Durable Context Boundary

Opened: 2026-08-28 15-24-01 KST
Recorded by agent: codex

## Question

Why did Qwen3.8-27B Q4 crash around 93k context despite a successful 200k fit test, why did the following request miss the cache, and what context is durable on a 24 GiB GPU with MTP retained?

## Original Incident

The affected OpenCode session was `ses_fb997b39dffe2VDvbh83fcCCBj` (`Diagnosing internet outage at router 10.0.0.1`). Its database record contains 31 messages and no image, file, or attachment parts. Vision input did not trigger the incident.

The cache worked before the crash. Requests reused the prefix from about 73,915 through 90,656 tokens. At 12:41:52 KST, a request completed with `n_tokens = 91386`. The next request selected the same slot with LCP similarity 0.992, then failed nine milliseconds later in flash-attention scratch allocation:

```text
ggml_cuda_pool_vmm::alloc
  -> launch_fattn<256,8,8>
  -> ggml_cuda_flash_attn_ext_mma_f16_case
  -> cuMemCreate: CUDA_ERROR_OUT_OF_MEMORY
```

The process abort destroyed its in-memory KV cache. After restart, OpenCode retried 92,113 input tokens with `cache_read = 0`. The miss was a consequence of the crash.

## Why the 200k Fit Test Was Insufficient

Four-bit quantization describes the model weights, not the complete live allocation. The service also carries the allocated KV window, BF16 vision projector, Q4_0 MTP head, compute buffers, and temporary flash-attention buffers. With all features and a 200k window, startup left only a few hundred MiB free.

Flash attention temporarily dequantizes quantized K/V into f16 buffers whose size grows with the active prompt. The old CUDA VMM path retained its high-water physical mappings, so an allocated KV window could fit while a later prompt still failed when it needed working memory.

## Engine Patch

The current engine patch keeps f16 flash-attention K/V scratch outside the VMM pool. CUDA graphs are disabled, and the scoped buffers use `cudaMalloc`/`cudaFree` so working memory returns after each operation. This removes retained scratch growth, but it cannot make a physically overcommitted request fit.

## Durable Boundary: MTP and Vision Enabled

Tests used `qwen3.8-27B-low`, the BF16 vision projector, the Q4_0 MTP head, turbo4 K/V policy, and low reasoning.

| Allocated context | Startup/result | Near-limit result |
| ---: | --- | --- |
| 131,072 | Clean start, 23,216 MiB | Passed 125,582-token and 128,082-token cold image requests; both identified red and blue correctly |
| 147,456 | Clean start, 23,780 MiB | Crashed during cold image prefill at 133,150 processed tokens |
| 163,840 | Opened a port after a failed 248 MiB startup allocation; 24,094 MiB resident | Rejected as non-durable without prompt testing |

The second 128,082-token image request reused only 30 tokens, so it was effectively another full cold prefill rather than an easy cache-hit repeat.

A real read-only OpenCode repository audit on the 131,072 server progressed through 72,930, 79,777, 100,959, 106,549, and 117,676 tokens with normal prefix reuse. OpenCode then compacted successfully to a 15,894-token summary. The same session resumed at 75,539 tokens and continued through 87,447 and 103,937 tokens. The server PID did not change.

The durable MTP-plus-vision context is therefore **131,072 tokens**. OpenCode's observed compaction point for this configuration was **117,676 tokens**.

## Comparison: MTP Enabled, Vision Unloaded

| Allocated context | Startup/result | Near-limit result |
| ---: | --- | --- |
| 163,840 | Clean start, 23,206 MiB | Passed two different 159,046-token cold prompts and a 159,546-token extension with 158,530 cached tokens |
| 180,224 | Clean start, 23,770 MiB | Crashed during cold text prefill at 135,198 processed tokens |
| 200,000 | Failed startup | MTP context could not allocate a 263.52 MiB compute buffer |

A real OpenCode audit on the 163,840 text-only server reached 153,092 tokens with prefix reuse, compacted successfully to an 18,400-token summary, then continued at 86,419 and 95,620 tokens. The first post-compaction request was a full 86,200-token prefill and also survived.

Unloading vision therefore buys **32,768 tokens** of tested durable allocated context: 163,840 instead of 131,072. It does not make 180k or 200k durable because spending all freed memory on a larger KV allocation recreates the same lack of flash-attention working room.

## Cache Findings

- The original post-crash miss was expected: the server process and its in-memory cache had died.
- Within healthy OpenCode runs, long prefixes were repeatedly reused.
- A separate resumed `opencode run --session` invocation once reprocessed 162,330 tokens without a server restart. Subsequent steps cached normally. This is a client/request-prefix behavior across CLI invocations, separate from crash-related cache loss.
- Multimodal cold-replacement tests reused only 30 tokens, which made them useful full-prefill durability tests.

## Accepted Configuration

- Reasoning aliases: `qwen3.8-27B-low`, `qwen3.8-27B-medium`, and `qwen3.8-27B-xhigh`
- Context: 131,072
- Vision: BF16 projector enabled
- MTP: Q4_0 head enabled
- KV policy: turbo4 registry policy; the engine automatically upgrades K to q8_0 for this 6:1 GQA model
- CUDA graphs: disabled for scoped flash-attention scratch allocation

The plain `qwen3.8-27B` alias is not part of the accepted registry.

## Follow-up

A graph-safe reusable scratch reservation could restore CUDA graphs later. It is not required for the tested durable configuration.
