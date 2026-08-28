# RSH-20260828-001: Qwen3.8 Long-Context Crash and Durable Context Boundary

Opened: 2026-08-28 15-24-01 KST
Recorded by agent: codex

## Question

Why did Qwen3.8-27B Q4 crash around 93k context despite a successful 200k fit test, why did the following request miss the cache, and what 200k configuration is durable on one 24 GiB GPU with vision and MTP retained?

## Incident Audit

The affected OpenCode session was `ses_fb997b39dffe2VDvbh83fcCCBj` (`Diagnosing internet outage at router 10.0.0.1`). Its database record contains 31 messages and no image, file, or attachment parts. Vision input did not trigger the original incident.

The cache worked before the crash. Requests reused the prefix from about 73,915 through 90,656 tokens. At 12:41:52 KST, a request completed with `n_tokens = 91386`. The next request selected the same slot with LCP similarity 0.992, then failed in flash-attention scratch allocation:

```text
ggml_cuda_pool_vmm::alloc
  -> launch_fattn<256,8,8>
  -> ggml_cuda_flash_attn_ext_mma_f16_case
  -> CUDA out of memory
```

The process abort destroyed its in-memory KV cache. After restart, OpenCode retried 92,113 input tokens with `cache_read = 0`. That miss was a consequence of the crash.

## Exact Production Cache Policy

Grimoire sets `TURBO_AUTO_ASYMMETRIC=0` in Compose. The served Qwen3.8 profile therefore keeps both K and V at turbo4. Early direct test containers omitted this environment variable, so the engine changed K from turbo4 to q8_0 for the model's 6:1 GQA ratio. Those tests measured a heavier q8_0-K/turbo4-V configuration and understated the context available to production.

At a 131,072-token allocation, the unintended q8_0-K test used about 23,216 MiB. The exact symmetric turbo4 production policy used about 22,186 MiB. Fit-only testing was still insufficient: model weights, the allocated KV window, BF16 projector, Q4_0 MTP head, compute buffers, and temporary f16 flash-attention buffers must all fit during a real near-limit request.

## Engine Patch

The current engine patch keeps f16 flash-attention K/V scratch outside the VMM pool. CUDA graphs are disabled, and the scoped buffers use direct allocation so working memory returns after each operation. This removes retained scratch growth, but it cannot make a physically overcommitted request fit.

## Durable 200k Configuration

The accepted test server used:

- 200,000 requested context, aligned by the engine to 200,192
- Qwen3.8-27B Q4_K_M target fully on GPU
- Q4_0 MTP head on GPU
- BF16 mmproj loaded with `--no-mmproj-offload`
- turbo4 K and turbo4 V with `TURBO_AUTO_ASYMMETRIC=0`
- low reasoning
- CUDA graphs disabled

Idle GPU use was 22,756 MiB. During the near-limit prompt it was about 22,980 MiB.

Two independent cold multimodal requests passed with zero cached prompt tokens:

| Prompt | Result | Wall time | MTP acceptance |
| --- | --- | ---: | ---: |
| 196,082 tokens, red/blue image | Correctly identified both colors | 396.675 s | 37/75 |
| 196,082 tokens, green/yellow image | `Green, Yellow` | 394.310 s | 46/66 |

The server PID did not change. The second prompt used a different repeated token and different image, so it was an independent cold replacement rather than a cache-hit replay.

A real read-only OpenCode audit then reached a 101,516-token parent conversation with normal append-only prefix reuse. OpenCode compacted it and continued in the same server process. Before compaction, a 72,501-token request was followed by a request that processed 1,644 new tokens and reused 72,497 cached tokens. The audit also launched four worker branches, causing repeated 20k-63k prompt changes and prompt-cache entry eviction. The soak ran for more than fifteen minutes without a server restart and was ended intentionally after the compaction and branch-churn behavior was established.

## Rejected 200k Alternatives

| Configuration | Result |
| --- | --- |
| BF16 mmproj on GPU, default micro-batch 512 | CUDA OOM at 90,112 prompt tokens |
| BF16 mmproj on GPU, micro-batch 256 | CUDA OOM at 170,014 prompt tokens |
| BF16 mmproj on GPU, micro-batch 128 | Reached 195,050 text tokens, then CUDA OOM while decoding the image embeddings |
| Forced q8_0 draft KV | Increased idle GPU use from 23,894 to 24,310 MiB by allocating a separate draft cache |
| MTP draft on CPU | Freed about 0.9 GiB but reduced generation to 26.8 tokens/s |
| One target layer on CPU | Freed about 0.9 GiB but reduced generation to 49.1 tokens/s |
| Quantized GPU mmproj | The bundled quantizer rejects the `clip` architecture |

The CPU mmproj profile keeps normal text generation and GPU MTP performance. Its cost is image ingestion: the same 128x128 image request took 31.411 seconds with CPU mmproj and 2.986 seconds with GPU mmproj. This cost applies to image-bearing turns, not the text-only OpenCode workload.

## Cache Findings

- A process crash necessarily loses the in-memory KV cache.
- Healthy append-only OpenCode turns reused the full stable prefix.
- Changing text before an image shifted the image-token positions and forced recomputation despite 0.995 LCP similarity. Image order and placement are part of the cache key in practice.
- OpenCode compaction deliberately replaces the old prompt prefix, so the first compacted request is a cold prefill.
- Reasoning/template/model-setting changes can also change the serialized prefix and prevent reuse.

## Accepted Configuration

- Aliases: `qwen3.8-27B-low`, `qwen3.8-27B-medium`, and `qwen3.8-27B-xhigh`
- Context: 200,000 requested; 200,192 engine-aligned
- Vision: BF16 projector retained, CPU execution via `--no-mmproj-offload`
- MTP: Q4_0 head retained on GPU
- KV policy: symmetric turbo4 K/V
- CUDA graphs: disabled

The plain `qwen3.8-27B` alias is not part of the accepted registry.

## Follow-up

A supported quantized mmproj could restore GPU vision latency if its quality and 200k scratch headroom are validated. The current bundled quantizer cannot produce that artifact.
