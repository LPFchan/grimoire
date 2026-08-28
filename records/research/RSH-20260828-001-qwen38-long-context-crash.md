# RSH-20260828-001: Qwen3.8 Long-Context Crash and OpenCode Soak

Opened: 2026-08-28 15-24-01 KST
Recorded by agent: codex

## Question

Why did the 27B Q4 model crash around 93k context despite a successful 200k fit test, why did the following request miss the cache, and how much real OpenCode conversation can the service carry while retaining vision?

## Original Incident

The affected OpenCode session was `ses_fb997b39dffe2VDvbh83fcCCBj` (`Diagnosing internet outage at router 10.0.0.1`). Its database record contains 31 messages and no image, file, or attachment parts. The only part types are text, reasoning, tool, step-start, and step-finish. Vision input did not trigger this incident.

The server cache was working before the crash. Requests reused the prefix from about 73,915 through 90,656 tokens. At 12:41:52 KST, a request completed with `n_tokens = 91386`. The next request selected the same slot with LCP similarity 0.992, then failed nine milliseconds later in:

```text
ggml_cuda_pool_vmm::alloc
  -> launch_fattn<256,8,8>
  -> ggml_cuda_flash_attn_ext_mma_f16_case
  -> cuMemCreate: CUDA_ERROR_OUT_OF_MEMORY
```

The process abort destroyed its in-memory KV cache. After restart, OpenCode retried 92,113 input tokens and reported `cache_read = 0`. This cache miss was a consequence of the server crash, not the cause.

## Why the 200k Fit Test Was Misleading

The 4-bit figure describes model weights, not the complete live allocation. With 200,192 allocated context, turbo4 K/V, BF16 vision projection, and the Q4_0 MTP head, measured startup use was:

| Component set | GPU memory |
| --- | ---: |
| Base model and 200k KV | 20,084 MiB |
| Vision projector | +1,138 MiB |
| MTP head | +2,672 MiB |
| Full service | about 23,894 MiB |

That left only a few hundred MiB on a 24 GiB card. Flash attention temporarily dequantizes quantized K/V into f16 scratch buffers. Those buffers grow with the active prompt. The existing CUDA VMM pool retained its high-water physical mappings, so a prompt could fit in the preallocated KV cache while still failing when flash attention requested temporary working memory.

## Patch Experiment

The old Atomic patch that kept f16 flash-attention scratch outside the VMM pool had disappeared when Grimoire migrated to the current TheTom fork. A port to the pinned engine SHA was built with CUDA graphs disabled. The temporary NVIDIA K/V buffers use scoped `cudaMalloc`/`cudaFree`, so they return memory after each operation instead of permanently raising the VMM pool high-water mark.

With vision and MTP enabled, synthetic text requests succeeded at 74,058, 88,058, 93,058, and 100,058 tokens. A 91,089-token request containing a real 128x128 image also succeeded. A 101,089-token image request still failed: the full-stack baseline was so close to the card limit that even correctly released scratch could not physically fit. A real OpenCode audit also still crashed after reaching 89,588 tokens when its next tool result added 12,288 tokens. GPU use had reached 24,288 MiB before that allocation.

This separates two problems:

1. Retained flash-attention scratch caused premature failure and is fixed by the patch.
2. The MTP head consumes about 2.7 GiB, leaving insufficient working room for a long, irregular agent workload. No allocator fix can make a physically overcommitted request fit.

## Real OpenCode Soak With Vision Kept and MTP Disabled

The same patched server was run with the BF16 vision projector still loaded and only MTP removed. Startup use fell to 21,206 MiB.

A fresh OpenCode session (`ses_fb90e6a43ffeHCkxSJDRzqgU4I`) performed a read-only repository audit using real searches and file reads. It crossed the old crash point and progressed through 98k, 106k, 116k, 126k, 137k, 149k, 155k, and 162k tokens. The same server process survived throughout and usually returned to about 21.2 GiB between requests.

When the session was resumed in a second `opencode run --session` invocation, its first request reprocessed 162,330 prompt tokens with no cache hit even though the server had not restarted. Subsequent steps reused the prefix normally. This is a separate client/payload-prefix behavior across CLI invocations, not cache loss from an inference crash.

The conversation reached 169,683 tokens. OpenCode then compacted it automatically: the summary request was 19,237 total tokens, and the next working request was about 90,764 tokens because OpenCode's system prompt and tool schema alone occupy roughly 72k. The agent continued making tool calls after compaction and again exceeded 140k tokens without a server restart.

The resumed run made 36 real tool calls across 21 model steps. It was deliberately stopped at 161,539 tokens on the second climb after compaction; the server was still healthy at about 21.2 GiB.

A final vision A/B sent 100,000 repeated text tokens plus the same real red/blue image from a cold prefix. The server counted 101,082 prompt tokens, returned HTTP 200 in 138.147 seconds, and correctly identified red and blue. The server process survived. This is the request shape that failed at 101,089 prompt tokens with MTP loaded.

The practical limit for this OpenCode configuration is therefore not the nominal 200k model window. OpenCode deliberately compacts around 170k, then continues the same conversation from a smaller summary. The no-MTP server carried that full cycle successfully.

## Conclusion

- The original session contained no images. Its failure was CUDA working-memory exhaustion during flash attention.
- Cache hits worked until the process aborted; the immediate miss was expected because an in-memory cache cannot survive a process restart.
- A 200k KV allocation test proves that static weights and cache fit. It does not prove that the runtime has enough headroom for flash-attention scratch, vision, MTP, and bursty tool-result prefills at the same time.
- Vision can remain enabled. The reliable configuration is the scoped flash-attention scratch patch, CUDA graphs off, turbo4 K/V, BF16 vision projection, and no MTP head.
- MTP is a throughput feature, not part of the model's core capability. Removing it trades generation speed for about 2.7 GiB of safety margin and is what allowed the real agent conversation to reach OpenCode's own compaction boundary.

## Follow-up

CUDA graphs remain disabled for this scoped-allocation implementation. A graph-safe reusable scratch reservation could restore graphs later, but it is not required for the verified long-context configuration.
