# RSH-20260829-001: Qwen3.8 GPU Vision at Long Context
Opened: 2026-08-29 15-07-00 KST
Recorded by agent: codex

## Question

What production context remains durable for Qwen3.8-27B on one RTX 3090 when the BF16 vision projector and Q4 MTP head both stay on GPU?

## Findings

- A physical prompt batch of 128 reduced temporary flash-attention memory enough for a 196,608-token allocation to complete two near-cold multimodal requests.
- Both requests placed a 128x128 image after the long text, exercising the failure point seen in the 200,000-token allocation.
- Each request contained 192,576 prompt tokens. The first evaluated all 192,576 tokens; the second reused only 30 shared template tokens and evaluated 192,546 tokens.
- Both returned the correct image colors without CUDA OOM, truncation, or server failure.
- Prompt processing measured 353.68 and 349.87 tokens/s. Generation measured 25.20 and 30.70 tokens/s with MTP enabled.
- At roughly 90,000 prompt tokens, physical batch 128 measured about 606 tokens/s versus 764 tokens/s for batch 512. The smaller batch costs about 21% cold-prefill throughput there but preserves substantially more long-context memory.
- The 200,000-token allocation with physical batch 128 was not durable: it processed about 195,000 text tokens and then ran out of CUDA memory during the final image projection.

## Accepted Default

Use a 190,000-token context, physical prompt batch 128, GPU BF16 vision projector, GPU Q4 MTP head, flash attention, and symmetric turbo4 K/V for the Qwen3.8 reasoning aliases.

The production context is intentionally below the repeated 196,608-token allocation boundary so normal agent output and runtime variation retain headroom.
