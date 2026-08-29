# RSH-20260829-002: Qwen3.8 CPU Vision Maximum Context
Opened: 2026-08-29 18-06-41 KST
Recorded by agent: codex

## Question

What is the durable context boundary for Qwen3.8-27B on one RTX 3090 when the BF16 vision projector runs on CPU, the physical prompt batch is 128, and the Q4 MTP head remains on GPU?

## Boundary Results

All requests used symmetric turbo4 K/V, low reasoning, a cold prompt, and a 128x128 image after the long text.

| Allocated context | Idle VRAM | Prompt result |
| ---: | ---: | --- |
| 262,144 | 23,868 MiB | CUDA OOM at 108,574 prompt tokens |
| 245,760 | 23,468 MiB | CUDA OOM at 210,974 prompt tokens |
| 241,664 | 23,368 MiB | CUDA OOM at 237,598 prompt tokens |
| 237,568 | 23,268 MiB | Passed 235,076 prompt tokens |
| 237,568 | 23,268 MiB | Independent cold repeat passed 235,076 prompt tokens |
| 229,376 | 23,068 MiB | Passed 226,076 prompt tokens |

The two 237,568-token allocation passes used different repeated text and different images. Both processed all 235,076 prompt tokens without cache reuse, correctly identified the image colors, generated with MTP, and completed without truncation or server failure.

| 237,568 pass | Prompt speed | Generation speed | Result |
| --- | ---: | ---: | --- |
| Navy/gold | 289.49 tok/s | 23.85 tok/s | Correct |
| Black/white | 288.39 tok/s | 27.20 tok/s | Correct |

## Conclusion

Use 237,568 allocated context with CPU BF16 vision projection and physical prompt batch 128. The next tested boundary, 241,664, fails near the end of the text prompt, leaving only a 4,096-token allocation gap above the repeated passing boundary.

The accepted boundary leaves about 2,492 tokens between the tested 235,076-token prompt and the allocated window. CPU image preprocessing remains substantially slower than GPU projection, but text-only generation and GPU MTP remain intact.
