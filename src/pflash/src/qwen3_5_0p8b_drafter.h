// Qwen3.5-0.8B drafter for pflash speculative prefill, in-process.
//
// Follows the same API pattern as qwen3_0p6b_drafter.h but handles the
// qwen35 hybrid architecture (M-RoPE + Gated DeltaNet + full attention).
//
// Public API:
//   bool load_qwen35_0p8b_drafter(path, backend, out)
//   bool forward_qwen35_0p8b_drafter(weights, ids, n_lookahead, running_max)
//   void free_qwen35_0p8b_drafter(weights)

#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;
struct ggml_backend_buffer;

namespace dflash27b {

// Per-layer weights for Qwen3.5 hybrid architecture.
// Each layer is either FULL ATTENTION or GATED DELTANET.
// Unused tensor pointers stay nullptr.
struct Qwen35DrafterLayer {
    // Shared across both layer types
    ggml_tensor * attn_norm      = nullptr;  // [hidden]
    ggml_tensor * ffn_norm       = nullptr;  // [hidden]
    ggml_tensor * w_gate         = nullptr;  // [hidden, intermediate] SwiGLU gate
    ggml_tensor * w_up           = nullptr;  // [hidden, intermediate] SwiGLU up
    ggml_tensor * w_down         = nullptr;  // [intermediate, hidden] SwiGLU down

    // Full-attention layer (non-null when (il+1) % full_attn_interval == 0)
    ggml_tensor * attn_q         = nullptr;  // [hidden, q_dim*2]  Q || gate packed
    ggml_tensor * attn_k         = nullptr;  // [hidden, kv_dim]
    ggml_tensor * attn_v         = nullptr;  // [hidden, kv_dim]
    ggml_tensor * attn_o         = nullptr;  // [q_dim, hidden]
    ggml_tensor * q_norm         = nullptr;  // [head_dim]
    ggml_tensor * k_norm         = nullptr;  // [head_dim]

    // Gated DeltaNet layer (non-null for all other layers)
    ggml_tensor * wqkv           = nullptr;  // fused Q/K/V projection
    ggml_tensor * wqkv_gate      = nullptr;  // "z" projection
    ggml_tensor * ssm_conv1d     = nullptr;  // [kernel, dim] depthwise causal conv
    ggml_tensor * ssm_alpha      = nullptr;  // [hidden, n_v_heads]
    ggml_tensor * ssm_beta       = nullptr;  // [hidden, n_v_heads]
    ggml_tensor * ssm_a          = nullptr;  // [dt_rank]
    ggml_tensor * ssm_dt_bias    = nullptr;  // [dt_rank]
    ggml_tensor * ssm_norm       = nullptr;  // [head_v_dim]
    ggml_tensor * ssm_out        = nullptr;  // [value_dim, hidden]
};

struct Qwen35DrafterWeights {
    ggml_context *        ctx     = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf     = nullptr;

    ggml_tensor * tok_embd    = nullptr;  // [hidden, vocab]
    ggml_tensor * out_norm    = nullptr;  // [hidden]
    ggml_tensor * output      = nullptr;  // [hidden, vocab] (lm_head)

    std::vector<Qwen35DrafterLayer> layers;

    // Architecture metadata — read from GGUF at load time
    int n_layer               = 24;
    int n_head                = 8;
    int n_head_kv             = 2;
    int n_embd                = 1024;
    int n_ff                  = 3584;
    int head_dim              = 256;
    int n_vocab               = 248320;
    int n_ctx_max             = 131072;
    int full_attn_interval    = 4;
    int ssm_state_size        = 128;
    int ssm_conv_kernel       = 4;
    int ssm_inner_size        = 2048;
    int ssm_dt_rank           = 16;
    int ssm_group_count       = 16;
    float rope_theta          = 10000000.0f;
    int rope_sections[4]      = {16, 16, 0, 0};  // 32 pairs = 64 rotated dims
};

bool load_qwen35_0p8b_drafter(const std::string & gguf_path,
                               ggml_backend_t backend,
                               Qwen35DrafterWeights & out);

void free_qwen35_0p8b_drafter(Qwen35DrafterWeights & w);

bool forward_qwen35_0p8b_drafter(
    const Qwen35DrafterWeights & w,
    const std::vector<int32_t> & ids,
    int n_lookahead,
    std::vector<float> & running_max);

} // namespace dflash27b
