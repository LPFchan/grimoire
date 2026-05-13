// decode_context.h — shared state for the dflash daemon pipeline.
//
// This header defines the single DecodeCtx struct that holds every piece of
// mutable state that was previously captured by reference inside the ~1000
// line generation lambda in test_dflash.cpp.  All decode_* modules receive
// a DecodeCtx& so they can read / update state without needing global
// variables.

#pragma once

#include "dflash27b.h"
#include "internal.h"
#include "dflash_graph.h"
#include "qwen3_drafter.h"
#include "sampler.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <unordered_map>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

// ggml-cuda dequantize helper (defined in a .cpp somewhere; declared here
// so decode modules can call it without pulling in test_dflash.cpp).
using to_fp32_cuda_t = void (*)(const void *, float *, int64_t, cudaStream_t);
to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type);

// Global config variables (defined in test_dflash.cpp; extern here).
// These are at file scope (not inside the dflash27b namespace) because
// test_dflash.cpp defines them at file scope with `using namespace dflash27b;`.
extern int g_kq_stride_pad;
extern int g_max_ctx_override;
extern int g_fa_window;
extern int g_draft_swa_window;
extern int g_draft_ctx_max;

// Global sampler state (defined in test_dflash.cpp; extern here).
namespace dflash27b { struct SamplerCfg; }
extern dflash27b::SamplerCfg g_sampler;
extern std::mt19937_64      g_sampler_rng;

namespace dflash27b {

// ── StepGraph (mirrors the definition in test_dflash.cpp) ─────────
struct StepGraph {
    ggml_context *  ctx = nullptr;
    ggml_cgraph *   gf  = nullptr;
    ggml_gallocr_t  alloc = nullptr;
    ggml_tensor *   inp_embed = nullptr;
    ggml_tensor *   positions = nullptr;
    ggml_tensor *   attn_mask = nullptr;
    ggml_tensor *   parent_ids = nullptr;
    ggml_tensor *   target_hidden_cat = nullptr;
    ggml_tensor *   positions_k = nullptr;
    ggml_tensor *   hidden_input = nullptr;
    ggml_tensor *   logits = nullptr;
    ggml_tensor *   hidden_states = nullptr;
    ggml_tensor *   argmax_tokens = nullptr;
    ggml_tensor *   topk_indices = nullptr;
    std::vector<DeltaNetCapture> delta_captures;
};

// ── DraftFeatureMirror ────────────────────────────────────────────
struct DraftFeatureMirror {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * target_feat = nullptr; // F32 [5*hidden, cap]
    void * bf16_staging = nullptr;
    size_t bf16_staging_elems = 0;
    int device = 0;
    int target_device = 0;
    int cap = 0;
};

// ── TargetLayerSplitShard ─────────────────────────────────────────
struct TargetLayerSplitShard {
    int gpu = 0;
    int layer_begin = 0;
    int layer_end = 0;
    ggml_backend_t backend = nullptr;
    TargetWeights weights;
    TargetCache cache;
    StepGraph layer_graph;
};

// ── ActivationPair ────────────────────────────────────────────────
struct ActivationPair {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;
    ggml_backend_t backend = nullptr;
    int n_tokens = 0;
};

// ── DDTree ────────────────────────────────────────────────────────
struct DDTree {
    int                         n_nodes = 0;          // excludes root
    std::vector<int32_t>        token_ids;            // size n_nodes
    std::vector<int>            depths;               // size n_nodes (1..L)
    std::vector<int>            parents;              // size n_nodes + 1
    std::vector<std::unordered_map<int32_t, int>> child_maps;  // size n_nodes + 1
    std::vector<uint8_t>        visibility;           // (1 + n_nodes)^2 row-major
};

// ── DecodeCtx — everything that was a local / captured variable ───
struct DecodeCtx {
    // ---- global / env-derived config --------------------------------
    int   kq_stride_pad = 32;
    int   max_ctx_override = 0;
    int   fa_window = 2048;
    int   draft_swa_window = 0;
    int   draft_ctx_max = 4096;

    SamplerCfg     sampler;
    std::mt19937_64 sampler_rng{std::random_device{}()};

    // ---- paths ------------------------------------------------------
    const char * target_path = nullptr;
    const char * draft_path  = nullptr;   // nullptr = no-draft mode

    // ---- model weights ----------------------------------------------
    TargetWeights  w;
    DraftWeights   dw;
    bool           has_draft = false;

    // ---- backends / GPUs --------------------------------------------
    ggml_backend_t target_backend = nullptr;
    ggml_backend_t draft_backend  = nullptr;
    int            target_gpu = 0;
    int            draft_gpu  = 0;
    bool           split_gpus = false;

    // ---- KV cache & snapshots ---------------------------------------
    TargetCache           cache;
    StepGraph             sg;            // target prefill / verify graph
    StepGraph             draft_sg;      // draft forward graph
    StepGraph             proj_sg;       // lm-head projection graph
    DraftFeatureMirror    feature_mirror;

    // ---- prefix-cache ring (daemon mode) ----------------------------
    static constexpr int PREFIX_CACHE_SLOTS = 8;
    PrefixSnapshot       prefix_snapshots[PREFIX_CACHE_SLOTS];

    // ---- runtime flags ----------------------------------------------
    bool daemon_mode          = false;
    bool seq_verify           = false;
    bool fast_rollback        = false;
    bool ddtree_mode          = false;
    int  ddtree_budget        = 64;
    float ddtree_temp         = 1.0f;
    bool ddtree_chain_seed    = true;
    bool draft_feature_mirror_flag = false;
    bool pflash               = false;
    bool target_parked        = false;
    bool draft_parked         = false;
    bool drafter_loaded       = false;

    // ---- stream fd (daemon mode) ------------------------------------
    int  stream_fd = -1;

    // ---- drafter context (pflash compression) -----------------------
    DrafterContext drafter_ctx;

    // ---- target-split shards (optional) -----------------------------
    std::vector<TargetLayerSplitShard> shards;
    bool target_split_dflash = false;

    // ---- per-request generation parameters (daemon mode) ------------
    int  n_gen = 0;
    std::string prompt_file_str;
    bool restore_from_slot = false;
    int  restore_slot_id = -1;
    bool chain_restore_requested = false;
    int  chain_thick_slot = -1;
    std::vector<int> chain_thin_ids;
    int  snap_pos = -1;
    int  snap_slot = -1;
    bool daemon_first_iter = true;

    // ---- decode state -----------------------------------------------
    int  committed = 0;
    int32_t last_tok = -1;

    // ---- convenience helpers ----------------------------------------
    int hidden() const { return DFLASH27B_TARGET_HIDDEN; }
    int vocab()  const { return DFLASH27B_TARGET_VOCAB; }
    int q_len()  const { return DFLASH27B_DRAFT_BLOCK_SIZE; }
};

// ── Small helpers (inline so all decode_* TUs can use them) ──────
inline void step_graph_free(StepGraph & sg) {
    ggml_free(sg.ctx);
    sg = StepGraph{};
}
inline void step_graph_destroy(StepGraph & sg) {
    if (sg.alloc) { ggml_gallocr_free(sg.alloc); sg.alloc = nullptr; }
    if (sg.ctx)   { ggml_free(sg.ctx);           sg.ctx  = nullptr; }
    sg = StepGraph{};
}
bool build_lm_head_projection_step(StepGraph & sg,
                                    const TargetWeights & w,
                                    ggml_backend_t backend,
                                    int n_tokens);
bool enable_peer_access_one_way(int device, int peer);
bool enable_peer_access_pair(int a, int b);
bool copy_peer_async(void * dst, int dst_device,
                     const void * src, int src_device,
                     size_t nbytes,
                     cudaStream_t stream = 0);
bool ensure_bf16_staging(DraftFeatureMirror & mirror, size_t elems);
void draft_feature_mirror_free(DraftFeatureMirror & mirror);
bool draft_feature_mirror_init(DraftFeatureMirror & mirror,
                                ggml_backend_t backend,
                                int device, int target_device, int cap);
bool draft_feature_mirror_can_view(const DraftFeatureMirror & mirror,
                                    int committed, int ctx_len, int & slot0);
bool draft_feature_mirror_sync_range(const TargetCache & cache,
                                      const DraftFeatureMirror & mirror,
                                      int start_pos, int n_tokens);
bool draft_feature_mirror_sync_tail(const TargetCache & cache,
                                     const DraftFeatureMirror & mirror,
                                     int committed);

// ── Graph builders (need external linkage for decode_* TUs) ───────
bool build_target_step(StepGraph & sg,
                       const TargetWeights & w,
                       TargetCache & cache,
                       ggml_backend_t backend,
                       int kv_start, int n_tokens,
                       bool with_mask, bool capture,
                       bool capture_delta_intermediate = false,
                       int fa_window = 0,
                       bool last_token_logits_only = false);
bool build_target_step_tree(StepGraph & sg,
                            const TargetWeights & w,
                            TargetCache & cache,
                            ggml_backend_t backend,
                            int kv_start, int n_tokens,
                            int fa_window = 0);
bool build_draft_step(StepGraph & sg,
                      const DraftWeights & dw,
                      const TargetWeights * tw,
                      ggml_backend_t backend,
                      int ctx_len,
                      const DraftFeatureMirror * mirror,
                      int committed);
void build_causal_mask(std::vector<uint16_t> & out,
                       int kv_len, int n_tokens, int kv_start,
                       int win_start = 0);

// ── Additional graph builders (defined in decode_graph.cpp) ───────
bool build_layer_step(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int layer_idx,
    ggml_tensor * act_in,
    ggml_tensor * act_out,
    int chunk_start,
    int n_tokens,
    int kv_start,
    bool with_mask,
    bool capture,
    int fa_window = 0);

void activation_pair_free(ActivationPair & p);
bool activation_pair_init(ActivationPair & p,
                          ggml_backend_t backend,
                          int hidden,
                          int n_tokens);

// ── DDTree helpers ────────────────────────────────────────────────
DDTree build_ddtree(const float * top_log_probs,
                    const int32_t * top_token_ids,
                    int L, int K, int budget,
                    bool chain_seed = true);
std::vector<int> follow_verified_tree(const DDTree & tree,
                                      const int32_t * posterior,
                                      int & out_next_token,
                                      int * out_node_idx = nullptr);
void extract_draft_topk(const float * logits,
                        int n_positions, int vocab, int K,
                        float * out_log_probs,
                        int32_t * out_token_ids,
                        float temperature = 1.0f);

// ── Layer-split harness (defined in decode_target_split.cpp) ──────
int run_target_layer_split_harness(
        const char * target_path,
        const char * draft_path,
        const char * prompt_path,
        int n_gen,
        const char * out_path,
        const std::vector<int> & target_gpus,
        const std::vector<double> & split_weights,
        int draft_gpu,
        bool load_draft,
        bool run_draft_smoke,
        bool run_dflash,
        int max_ctx,
        int max_verify_tokens);

// ── Other helpers ─────────────────────────────────────────────────
int argmax_f32(const float * x, int n);
std::vector<int32_t> read_int32_file(const std::string & path);
bool write_int32_file(const std::string & path, const std::vector<int32_t> & v);
bool parse_int_list(const char * text, std::vector<int> & out);
bool parse_float_list(const char * text, std::vector<double> & out);

#define IS_EOS_TOK(tok, w)                                         \
    ( ((w).eos_chat_id >= 0 && (tok) == (w).eos_chat_id)                  \
   || ((w).eos_id      >= 0 && (tok) == (w).eos_id     ) )

// ── Stream emit helper (wrapper around write(stream_fd, ...)) ─────
inline void stream_emit(DecodeCtx & ctx, int32_t tok) {
    if (ctx.stream_fd < 0) return;
    int32_t v = tok;
#if defined(_WIN32)
    DWORD written;
    WriteFile((HANDLE)(intptr_t)ctx.stream_fd, &v, sizeof(v), &written, nullptr);
#else
    ssize_t n = ::write(ctx.stream_fd, &v, sizeof(v));
    (void)n;
#endif
}

// ── Extracted decode functions ────────────────────────────────────
bool handle_daemon_command(DecodeCtx & ctx, const std::string & line);
bool run_dflash_decode(DecodeCtx & ctx, const std::vector<int32_t> & prompt, int n_gen, std::vector<int32_t> & out_all);
bool run_target_only_decode(DecodeCtx & ctx, const std::vector<int32_t> & prompt, int n_gen, std::vector<int32_t> & out_all);

} // namespace dflash27b
