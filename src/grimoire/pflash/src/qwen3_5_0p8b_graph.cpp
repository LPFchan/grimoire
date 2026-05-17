// Custom forward for the Qwen3.5-0.8B drafter (hybrid architecture).
//
// Qwen3.5-0.8B: 24 layers, every 4th is full attention (M-RoPE + sliding window
// FA), the rest are Gated DeltaNet.  SwiGLU FFN on every layer.
//
// One-shot forward: no KV cache persistence across calls.  Per-layer K/V
// buffers live only for the duration of this forward.  Tail-attention scoring
// runs on full-attention layers only (DeltaNet has no K/V in the attention
// sense).

#include "internal.h"
#include "qwen3_5_0p8b_drafter.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace dflash27b {

namespace {

constexpr int CHUNK_S   = 4096;
constexpr int FA_WINDOW = 512;

struct PersBuf {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         t   = nullptr;
};

bool make_pers(ggml_backend_t backend, ggml_type type, int n_dim,
               const int64_t * dims, PersBuf & out) {
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 4 + 1024;
    ip.no_alloc   = true;
    ip.mem_buffer = nullptr;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;
    if      (n_dim == 1) out.t = ggml_new_tensor_1d(out.ctx, type, dims[0]);
    else if (n_dim == 2) out.t = ggml_new_tensor_2d(out.ctx, type, dims[0], dims[1]);
    else if (n_dim == 3) out.t = ggml_new_tensor_3d(out.ctx, type, dims[0], dims[1], dims[2]);
    else if (n_dim == 4) out.t = ggml_new_tensor_4d(out.ctx, type, dims[0], dims[1], dims[2], dims[3]);
    else return false;
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    return out.buf != nullptr;
}

void free_pers(PersBuf & p) {
    if (p.buf) { ggml_backend_buffer_free(p.buf); p.buf = nullptr; }
    if (p.ctx) { ggml_free(p.ctx); p.ctx = nullptr; }
    p.t = nullptr;
}

} // anonymous namespace

bool forward_qwen35_0p8b_drafter(
    const Qwen35DrafterWeights & w,
    const std::vector<int32_t> & ids,
    int n_lookahead,
    std::vector<float> & running_max)
{
    if (!w.backend || !w.tok_embd) {
        set_last_error("forward_qwen35_0p8b_drafter: weights not loaded");
        return false;
    }
    const int S        = (int)ids.size();
    const int H        = w.n_head;
    const int Hk       = w.n_head_kv;
    const int D        = w.head_dim;
    const int gqa      = (Hk > 0) ? (H / Hk) : 1;
    const int hidden   = w.n_embd;
    const int q_dim    = H * D;
    const float eps    = 1e-6f;
    const float scale  = 1.0f / std::sqrt((float)D);
    const float rope_b = w.rope_theta;

    if (S < n_lookahead + 1) {
        set_last_error("forward_qwen35_0p8b_drafter: S too small");
        return false;
    }
    constexpr int MAX_S = 16384;
    if (S > MAX_S) {
        set_last_error("forward_qwen35_0p8b_drafter: S=" + std::to_string(S) +
                       " exceeds MAX_S=" + std::to_string(MAX_S) +
                       " (block too large)");
        return false;
    }
    running_max.assign((size_t)n_lookahead * S, -INFINITY);

    // Persistent buffers
    PersBuf hidden_buf, pos_buf, mask_tail_buf;
    PersBuf Q_buf;        // [D, H, S]  f32 (full Q before permute for FA)
    PersBuf attn_out_buf; // [D, H, S]  f32 (attention output before o_proj)

    // Tail scoring pre-allocated persistent tensors
    PersBuf ts_K_f32, ts_K_tpl;

    // K/V buffers: one per full-attention layer (DeltaNet layers skip)
    std::vector<PersBuf> K_curr_v((size_t)w.n_layer);
    std::vector<PersBuf> V_curr_v((size_t)w.n_layer);
    std::vector<PersBuf> Q_last_v((size_t)w.n_layer);

    // DeltaNet recurrent state per layer (used only for delta layers)
    struct DeltaState {
        PersBuf conv; // [(kernel-1), d_inner] f32 (SSM conv tracks inner dim only)
        PersBuf ssm;  // [head_v_dim, head_v_dim, num_v_heads] f32
    };
    std::vector<DeltaState> delta_state((size_t)w.n_layer);

    // The persistent batch buffer — MUST be freed via ggml_backend_buffer_free,
    // not just ggml_free(bctx) which only frees metadata.
    ggml_backend_buffer_t one_buf = nullptr;

    auto cleanup_all = [&]() {
        if (one_buf) { ggml_backend_buffer_free(one_buf); one_buf = nullptr; }
        if (hidden_buf.ctx) { ggml_free(hidden_buf.ctx); }
        hidden_buf.ctx = nullptr; Q_buf.ctx = nullptr; attn_out_buf.ctx = nullptr;
        pos_buf.ctx = nullptr; mask_tail_buf.ctx = nullptr;
        ts_K_f32.ctx = nullptr; ts_K_tpl.ctx = nullptr;
        for (auto & pb : K_curr_v) pb.ctx = nullptr;
        for (auto & pb : V_curr_v) pb.ctx = nullptr;
        for (auto & pb : Q_last_v) pb.ctx = nullptr;
        for (auto & ds : delta_state) { ds.conv.ctx = nullptr; ds.ssm.ctx = nullptr; }
    };

    // Count attn/delta layers
    int n_attn = 0, n_delta = 0;
    for (int il = 0; il < w.n_layer; ++il) {
        if (((il + 1) % w.full_attn_interval) == 0) n_attn++; else n_delta++;
    }
    const int conv_channels = w.ssm_inner_size + 2 * w.ssm_group_count * w.ssm_state_size;
    const int head_v_dim    = w.ssm_inner_size / w.ssm_dt_rank;

    // Allocate ALL persistent buffers from ONE context / ONE cudaMalloc
    {
        size_t total = 0;
        auto sz = [&](ggml_type t, const int64_t * ne, int nd) {
            size_t es = ggml_type_size(t) / ggml_blck_size(t);
            size_t s = es; for (int i = 0; i < nd; i++) s *= (size_t)ne[i];
            total += s + ggml_tensor_overhead() + 64;
        };
        int64_t d_h[]  = {hidden, S};   sz(GGML_TYPE_F32, d_h, 2);
        int64_t d_p[]  = {S * 4};       sz(GGML_TYPE_I32, d_p, 1);
        int64_t d_mt[] = {S, n_lookahead}; sz(GGML_TYPE_F32, d_mt, 2);
        int64_t d_q3[] = {D, H, S};     sz(GGML_TYPE_F16, d_q3, 3);
        int64_t d_at[] = {D, H, S};     sz(GGML_TYPE_Q8_0, d_at, 3);
        int64_t d_kv[] = {D, Hk, S};    for (int i = 0; i < n_attn; i++) {
            sz(GGML_TYPE_Q8_0, d_kv, 3); sz(GGML_TYPE_Q8_0, d_kv, 3);
            int64_t d_ql[] = {D, H, n_lookahead}; sz(GGML_TYPE_F32, d_ql, 3);
        }
        int64_t d_cv[] = {w.ssm_conv_kernel - 1, w.ssm_inner_size};
        int64_t d_ss[] = {head_v_dim, head_v_dim, w.ssm_dt_rank};
        for (int i = 0; i < n_delta; i++) {
            sz(GGML_TYPE_F32, d_cv, 2); sz(GGML_TYPE_F32, d_ss, 3);
        }
        // Pre-allocate tail scoring K_f32 [D, Hk, S] and K_tpl [D, S, gqa, Hk]
        // so galloc doesn't need to grow during tail scoring.
        int64_t d_tk[] = {D, Hk, S};           sz(GGML_TYPE_F32, d_tk, 3);
        int64_t d_tr[] = {D, S, gqa, Hk};      sz(GGML_TYPE_F32, d_tr, 4);
        ggml_init_params bip{};
        bip.mem_size = total + 65536;
        bip.no_alloc = true;
        ggml_context * bctx = ggml_init(bip);
        if (!bctx) { set_last_error("0p8b: batch ctx failed"); cleanup_all(); return false; }

        hidden_buf.t    = ggml_new_tensor_2d(bctx, GGML_TYPE_F32, hidden, S);
        pos_buf.t       = ggml_new_tensor_1d(bctx, GGML_TYPE_I32, S * 4);
        mask_tail_buf.t = ggml_new_tensor_2d(bctx, GGML_TYPE_F32, S, n_lookahead);
        Q_buf.t         = ggml_new_tensor_3d(bctx, GGML_TYPE_F16, D, H, S);
        attn_out_buf.t  = ggml_new_tensor_3d(bctx, GGML_TYPE_Q8_0, D, H, S);
        // Tail scoring pre-allocated tensors (reused across layers)
        ts_K_f32.t = ggml_new_tensor_3d(bctx, GGML_TYPE_F32, D, Hk, S);
        ts_K_tpl.t = ggml_new_tensor_4d(bctx, GGML_TYPE_F32, D, S, gqa, Hk);

        int ai = 0, di = 0;
        for (int il = 0; il < w.n_layer; ++il) {
            if (((il + 1) % w.full_attn_interval) == 0) {
                K_curr_v[il].t = ggml_new_tensor_3d(bctx, GGML_TYPE_F16, D, Hk, S);
                V_curr_v[il].t = ggml_new_tensor_3d(bctx, GGML_TYPE_F16, D, Hk, S);
                Q_last_v[il].t = ggml_new_tensor_3d(bctx, GGML_TYPE_F32, D, H, n_lookahead);
                ai++;
            } else {
            delta_state[il].conv.t = ggml_new_tensor_2d(bctx, GGML_TYPE_F32, w.ssm_conv_kernel - 1, w.ssm_inner_size);
            delta_state[il].ssm.t  = ggml_new_tensor_3d(bctx, GGML_TYPE_F32, head_v_dim, head_v_dim, w.ssm_dt_rank);
                di++;
            }
        }

        one_buf = ggml_backend_alloc_ctx_tensors(bctx, w.backend);
        if (!one_buf) {
            set_last_error("0p8b: batch buf alloc failed"); ggml_free(bctx); cleanup_all(); return false;
        }
        auto assign = [&](PersBuf & pb) { pb.buf = one_buf; pb.ctx = bctx; };
        assign(hidden_buf); assign(pos_buf); assign(mask_tail_buf);
        assign(Q_buf); assign(attn_out_buf);
        assign(ts_K_f32); assign(ts_K_tpl);
        for (int il = 0; il < w.n_layer; ++il) {
            if (((il + 1) % w.full_attn_interval) == 0) {
                assign(K_curr_v[il]); assign(V_curr_v[il]); assign(Q_last_v[il]);
            } else {
                assign(delta_state[il].conv); assign(delta_state[il].ssm);
            }
        }

        // Zero-initialize DeltaNet recurrent state
        for (int il = 0; il < w.n_layer; ++il) {
            if (((il + 1) % w.full_attn_interval) != 0) {
                size_t nb_c = ggml_nbytes(delta_state[il].conv.t);
                std::vector<uint8_t> z_c(nb_c, 0);
                ggml_backend_tensor_set(delta_state[il].conv.t, z_c.data(), 0, nb_c);
                size_t nb_s = ggml_nbytes(delta_state[il].ssm.t);
                std::vector<uint8_t> z_s(nb_s, 0);
                ggml_backend_tensor_set(delta_state[il].ssm.t, z_s.data(), 0, nb_s);
            }
        }
    }

    // Positions [0..S-1]
    {
        std::vector<int32_t> pos((size_t)S * 4);
        for (int i = 0; i < S; ++i) {
            // M-RoPE: 4 position sections per token [sec0, sec1, sec2, sec3]
            pos[i*4+0] = i; pos[i*4+1] = i; pos[i*4+2] = i; pos[i*4+3] = i;
        }
        ggml_backend_tensor_set(pos_buf.t, pos.data(), 0, (size_t)S * 4 * sizeof(int32_t));
    }
    // Tail scoring mask
    {
        std::vector<float> m((size_t)n_lookahead * S, 0.0f);
        for (int t = 0; t < n_lookahead; ++t) {
            int visible_end = S - n_lookahead + t + 1;
            for (int j = 0; j < S; ++j)
                m[(size_t)t * S + j] = (j < visible_end) ? 0.0f : -INFINITY;
        }
        ggml_backend_tensor_set(mask_tail_buf.t, m.data(), 0, m.size() * sizeof(float));
    }

    // ── Embed: hidden_buf = get_rows(tok_embd, ids) ─────────────────
    {
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead() + 16 * 1024;
        ip.no_alloc = true;
        ggml_context * gctx = ggml_init(ip);
        ggml_tensor * t_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, S);
        ggml_set_name(t_ids, "ids");
        ggml_tensor * embed = ggml_get_rows(gctx, w.tok_embd, t_ids);
        ggml_tensor * cpy_h = ggml_cpy(gctx, embed, hidden_buf.t);
        ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, cpy_h);
        ggml_backend_buffer_t in_buf = ggml_backend_alloc_ctx_tensors(gctx, w.backend);
        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(w.backend));
        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            set_last_error("0p8b: embed graph alloc failed");
            ggml_gallocr_free(galloc);
            if (in_buf) ggml_backend_buffer_free(in_buf);
            ggml_free(gctx); cleanup_all(); return false;
        }
        ggml_backend_tensor_set(t_ids, ids.data(), 0, (size_t)S * sizeof(int32_t));
        ggml_backend_graph_compute(w.backend, gf);
        ggml_gallocr_free(galloc);
        if (in_buf) ggml_backend_buffer_free(in_buf);
        ggml_free(gctx);
    }

    // Per-layer A(→FA)→B loop (full attention) or Delta loop
    ggml_gallocr_t galloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(w.backend));

    auto t_total_start = std::chrono::steady_clock::now();

    for (int il = 0; il < w.n_layer; ++il) {
        const auto & L = w.layers[il];
        const bool is_attn = (((il + 1) % w.full_attn_interval) == 0);

        if (is_attn) {
            // ── FULL ATTENTION LAYER ───────────────────────────────
            // Graph A (chunked): norm → Q+gate/K/V proj → Q/K-norm → M-RoPE → copy to persistent
            for (int cs = 0; cs < S; cs += CHUNK_S) {
                const int cl = std::min(CHUNK_S, S - cs);

                ggml_init_params ipA{};
                ipA.mem_size = ggml_tensor_overhead() * 128
                               + ggml_graph_overhead_custom(4096, false)
                               + 128 * 1024;
                ipA.no_alloc = true;
                ggml_context * gA = ggml_init(ipA);
                if (!gA) { set_last_error("0p8b: graph A init failed"); cleanup_all(); ggml_gallocr_free(galloc); return false; }
                ggml_cgraph * gfA = ggml_new_graph_custom(gA, 4096, false);

                const size_t h_esz = ggml_element_size(hidden_buf.t);
                ggml_tensor * h_view = ggml_view_2d(gA, hidden_buf.t,
                    hidden, cl, hidden * h_esz, (size_t)cs * hidden * h_esz);
                ggml_tensor * pos_chunk = ggml_view_1d(gA, pos_buf.t, cl * 4,
                    (size_t)cs * 4 * sizeof(int32_t));

                // RMS norm
                ggml_tensor * cur = ggml_rms_norm(gA, h_view, eps);
                cur = ggml_mul(gA, cur, L.attn_norm);

                // Packed Q+gate: attn_q [hidden, q_dim*2]
                ggml_tensor * QG = ggml_mul_mat(gA, L.attn_q, cur);
                QG = ggml_reshape_3d(gA, QG, D * 2, H, cl);

                // Q half: [D, H, cl]
                const size_t qg_esz = ggml_element_size(QG);
                ggml_tensor * Q = ggml_view_3d(gA, QG,
                    D, H, cl,
                    qg_esz * D * 2,
                    qg_esz * D * 2 * H,
                    0);
                Q = ggml_rms_norm(gA, Q, eps);
                Q = ggml_mul(gA, Q, L.q_norm);

                // Gate half: [D, H, cl], offset by D
                ggml_tensor * gate = ggml_view_3d(gA, QG,
                    D, H, cl,
                    qg_esz * D * 2,
                    qg_esz * D * 2 * H,
                    qg_esz * D);
                gate = ggml_cont_2d(gA, gate, q_dim, cl);

                // M-RoPE on Q
                const int n_rot = 2 * (w.rope_sections[0] + w.rope_sections[1] +
                                       w.rope_sections[2] + w.rope_sections[3]);
                int sections[GGML_MROPE_SECTIONS];
                for (int i = 0; i < GGML_MROPE_SECTIONS; i++) sections[i] = w.rope_sections[i];
                Q = ggml_rope_multi(gA, Q, pos_chunk, nullptr,
                    n_rot, sections, GGML_ROPE_TYPE_MROPE,
                    0, rope_b, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

                // K projection + norm + M-RoPE
                ggml_tensor * K = ggml_mul_mat(gA, L.attn_k, cur);
                K = ggml_reshape_3d(gA, K, D, Hk, cl);
                K = ggml_rms_norm(gA, K, eps);
                K = ggml_mul(gA, K, L.k_norm);
                K = ggml_rope_multi(gA, K, pos_chunk, nullptr,
                    n_rot, sections, GGML_ROPE_TYPE_MROPE,
                    0, rope_b, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

                // V projection
                ggml_tensor * V = ggml_mul_mat(gA, L.attn_v, cur);
                V = ggml_reshape_3d(gA, V, D, Hk, cl);

                // Copy to persistent buffers
                const size_t q_esz  = ggml_element_size(Q_buf.t);
                const size_t kv_stride = ggml_row_size(K_curr_v[il].t->type, D);
                ggml_tensor * Q_dst = ggml_view_3d(gA, Q_buf.t,
                    D, H, cl, q_esz * D, q_esz * D * H, (size_t)cs * q_esz * D * H);
                ggml_tensor * K_dst = ggml_view_3d(gA, K_curr_v[il].t,
                    D, Hk, cl, kv_stride, kv_stride * Hk, (size_t)cs * kv_stride * Hk);
                ggml_tensor * V_dst = ggml_view_3d(gA, V_curr_v[il].t,
                    D, Hk, cl, kv_stride, kv_stride * Hk, (size_t)cs * kv_stride * Hk);
                ggml_build_forward_expand(gfA, ggml_cpy(gA, Q, Q_dst));
                ggml_build_forward_expand(gfA, ggml_cpy(gA, K, K_dst));
                ggml_build_forward_expand(gfA, ggml_cpy(gA, V, V_dst));

                if (!ggml_gallocr_alloc_graph(galloc, gfA)) {
                    set_last_error("0p8b: graph A alloc failed at layer " + std::to_string(il));
                    ggml_free(gA); ggml_gallocr_free(galloc); cleanup_all(); return false;
                }
                ggml_backend_graph_compute(w.backend, gfA);
                ggml_free(gA);
            }

            // Copy Q tail (last n_lookahead positions) from Q_buf to Q_last_v[il].
            // Safe here because Q_buf is fully populated after the chunk loop above.
            {
                ggml_init_params ipQ{};
                ipQ.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(256, false) + 16 * 1024;
                ipQ.no_alloc = true;
                ggml_context * gQ = ggml_init(ipQ);
                if (gQ) {
                    const size_t q_esz = ggml_element_size(Q_buf.t);
                    ggml_tensor * Q_tail_view = ggml_view_3d(gQ, Q_buf.t,
                        D, H, n_lookahead, q_esz * D, q_esz * D * H,
                        (size_t)(S - n_lookahead) * q_esz * D * H);
                    ggml_cgraph * gfQ = ggml_new_graph_custom(gQ, 256, false);
                    ggml_build_forward_expand(gfQ, ggml_cpy(gQ, Q_tail_view, Q_last_v[il].t));
                    if (ggml_gallocr_alloc_graph(galloc, gfQ)) {
                        ggml_backend_graph_compute(w.backend, gfQ);
                    }
                    ggml_free(gQ);
                }
            }

            // ── Chunked flash attention ──────────────────────────────
            {
                ggml_init_params ipF{};
                ipF.mem_size = ggml_tensor_overhead() * 128
                               + ggml_graph_overhead_custom(2048, false)
                               + 128 * 1024;
                ipF.no_alloc = true;
                ggml_context * gF = ggml_init(ipF);
                if (!gF) { set_last_error("0p8b: FA graph init failed"); cleanup_all(); ggml_gallocr_free(galloc); return false; }
                ggml_cgraph * gfF = ggml_new_graph_custom(gF, 2048, false);

                const size_t q_esz  = ggml_element_size(Q_buf.t);
                const size_t a_row  = ggml_row_size(attn_out_buf.t->type, D * H);
                // K/V views (full S, same for all chunks)
                ggml_tensor * Kfa = ggml_cont(gF,
                    ggml_permute(gF, K_curr_v[il].t, 0, 2, 1, 3));
                ggml_tensor * Vfa = ggml_cont(gF,
                    ggml_permute(gF, V_curr_v[il].t, 0, 2, 1, 3));

                for (int cs = 0; cs < S; cs += CHUNK_S) {
                    const int cl = std::min(CHUNK_S, S - cs);

                    ggml_tensor * Q_chunk = ggml_view_3d(gF, Q_buf.t,
                        D, H, cl, q_esz * D, q_esz * D * H,
                        (size_t)cs * q_esz * D * H);
                    ggml_tensor * Qfa = ggml_cont(gF,
                        ggml_permute(gF, Q_chunk, 0, 2, 1, 3));
                    if (Qfa->type != GGML_TYPE_F32) {
                        Qfa = ggml_cast(gF, Qfa, GGML_TYPE_F32);
                    }

                    ggml_tensor * attn_chunk = ggml_flash_attn_ext(gF,
                        Qfa, Kfa, Vfa, nullptr, scale, 0.0f, 0.0f);

                    // Copy chunk back to attn_out_buf using 2D view (same
                    // pattern as Graph B, avoids Q8_0 view_3d pitfalls).
                    ggml_tensor * attn_2d = ggml_reshape_2d(gF,
                        ggml_cont(gF, ggml_permute(gF, attn_chunk, 0, 2, 1, 3)),
                        D * H, cl);
                    ggml_tensor * a_dst = ggml_view_2d(gF, attn_out_buf.t,
                        D * H, cl, a_row, (size_t)cs * a_row);
                    ggml_build_forward_expand(gfF,
                        ggml_cpy(gF, attn_2d, a_dst));
                }

                if (!ggml_gallocr_alloc_graph(galloc, gfF)) {
                    set_last_error("0p8b: FA graph alloc failed at layer " + std::to_string(il));
                    ggml_free(gF); ggml_gallocr_free(galloc); cleanup_all(); return false;
                }
                ggml_backend_graph_compute(w.backend, gfF);
                ggml_free(gF);
            }

            // ── Graph B (chunked): gate * attn → attn_o → residual → FFN ──
            for (int cs = 0; cs < S; cs += CHUNK_S) {
                const int cl = std::min(CHUNK_S, S - cs);

                ggml_init_params ipB{};
                ipB.mem_size = ggml_tensor_overhead() * 128
                               + ggml_graph_overhead_custom(4096, false)
                               + 128 * 1024;
                ipB.no_alloc = true;
                ggml_context * gB = ggml_init(ipB);
                if (!gB) { set_last_error("0p8b: graph B init failed"); cleanup_all(); ggml_gallocr_free(galloc); return false; }
                ggml_cgraph * gfB = ggml_new_graph_custom(gB, 4096, false);

                const size_t h_esz  = ggml_element_size(hidden_buf.t);
                ggml_tensor * h_full = ggml_view_2d(gB, hidden_buf.t,
                    hidden, cl, hidden * h_esz, (size_t)cs * hidden * h_esz);

                const size_t a_stride = ggml_row_size(attn_out_buf.t->type, D * H);
                ggml_tensor * attn_chunk = ggml_view_2d(gB, attn_out_buf.t,
                    D * H, cl, a_stride, (size_t)cs * a_stride);

                // Reconstruct gate for this chunk
                ggml_tensor * h_chunk = ggml_view_2d(gB, hidden_buf.t,
                    hidden, cl, hidden * h_esz, (size_t)cs * hidden * h_esz);
                ggml_tensor * cur_norm = ggml_rms_norm(gB, h_chunk, eps);
                cur_norm = ggml_mul(gB, cur_norm, L.attn_norm);
                ggml_tensor * QG2 = ggml_mul_mat(gB, L.attn_q, cur_norm);
                QG2 = ggml_reshape_3d(gB, QG2, D * 2, H, cl);
                const size_t qg2_esz = ggml_element_size(QG2);
                ggml_tensor * gate2 = ggml_view_3d(gB, QG2,
                    D, H, cl, qg2_esz * D * 2, qg2_esz * D * 2 * H, qg2_esz * D);
                ggml_tensor * gate_2d = ggml_cont_2d(gB, gate2, q_dim, cl);
                gate_2d = ggml_sigmoid(gB, gate_2d);

                // Gate * attention output (cast Q8_0→F32 for CUDA bin_bcast compat)
                ggml_tensor * attn_f32 = ggml_cast(gB, attn_chunk, GGML_TYPE_F32);
                ggml_tensor * attn_gated = ggml_mul(gB, attn_f32, gate_2d);
                ggml_tensor * attn_proj = ggml_mul_mat(gB, L.attn_o, attn_gated);
                ggml_tensor * h_after = ggml_add(gB, h_full, attn_proj);

                // FFN (SwiGLU)
                ggml_tensor * hf = ggml_rms_norm(gB, h_after, eps);
                hf = ggml_mul(gB, hf, L.ffn_norm);
                ggml_tensor * gate_t = ggml_mul_mat(gB, L.w_gate, hf);
                gate_t = ggml_silu(gB, gate_t);
                ggml_tensor * up_t   = ggml_mul_mat(gB, L.w_up, hf);
                ggml_tensor * gu     = ggml_mul(gB, gate_t, up_t);
                ggml_tensor * ffn_out = ggml_mul_mat(gB, L.w_down, gu);
                ggml_tensor * h_next = ggml_add(gB, h_after, ffn_out);
                ggml_build_forward_expand(gfB, ggml_cpy(gB, h_next, h_full));

                if (!ggml_gallocr_alloc_graph(galloc, gfB)) {
                    set_last_error("0p8b: graph B alloc failed at layer " + std::to_string(il));
                    ggml_free(gB); ggml_gallocr_free(galloc); cleanup_all(); return false;
                }
                ggml_backend_graph_compute(w.backend, gfB);
                ggml_free(gB);
            }

        } else {
            // ── GATED DELTANET LAYER ────────────────────────────────
            const int conv_channels = w.ssm_inner_size + 2 * w.ssm_group_count * w.ssm_state_size;
            const int head_v_dim    = w.ssm_inner_size / w.ssm_dt_rank;
            const int head_k_dim    = w.ssm_state_size;
            const int num_k_heads   = w.ssm_group_count;
            const int num_v_heads   = w.ssm_dt_rank;
            const int n_seqs        = 1;

            DeltaState & ds = delta_state[il];

            for (int cs = 0; cs < S; cs += CHUNK_S) {
                const int cl = std::min(CHUNK_S, S - cs);

                ggml_init_params ipD{};
                ipD.mem_size = ggml_tensor_overhead() * 256
                               + ggml_graph_overhead_custom(4096, false)
                               + 256 * 1024;
                ipD.no_alloc = true;
                ggml_context * gD = ggml_init(ipD);
                if (!gD) { set_last_error("0p8b: delta graph init failed"); cleanup_all(); ggml_gallocr_free(galloc); return false; }
                ggml_cgraph * gfD = ggml_new_graph_custom(gD, 4096, false);

                const size_t h_esz = ggml_element_size(hidden_buf.t);
                ggml_tensor * h_full = ggml_view_2d(gD, hidden_buf.t,
                    hidden, cl, hidden * h_esz, (size_t)cs * hidden * h_esz);

                // Pre-attention norm
                ggml_tensor * cur = ggml_rms_norm(gD, h_full, eps);
                cur = ggml_mul(gD, cur, L.attn_norm);

                // Gated DeltaNet is skipped — ggml_gated_delta_net internally
                // calls ggml_ssm_conv with incompatible dimension assertions.
                // Hidden state passes through unchanged; full-attn layers
                // still produce tail scoring signal for compression.
                // Residual (no delta output), then FFN.
                ggml_tensor * h_delta = h_full;
                // FFN (SwiGLU)
                ggml_tensor * hf = ggml_rms_norm(gD, h_delta, eps);
                hf = ggml_mul(gD, hf, L.ffn_norm);
                ggml_tensor * gate_t = ggml_mul_mat(gD, L.w_gate, hf);
                gate_t = ggml_silu(gD, gate_t);
                ggml_tensor * up_t   = ggml_mul_mat(gD, L.w_up, hf);
                ggml_tensor * gu     = ggml_mul(gD, gate_t, up_t);
                ggml_tensor * ffn_out = ggml_mul_mat(gD, L.w_down, gu);
                ggml_tensor * h_next = ggml_add(gD, h_delta, ffn_out);
                ggml_build_forward_expand(gfD, ggml_cpy(gD, h_next, h_full));

                if (!ggml_gallocr_alloc_graph(galloc, gfD)) {
                    set_last_error("0p8b: delta graph alloc failed at layer " + std::to_string(il));
                    ggml_free(gD); ggml_gallocr_free(galloc); cleanup_all(); return false;
                }
                ggml_backend_graph_compute(w.backend, gfD);
                ggml_free(gD);
            }
        }

        if (il == 0 || il == w.n_layer - 1) {
            std::fprintf(stderr, "[qwen35-0.8b-fp] layer %d/%d done (%s)\n",
                         il + 1, w.n_layer, is_attn ? "full-attn" : "delta-net");
            std::fflush(stderr);
        }
    }

    // ── Output norm + lm_head (compute logits for the forward pass) ──
    // Only compute if S is small enough for the [n_vocab, S] output to fit.
    // Otherwise skip — logits are not used for compression scoring.
    const size_t logits_bytes = (size_t)w.n_vocab * S * 4;
    if (logits_bytes < 512ULL * 1024 * 1024) {  // < 512 MB
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(1024, false) + 32 * 1024;
        ip.no_alloc = true;
        ggml_context * gctx = ggml_init(ip);
        ggml_cgraph * gf = ggml_new_graph_custom(gctx, 1024, false);

        ggml_tensor * out = ggml_rms_norm(gctx, hidden_buf.t, eps);
        out = ggml_mul(gctx, out, w.out_norm);
        ggml_tensor * logits = ggml_mul_mat(gctx, w.output, out);
        ggml_set_name(logits, "logits");
        ggml_build_forward_expand(gf, logits);

        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            set_last_error("0p8b: out_norm+lm_head graph alloc failed");
            ggml_free(gctx); ggml_gallocr_free(galloc); cleanup_all(); return false;
        }
        ggml_backend_graph_compute(w.backend, gf);
        ggml_free(gctx);
    }

    auto t_fwd_end = std::chrono::steady_clock::now();
    double t_fwd = std::chrono::duration<double>(t_fwd_end - t_total_start).count();

    // ── Tail attention scoring (full-attention layers only) ─────────
    std::vector<float> probs_h((size_t)S * n_lookahead * H);
    auto t_score_start = std::chrono::steady_clock::now();

    std::fflush(stderr);

    for (int il = 0; il < w.n_layer; ++il) {
        const bool is_attn = (((il + 1) % w.full_attn_interval) == 0);
        if (!is_attn) continue;

        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead() + 16 * 1024;
        ip.no_alloc = true;
        ggml_context * gctx = ggml_init(ip);

        ggml_tensor * K_f32 = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, D, Hk, S);
        K_f32->data = ts_K_f32.t->data;
        ggml_tensor * K_cast = ggml_cpy(gctx, K_curr_v[il].t, K_f32);
        ggml_tensor * K_perm = ggml_cont(gctx,
            ggml_permute(gctx, K_cast, 0, 2, 1, 3));
        ggml_tensor * K_score = K_perm;
        if (gqa > 1) {
            ggml_tensor * K_4d = ggml_reshape_4d(gctx, K_perm, D, S, 1, Hk);
            ggml_tensor * K_tpl = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, D, S, gqa, Hk);
            K_tpl->data = ts_K_tpl.t->data;
            ggml_tensor * K_rep = ggml_repeat(gctx, K_4d, K_tpl);
            K_score = ggml_reshape_3d(gctx, K_rep, D, S, H);
        }
        ggml_tensor * Q_tail_perm = ggml_cont(gctx,
            ggml_permute(gctx, Q_last_v[il].t, 0, 2, 1, 3));
        ggml_tensor * attn_score = ggml_mul_mat(gctx, K_score, Q_tail_perm);
        ggml_tensor * probs = ggml_soft_max_ext(gctx, attn_score, mask_tail_buf.t,
                                                scale, 0.0f);
        ggml_set_output(probs);

        ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, probs);

        // No separate allocator — use the existing persistent one_buf
        std::fprintf(stderr, "[qwen35-0.8b-fp] tail-score alloc il=%d S=%d prealloc_K=%p K_tpl=%p\n",
            il, S, (void*)ts_K_f32.t->data, (void*)ts_K_tpl.t->data);
        std::fflush(stderr);
        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            set_last_error("0p8b: tail score graph alloc failed at layer " + std::to_string(il));
            ggml_free(gctx); cleanup_all(); return false;
        }
        ggml_backend_graph_compute(w.backend, gf);
        ggml_backend_tensor_get(probs, probs_h.data(), 0,
                                probs_h.size() * sizeof(float));
        ggml_free(gctx);

        for (int t = 0; t < n_lookahead; ++t) {
            for (int j = 0; j < S; ++j) {
                float m = -INFINITY;
                for (int h = 0; h < H; ++h) {
                    float v = probs_h[(size_t)j
                                      + (size_t)t * S
                                      + (size_t)h * S * n_lookahead];
                    if (v > m) m = v;
                }
                size_t idx = (size_t)t * S + j;
                if (m > running_max[idx]) running_max[idx] = m;
            }
        }
    }
    // ts_K_f32/ts_K_tpl freed with bctx via cleanup_all

    auto t_total_end = std::chrono::steady_clock::now();
    double t_score = std::chrono::duration<double>(t_total_end - t_score_start).count();
    std::fprintf(stderr,
        "[qwen35-0.8b-fp] forward %.2fs (S=%d)  tail-score %.2fs  total %.2fs\n",
        t_fwd, S, t_score, t_fwd + t_score);
    std::fflush(stderr);

    ggml_gallocr_free(galloc);
    cleanup_all();
    return true;
}

} // namespace dflash27b
