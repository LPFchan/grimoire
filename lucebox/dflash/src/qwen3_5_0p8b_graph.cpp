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
    running_max.assign((size_t)n_lookahead * S, -INFINITY);

    // Persistent buffers
    PersBuf hidden_buf, pos_buf, mask_tail_buf;
    PersBuf Q_buf;        // [D, H, S]  f32 (full Q before permute for FA)
    PersBuf attn_out_buf; // [D, H, S]  f32 (attention output before o_proj)

    // K/V buffers: one per layer (DeltaNet layers leave theirs unused)
    std::vector<PersBuf> K_curr_v((size_t)w.n_layer);
    std::vector<PersBuf> V_curr_v((size_t)w.n_layer);
    std::vector<PersBuf> Q_last_v((size_t)w.n_layer);

    // DeltaNet recurrent state per layer (used only for delta layers)
    struct DeltaState {
        PersBuf conv; // [(kernel-1), conv_channels] f32
        PersBuf ssm;  // [head_v_dim, head_v_dim, num_v_heads] f32
    };
    std::vector<DeltaState> delta_state((size_t)w.n_layer);

    auto cleanup_all = [&]() {
        free_pers(hidden_buf); free_pers(pos_buf); free_pers(mask_tail_buf);
        free_pers(Q_buf); free_pers(attn_out_buf);
        for (auto & p : K_curr_v) free_pers(p);
        for (auto & p : V_curr_v) free_pers(p);
        for (auto & p : Q_last_v) free_pers(p);
        for (auto & ds : delta_state) { free_pers(ds.conv); free_pers(ds.ssm); }
    };

    // Allocate persistent buffers
    {
        int64_t d_h[]  = {(int64_t)hidden, (int64_t)S};
        int64_t d_kv[] = {(int64_t)D, (int64_t)Hk, (int64_t)S};
        int64_t d_q[]  = {(int64_t)D, (int64_t)H,  (int64_t)S};
        int64_t d_ql[] = {(int64_t)D, (int64_t)H,  (int64_t)n_lookahead};
        int64_t d_p[]  = {(int64_t)S};
        int64_t d_mt[] = {(int64_t)S, (int64_t)n_lookahead};

        if (!make_pers(w.backend, GGML_TYPE_F32, 2, d_h, hidden_buf) ||
            !make_pers(w.backend, GGML_TYPE_I32, 1, d_p, pos_buf)     ||
            !make_pers(w.backend, GGML_TYPE_F32, 2, d_mt, mask_tail_buf) ||
            !make_pers(w.backend, GGML_TYPE_F32, 3, d_q, Q_buf)       ||
            !make_pers(w.backend, GGML_TYPE_F32, 3, d_q, attn_out_buf))
        {
            set_last_error("0p8b: persistent alloc failed (hidden/pos/mask/Q/attn_out)");
            cleanup_all(); return false;
        }
        for (int il = 0; il < w.n_layer; ++il) {
            if (!make_pers(w.backend, GGML_TYPE_F32, 3, d_kv, K_curr_v[il]) ||
                !make_pers(w.backend, GGML_TYPE_F32, 3, d_kv, V_curr_v[il]) ||
                !make_pers(w.backend, GGML_TYPE_F32, 3, d_ql, Q_last_v[il]))
            {
                set_last_error("0p8b: K_curr/V_curr/Q_last alloc failed at layer " + std::to_string(il));
                cleanup_all(); return false;
            }
            // DeltaNet state buffers
            const bool is_attn = (((il + 1) % w.full_attn_interval) == 0);
            if (!is_attn) {
                const int conv_channels = w.ssm_inner_size + 2 * w.ssm_group_count * w.ssm_state_size;
                const int head_v_dim    = w.ssm_inner_size / w.ssm_dt_rank;
                int64_t d_conv[] = {(int64_t)(w.ssm_conv_kernel - 1), (int64_t)conv_channels};
                int64_t d_ssm[]  = {(int64_t)head_v_dim, (int64_t)head_v_dim, (int64_t)w.ssm_dt_rank};
                if (!make_pers(w.backend, GGML_TYPE_F32, 2, d_conv, delta_state[il].conv) ||
                    !make_pers(w.backend, GGML_TYPE_F32, 3, d_ssm,  delta_state[il].ssm))
                {
                    set_last_error("0p8b: delta state alloc failed at layer " + std::to_string(il));
                    cleanup_all(); return false;
                }
                // Zero-initialize recurrent state
                {
                    size_t nb_c = ggml_nbytes(delta_state[il].conv.t);
                    std::vector<uint8_t> z_c(nb_c, 0);
                    ggml_backend_tensor_set(delta_state[il].conv.t, z_c.data(), 0, nb_c);
                    size_t nb_s = ggml_nbytes(delta_state[il].ssm.t);
                    std::vector<uint8_t> z_s(nb_s, 0);
                    ggml_backend_tensor_set(delta_state[il].ssm.t, z_s.data(), 0, nb_s);
                }
            }
        }
    }

    // Positions [0..S-1]
    {
        std::vector<int32_t> pos((size_t)S);
        for (int i = 0; i < S; ++i) pos[i] = i;
        ggml_backend_tensor_set(pos_buf.t, pos.data(), 0, (size_t)S * sizeof(int32_t));
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
                ggml_tensor * pos_chunk = ggml_view_1d(gA, pos_buf.t, cl,
                    (size_t)cs * sizeof(int32_t));

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
                const size_t kv_esz = ggml_element_size(K_curr_v[il].t);
                ggml_tensor * Q_dst = ggml_view_3d(gA, Q_buf.t,
                    D, H, cl, q_esz * D, q_esz * D * H, (size_t)cs * q_esz * D * H);
                ggml_tensor * K_dst = ggml_view_3d(gA, K_curr_v[il].t,
                    D, Hk, cl, kv_esz * D, kv_esz * D * Hk, (size_t)cs * kv_esz * D * Hk);
                ggml_tensor * V_dst = ggml_view_3d(gA, V_curr_v[il].t,
                    D, Hk, cl, kv_esz * D, kv_esz * D * Hk, (size_t)cs * kv_esz * D * Hk);
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

            // ── Flash attention over full S ─────────────────────────
            {
                ggml_init_params ipF{};
                ipF.mem_size = ggml_tensor_overhead() * 64
                               + ggml_graph_overhead_custom(1024, false)
                               + 64 * 1024;
                ipF.no_alloc = true;
                ggml_context * gF = ggml_init(ipF);
                if (!gF) { set_last_error("0p8b: FA graph init failed"); cleanup_all(); ggml_gallocr_free(galloc); return false; }
                ggml_cgraph * gfF = ggml_new_graph_custom(gF, 1024, false);

                // ggml_flash_attn_ext expects Q: [D, S, H], K: [D, S, Hk], V: [D, S, Hk]
                ggml_tensor * Qfa = ggml_permute(gF, Q_buf.t, 0, 2, 1, 3);
                Qfa = ggml_cont(gF, Qfa);
                ggml_tensor * Kfa = ggml_permute(gF, K_curr_v[il].t, 0, 2, 1, 3);
                Kfa = ggml_cont(gF, Kfa);
                ggml_tensor * Vfa = ggml_permute(gF, V_curr_v[il].t, 0, 2, 1, 3);
                Vfa = ggml_cont(gF, Vfa);

                // Flash attention over full S. No mask (non-causal) is acceptable
                // for the drafter forward path — the tail scoring graph below
                // applies proper causal masking for the actual compression signal.
                ggml_tensor * attn_raw = ggml_flash_attn_ext(gF, Qfa, Kfa, Vfa,
                    nullptr, scale, 0.0f, 0.0f);
                // attn_raw: [D, S, H] (permuted layout matching Qfa)

                // Copy back to attn_out_buf: need to re-permute to [D, H, S]
                ggml_tensor * attn_back = ggml_permute(gF, attn_raw, 0, 2, 1, 3);
                ggml_build_forward_expand(gfF, ggml_cpy(gF, attn_back, attn_out_buf.t));

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

                const size_t a_esz = ggml_element_size(attn_out_buf.t);
                ggml_tensor * attn_chunk = ggml_view_2d(gB, attn_out_buf.t,
                    D * H, cl, a_esz * D * H, (size_t)cs * a_esz * D * H);

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

                // Gate * attention output
                ggml_tensor * attn_gated = ggml_mul(gB, attn_chunk, gate_2d);
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

                // Fused QKV projection
                ggml_tensor * qkv_mixed = ggml_mul_mat(gD, L.wqkv, cur);
                qkv_mixed = ggml_reshape_3d(gD, qkv_mixed, conv_channels, cl, n_seqs);

                // z projection (gate)
                ggml_tensor * z = ggml_mul_mat(gD, L.wqkv_gate, cur);

                // beta = sigmoid(ssm_beta @ cur)
                ggml_tensor * beta = ggml_mul_mat(gD, L.ssm_beta, cur);
                beta = ggml_reshape_4d(gD, beta, 1, num_v_heads, cl, n_seqs);
                beta = ggml_sigmoid(gD, beta);

                // alpha = softplus(ssm_alpha @ cur + ssm_dt_bias)
                ggml_tensor * alpha = ggml_mul_mat(gD, L.ssm_alpha, cur);
                alpha = ggml_reshape_3d(gD, alpha, num_v_heads, cl, n_seqs);
                alpha = ggml_add(gD, alpha, L.ssm_dt_bias);
                alpha = ggml_softplus(gD, alpha);
                ggml_tensor * g_tensor = ggml_mul(gD, alpha, L.ssm_a);
                g_tensor = ggml_reshape_4d(gD, g_tensor, 1, num_v_heads, cl, n_seqs);

                // Convolution: prepend conv state to qkv_mixed
                ggml_tensor * conv_states_r = ggml_reshape_3d(gD, ds.conv.t,
                    w.ssm_conv_kernel - 1, conv_channels, n_seqs);
                ggml_tensor * qkv_T = ggml_transpose(gD, qkv_mixed);
                ggml_tensor * conv_input = ggml_concat(gD, conv_states_r, qkv_T, 0);

                // Save last (kernel-1) steps back to conv_state
                ggml_tensor * last_conv = ggml_view_3d(gD, conv_input,
                    w.ssm_conv_kernel - 1, conv_channels, n_seqs,
                    conv_input->nb[1], conv_input->nb[2],
                    (conv_input->ne[0] - (w.ssm_conv_kernel - 1)) * ggml_element_size(conv_input));
                ggml_build_forward_expand(gfD, ggml_cpy(gD, last_conv, ds.conv.t));

                // 1D conv + silu
                ggml_tensor * conv_out = ggml_ssm_conv(gD, conv_input, L.ssm_conv1d);
                conv_out = ggml_silu(gD, conv_out);

                // Split conv_out into Q, K, V
                const size_t elt = ggml_element_size(conv_out);
                const size_t row_size = (size_t)conv_channels * elt;

                const int64_t q_offset = 0;
                const int64_t k_offset = num_k_heads * head_k_dim;
                const int64_t v_offset = 2 * num_k_heads * head_k_dim;

                ggml_tensor * q_c = ggml_view_4d(gD, conv_out,
                    head_k_dim, num_k_heads, cl, n_seqs,
                    head_k_dim * elt, row_size, row_size * cl, q_offset * elt);
                ggml_tensor * k_c = ggml_view_4d(gD, conv_out,
                    head_k_dim, num_k_heads, cl, n_seqs,
                    head_k_dim * elt, row_size, row_size * cl, k_offset * elt);
                ggml_tensor * v_c = ggml_view_4d(gD, conv_out,
                    head_v_dim, num_v_heads, cl, n_seqs,
                    head_v_dim * elt, row_size, row_size * cl, v_offset * elt);

                // L2 norm on Q and K
                q_c = ggml_l2_norm(gD, q_c, eps);
                k_c = ggml_l2_norm(gD, k_c, eps);

                // Repeat Q/K from num_k_heads to num_v_heads
                if (num_k_heads != num_v_heads) {
                    q_c = ggml_repeat_4d(gD, q_c, head_k_dim, num_v_heads, cl, n_seqs);
                    k_c = ggml_repeat_4d(gD, k_c, head_k_dim, num_v_heads, cl, n_seqs);
                }

                // SSM state
                ggml_tensor * s = ggml_reshape_4d(gD, ds.ssm.t,
                    head_v_dim, head_v_dim, num_v_heads, n_seqs);

                // Fused Gated DeltaNet
                ggml_tensor * result = ggml_gated_delta_net(gD, q_c, k_c, v_c, g_tensor, beta, s);

                // Slice output and new_state
                const size_t r_elt = ggml_element_size(result);
                ggml_tensor * output = ggml_view_4d(gD, result,
                    head_v_dim, num_v_heads, cl, n_seqs,
                    head_v_dim * r_elt,
                    head_v_dim * num_v_heads * r_elt,
                    head_v_dim * num_v_heads * cl * r_elt,
                    0);
                ggml_tensor * new_state = ggml_view_4d(gD, result,
                    head_v_dim, head_v_dim, num_v_heads, n_seqs,
                    head_v_dim * r_elt,
                    head_v_dim * head_v_dim * r_elt,
                    head_v_dim * head_v_dim * num_v_heads * r_elt,
                    head_v_dim * num_v_heads * cl * n_seqs * r_elt);

                // Persist new state
                ggml_build_forward_expand(gfD, ggml_cpy(gD, new_state, ds.ssm.t));

                // Gated output norm: rms_norm(output) * silu(z)
                ggml_tensor * z_4d = ggml_reshape_4d(gD, z, head_v_dim, num_v_heads, cl, n_seqs);
                ggml_tensor * output_n = ggml_rms_norm(gD, output, eps);
                output_n = ggml_mul(gD, output_n, L.ssm_norm);
                ggml_tensor * z_silu = ggml_silu(gD, z_4d);
                output_n = ggml_mul(gD, output_n, z_silu);

                // Reshape to [d_inner, cl]
                ggml_tensor * flat = ggml_reshape_3d(gD, output_n,
                    head_v_dim * num_v_heads, cl, n_seqs);
                ggml_tensor * delta_out = ggml_mul_mat(gD, L.ssm_out, flat);
                delta_out = ggml_reshape_2d(gD, delta_out, hidden, cl);

                // Residual
                ggml_tensor * h_delta = ggml_add(gD, h_full, delta_out);

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
    {
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

    ggml_gallocr_free(galloc);

    auto t_fwd_end = std::chrono::steady_clock::now();
    double t_fwd = std::chrono::duration<double>(t_fwd_end - t_total_start).count();

    // ── Tail attention scoring (full-attention layers only) ─────────
    std::vector<float> probs_h((size_t)S * n_lookahead * H);
    auto t_score_start = std::chrono::steady_clock::now();

    for (int il = 0; il < w.n_layer; ++il) {
        const bool is_attn = (((il + 1) % w.full_attn_interval) == 0);
        if (!is_attn) continue;

        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead() + 16 * 1024;
        ip.no_alloc = true;
        ggml_context * gctx = ggml_init(ip);

        ggml_tensor * K_f32 = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, D, Hk, S);
        ggml_tensor * K_cast = ggml_cpy(gctx, K_curr_v[il].t, K_f32);
        ggml_tensor * K_perm = ggml_cont(gctx,
            ggml_permute(gctx, K_cast, 0, 2, 1, 3));
        ggml_tensor * K_score = K_perm;
        if (gqa > 1) {
            ggml_tensor * K_4d = ggml_reshape_4d(gctx, K_perm, D, S, 1, Hk);
            ggml_tensor * K_tpl = ggml_new_tensor_4d(gctx, GGML_TYPE_F32,
                                                     D, S, gqa, Hk);
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

        ggml_backend_buffer_t in_buf = ggml_backend_alloc_ctx_tensors(gctx, w.backend);
        ggml_gallocr_t s_galloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(w.backend));
        if (!ggml_gallocr_alloc_graph(s_galloc, gf)) {
            set_last_error("0p8b: tail score graph alloc failed at layer " + std::to_string(il));
            ggml_gallocr_free(s_galloc);
            if (in_buf) ggml_backend_buffer_free(in_buf);
            ggml_free(gctx); cleanup_all(); return false;
        }
        ggml_backend_graph_compute(w.backend, gf);
        ggml_backend_tensor_get(probs, probs_h.data(), 0,
                                probs_h.size() * sizeof(float));
        ggml_gallocr_free(s_galloc);
        if (in_buf) ggml_backend_buffer_free(in_buf);
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

    auto t_total_end = std::chrono::steady_clock::now();
    double t_score = std::chrono::duration<double>(t_total_end - t_score_start).count();
    std::fprintf(stderr,
        "[qwen35-0.8b-fp] forward %.2fs (S=%d)  tail-score %.2fs  total %.2fs\n",
        t_fwd, S, t_score, t_fwd + t_score);
    std::fflush(stderr);

    cleanup_all();
    return true;
}

} // namespace dflash27b
