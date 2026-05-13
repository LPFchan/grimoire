// decode_target_split.cpp — target layer-split harness functions extracted
// from test_dflash.cpp. All functions are inside namespace dflash27b.

#include "decode_context.h"
#include "internal.h"
#include "dflash_graph.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cinttypes>
#include <fstream>
#include <string>
#include <vector>

namespace dflash27b {

// All helper declarations come from decode_context.h (included above).
// No need for redundant extern declarations here.

// Define KQ_MASK_PAD locally since decode_helpers.cpp defines the globals.
// We only need the constant for align_up calls and the g_kq_stride_pad / KQ_MASK_PAD
// comparison in run_target_layer_split_forward.
static constexpr int KQ_MASK_PAD = 32;

// ── Target layer-split helpers ─────────────────────────────────────

bool parse_int_list(const char * text, std::vector<int> & out) {
    out.clear();
    if (!text || !*text) return false;
    const char * p = text;
    while (*p) {
        char * end = nullptr;
        long v = std::strtol(p, &end, 10);
        if (end == p || v < 0 || v > INT32_MAX) return false;
        out.push_back((int)v);
        if (*end == '\0') break;
        if (*end != ',') return false;
        p = end + 1;
    }
    return !out.empty();
}

bool parse_float_list(const char * text, std::vector<double> & out) {
    out.clear();
    if (!text || !*text) return false;
    const char * p = text;
    while (*p) {
        char * end = nullptr;
        double v = std::strtod(p, &end);
        if (end == p || v <= 0.0) return false;
        out.push_back(v);
        if (*end == '\0') break;
        if (*end != ',') return false;
        p = end + 1;
    }
    return !out.empty();
}

int inspect_target_layer_count(const char * target_path) {
    ggml_context * meta_ctx = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx = &meta_ctx;
    gguf_context * gctx = gguf_init_from_file(target_path, gip);
    if (!gctx) return -1;
    int64_t id = gguf_find_key(gctx, "qwen35.block_count");
    int n_layer = id >= 0 ? (int)gguf_get_val_u32(gctx, id) : -1;
    gguf_free(gctx);
    if (meta_ctx) ggml_free(meta_ctx);
    return n_layer;
}

std::vector<std::pair<int, int>> compute_layer_ranges(
        int n_layer,
        int n_shards,
        const std::vector<double> & weights) {
    std::vector<std::pair<int, int>> ranges;
    if (n_layer <= 0 || n_shards <= 0 || n_shards > n_layer) return ranges;
    std::vector<double> w = weights;
    if (w.empty()) w.assign((size_t)n_shards, 1.0);
    if ((int)w.size() != n_shards) return ranges;
    double sum = 0.0;
    for (double v : w) sum += v;
    if (sum <= 0.0) return ranges;
    ranges.reserve((size_t)n_shards);
    int begin = 0;
    double accum = 0.0;
    for (int i = 0; i < n_shards; i++) {
        accum += w[i];
        int end = (i == n_shards - 1)
            ? n_layer
            : (int)std::llround((accum / sum) * n_layer);
        const int min_end = begin + 1;
        const int max_end = n_layer - (n_shards - i - 1);
        end = std::max(min_end, std::min(max_end, end));
        ranges.push_back({begin, end});
        begin = end;
    }
    return ranges;
}

TargetLayerSplitShard * find_target_shard(
        std::vector<TargetLayerSplitShard> & shards,
        int layer_idx) {
    for (auto & shard : shards) {
        if (layer_idx >= shard.layer_begin && layer_idx < shard.layer_end) {
            return &shard;
        }
    }
    return nullptr;
}

int target_capture_index(const TargetWeights & w, int layer_idx) {
    for (int k = 0; k < DFLASH27B_DRAFT_N_TARGET_LAYERS; k++) {
        if (w.capture_layer_ids[k] == layer_idx) return k;
    }
    return -1;
}

bool copy_capture_slice_to_draft_ring(
        DraftFeatureMirror & feature_ring,
        int capture_idx,
        const ggml_tensor * act_out,
        int src_device,
        int chunk_start,
        int start_pos,
        int n_tokens) {
    if (!feature_ring.target_feat || capture_idx < 0 || n_tokens <= 0) return true;
    if (feature_ring.cap <= 0) return false;
    const int hidden = DFLASH27B_TARGET_HIDDEN;
    const size_t dst_stride = feature_ring.target_feat->nb[1];
    const size_t src_stride = act_out->nb[1];
    const size_t row_bytes = (size_t)hidden * sizeof(float);
    for (int i = 0; i < n_tokens; i++) {
        const int slot = (start_pos + i) % feature_ring.cap;
        const void * src = (const char *)act_out->data +
            (size_t)(chunk_start + i) * src_stride;
        void * dst = (char *)feature_ring.target_feat->data +
            (size_t)slot * dst_stride +
            (size_t)capture_idx * (size_t)hidden * sizeof(float);
        if (!copy_peer_async(dst, feature_ring.device, const_cast<void *>(src), src_device, row_bytes)) {
            return false;
        }
    }
    return cudaDeviceSynchronize() == cudaSuccess;
}

bool copy_feature_ring_range_to_tensor(
        const DraftFeatureMirror & feature_ring,
        ggml_tensor * dst,
        int start_pos,
        int n_tokens) {
    if (!feature_ring.target_feat || !dst || feature_ring.cap <= 0) return false;
    if (n_tokens <= 0 || n_tokens > feature_ring.cap) return false;

    const int fc_in = DFLASH27B_DRAFT_N_TARGET_LAYERS * DFLASH27B_TARGET_HIDDEN;
    const size_t row_bytes = (size_t)fc_in * sizeof(float);
    const size_t src_stride = feature_ring.target_feat->nb[1];
    const size_t dst_stride = dst->nb[1];
    int done = 0;
    while (done < n_tokens) {
        const int slot = (start_pos + done) % feature_ring.cap;
        const int run = std::min(n_tokens - done, feature_ring.cap - slot);
        const char * src_base =
            (const char *)feature_ring.target_feat->data + (size_t)slot * src_stride;
        char * dst_base = (char *)dst->data + (size_t)done * dst_stride;
        if (src_stride == row_bytes && dst_stride == row_bytes) {
            if (!copy_peer_async(dst_base, feature_ring.device,
                                 const_cast<char *>(src_base), feature_ring.device,
                                 row_bytes * (size_t)run)) {
                return false;
            }
        } else {
            for (int i = 0; i < run; i++) {
                if (!copy_peer_async(dst_base + (size_t)i * dst_stride,
                                     feature_ring.device,
                                     const_cast<char *>(src_base + (size_t)i * src_stride),
                                     feature_ring.device,
                                     row_bytes)) {
                    return false;
                }
            }
        }
        done += run;
    }
    return cudaDeviceSynchronize() == cudaSuccess;
}

bool compute_target_split_argmax(
        StepGraph & sg,
        const TargetWeights & w,
        ggml_backend_t backend,
        ggml_tensor * act,
        int token_offset,
        int n_tokens,
        int hidden,
        int vocab,
        std::vector<int32_t> & argmax_out) {
    step_graph_free(sg);
    ggml_init_params ip{};
    ip.mem_size = 256 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    ggml_tensor * act_view = ggml_view_2d(
        sg.ctx, act, hidden, n_tokens, act->nb[1],
        (size_t)token_offset * act->nb[1]);
    ggml_tensor * normed = ggml_rms_norm(sg.ctx, act_view, DFLASH27B_RMS_EPS);
    normed = ggml_mul(sg.ctx, normed, w.out_norm);
    ggml_tensor * logits = ggml_mul_mat(sg.ctx, w.output, normed);
    ggml_set_name(logits, "target_split_logits");
    sg.logits = logits;
    sg.argmax_tokens = ggml_argmax(sg.ctx, logits);
    ggml_set_name(sg.argmax_tokens, "target_split_argmax");
    ggml_set_output(sg.argmax_tokens);
    sg.gf = ggml_new_graph_custom(sg.ctx, 1024, false);
    ggml_build_forward_expand(sg.gf, sg.argmax_tokens);
    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    if (!ggml_gallocr_alloc_graph(sg.alloc, sg.gf)) return false;
    auto st = ggml_backend_graph_compute(backend, sg.gf);
    if (st != GGML_STATUS_SUCCESS) return false;
    (void)vocab;
    argmax_out.assign((size_t)n_tokens, 0);
    ggml_backend_tensor_get(sg.argmax_tokens, argmax_out.data(), 0,
                            sizeof(int32_t) * (size_t)n_tokens);
    return true;
}

bool run_target_layer_split_forward(
        std::vector<TargetLayerSplitShard> & shards,
        const TargetWeights & embed_source,
        const std::vector<int32_t> & tokens,
        int base_pos,
        int ubatch,
        int & last_tok,
        DraftFeatureMirror * feature_ring = nullptr,
        std::vector<int32_t> * argmax_out = nullptr,
        std::vector<float> * logits_out = nullptr) {
    if (shards.empty() || tokens.empty()) return false;
    const int hidden = DFLASH27B_TARGET_HIDDEN;
    const int vocab = DFLASH27B_TARGET_VOCAB;
    const int n_tokens_total = (int)tokens.size();
    ubatch = std::max(1, ubatch);

    ActivationPair acts;
    if (!activation_pair_init(acts, shards.front().backend, hidden, n_tokens_total)) {
        std::fprintf(stderr, "target-split activation alloc failed on gpu %d\n", shards.front().gpu);
        return false;
    }
    ggml_tensor * act_in = acts.a;
    ggml_tensor * act_out = acts.b;

    {
        const int EMBED_BATCH = 4096;
        std::vector<float> emb_buf((size_t)hidden * std::min(EMBED_BATCH, n_tokens_total));
        for (int i = 0; i < n_tokens_total; i += EMBED_BATCH) {
            const int n = std::min(EMBED_BATCH, n_tokens_total - i);
            if ((int)emb_buf.size() < hidden * n) emb_buf.resize((size_t)hidden * n);
            if (!embed_source.embedder.embed(tokens.data() + i, n, emb_buf.data())) {
                activation_pair_free(acts);
                return false;
            }
            ggml_backend_tensor_set(act_in, emb_buf.data(),
                                    (size_t)i * act_in->nb[1],
                                    sizeof(float) * (size_t)hidden * n);
        }
    }

    TargetLayerSplitShard * current_shard = &shards.front();
    std::vector<uint16_t> mask_buf;
    std::vector<int32_t> pos_buf;
    for (int il = 0; il < embed_source.n_layer; il++) {
        TargetLayerSplitShard * shard = find_target_shard(shards, il);
        if (!shard) {
            std::fprintf(stderr, "target-split missing owner for layer %d\n", il);
            activation_pair_free(acts);
            return false;
        }
        if (shard != current_shard) {
            ActivationPair next_acts;
            if (!activation_pair_init(next_acts, shard->backend, hidden, n_tokens_total)) {
                std::fprintf(stderr, "target-split activation alloc failed on gpu %d\n", shard->gpu);
                activation_pair_free(acts);
                return false;
            }
            ggml_backend_synchronize(current_shard->backend);
            ggml_backend_tensor_copy(act_in, next_acts.a);
            ggml_backend_synchronize(shard->backend);
            activation_pair_free(acts);
            acts = next_acts;
            act_in = acts.a;
            act_out = acts.b;
            current_shard = shard;
        }

        const bool is_attn = (((il + 1) % embed_source.full_attention_interval) == 0);
        const int capture_idx = target_capture_index(embed_source, il);
        for (int start = 0; start < n_tokens_total; start += ubatch) {
            const int n = std::min(ubatch, n_tokens_total - start);
            const int kv_start = base_pos + start;
            const int kv_len = kv_start + n;
            const bool with_mask = (g_kq_stride_pad > KQ_MASK_PAD) || (n > 1);
            if (!build_layer_step(shard->layer_graph, shard->weights, shard->cache,
                                  shard->backend, il, act_in, act_out,
                                  start, n, kv_start, with_mask,
                                  /*capture=*/false, g_fa_window)) {
                std::fprintf(stderr, "target-split build layer=%d @%d gpu=%d\n",
                             il, start, shard->gpu);
                activation_pair_free(acts);
                return false;
            }
            if (is_attn && shard->layer_graph.positions) {
                pos_buf.assign((size_t)4 * n, 0);
                for (int i = 0; i < n; i++) {
                    const int p = kv_start + i;
                    pos_buf[0 * n + i] = p;
                    pos_buf[1 * n + i] = p;
                    pos_buf[2 * n + i] = p;
                    pos_buf[3 * n + i] = 0;
                }
                ggml_backend_tensor_set(shard->layer_graph.positions, pos_buf.data(), 0,
                                        sizeof(int32_t) * pos_buf.size());
            }
            if (is_attn && with_mask && shard->layer_graph.attn_mask) {
                const int win_start_l = (g_fa_window > 0 && kv_start > g_fa_window)
                                            ? (kv_start - g_fa_window) : 0;
                const int win_len_l = kv_len - win_start_l;
                build_causal_mask(mask_buf, win_len_l, n, kv_start, win_start_l);
                ggml_backend_tensor_set(shard->layer_graph.attn_mask, mask_buf.data(), 0,
                                        sizeof(uint16_t) * mask_buf.size());
            }
            auto st = ggml_backend_graph_compute(shard->backend, shard->layer_graph.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "target-split compute layer=%d @%d gpu=%d status=%d\n",
                             il, start, shard->gpu, (int)st);
                activation_pair_free(acts);
                return false;
            }
            if (feature_ring && capture_idx >= 0) {
                if (!copy_capture_slice_to_draft_ring(*feature_ring, capture_idx,
                                                      act_out, shard->gpu,
                                                      start, base_pos + start, n)) {
                    std::fprintf(stderr,
                                 "target-split capture copy failed layer=%d capture=%d gpu=%d\n",
                                 il, capture_idx, shard->gpu);
                    activation_pair_free(acts);
                    return false;
                }
            }
        }
        std::swap(act_in, act_out);
    }

    StepGraph final_sg;
    std::vector<int32_t> argmax_tokens;
    TargetLayerSplitShard & last_shard = shards.back();
    const bool need_all_argmax = argmax_out != nullptr;
    const int argmax_offset = need_all_argmax ? 0 : (n_tokens_total - 1);
    const int argmax_count = need_all_argmax ? n_tokens_total : 1;
    const bool ok = compute_target_split_argmax(
        final_sg, last_shard.weights, last_shard.backend, act_in,
        argmax_offset, argmax_count, hidden, vocab, argmax_tokens);
    step_graph_destroy(final_sg);
    activation_pair_free(acts);
    if (!ok) return false;
    last_tok = argmax_tokens.empty() ? -1 : argmax_tokens.back();
    if (argmax_out) *argmax_out = std::move(argmax_tokens);
    if (logits_out) logits_out->clear();
    return true;
}

void free_target_layer_split_shards(std::vector<TargetLayerSplitShard> & shards) {
    for (auto & shard : shards) {
        step_graph_destroy(shard.layer_graph);
        free_target_cache(shard.cache);
        free_target_weights(shard.weights);
        if (shard.backend) {
            ggml_backend_free(shard.backend);
            shard.backend = nullptr;
        }
    }
    shards.clear();
}

bool run_target_layer_split_dflash_decode(
        std::vector<TargetLayerSplitShard> & shards,
        DraftWeights & draft_weights,
        ggml_backend_t draft_backend,
        int draft_gpu,
        DraftFeatureMirror & feature_ring,
        const std::vector<int32_t> & prompt,
        int n_gen,
        int last_tok,
        const char * out_path) {
    if (shards.empty() || !feature_ring.target_feat) return false;
    const int hidden = DFLASH27B_TARGET_HIDDEN;
    const int vocab = DFLASH27B_TARGET_VOCAB;
    const int q_len = DFLASH27B_DRAFT_BLOCK_SIZE;
    const int output_gpu = shards.back().gpu;
    ggml_backend_t output_backend = shards.back().backend;

    StepGraph draft_sg;
    StepGraph proj_sg;
    std::vector<float> noise_embed((size_t)hidden * q_len);
    std::vector<int32_t> noise_ids(q_len);
    std::vector<int32_t> draft_tok(q_len);
    std::vector<int32_t> target_tok(q_len);
    std::vector<int32_t> pos_q(q_len);
    std::vector<int32_t> pos_k;
    std::vector<int32_t> out_all = prompt;
    int committed = (int)prompt.size();
    int n_generated = 0;
    int n_draft_steps = 0;
    int n_accept_sum = 0;

    auto sync_all = [&]() {
        for (auto & shard : shards) ggml_backend_synchronize(shard.backend);
        ggml_backend_synchronize(draft_backend);
    };

    auto t_dec0 = std::chrono::steady_clock::now();
    while (n_generated < n_gen) {
        const int need_commit_budget = n_gen - n_generated;

        noise_ids[0] = last_tok;
        for (int i = 1; i < q_len; i++) noise_ids[i] = DFLASH27B_DRAFT_MASK_TOKEN_ID;
        if (!shards.front().weights.embedder.embed(noise_ids.data(), q_len,
                                                    noise_embed.data())) {
            std::fprintf(stderr, "target-split-dflash noise embed failed\n");
            step_graph_destroy(draft_sg);
            step_graph_destroy(proj_sg);
            return false;
        }

        constexpr int DRAFT_CTX_MAX = 2048;
        const int draft_ctx = std::min(committed, std::min(feature_ring.cap,
            std::max(DRAFT_CTX_MAX, g_draft_ctx_max)));
        const int draft_start = committed - draft_ctx;
        int mirror_slot0 = 0;
        const bool use_mirror_view =
            draft_feature_mirror_can_view(feature_ring, committed, draft_ctx, mirror_slot0);
        if (!build_draft_step(draft_sg, draft_weights, nullptr, draft_backend,
                              draft_ctx, use_mirror_view ? &feature_ring : nullptr,
                              committed)) {
            std::fprintf(stderr, "target-split-dflash draft build failed\n");
            step_graph_destroy(draft_sg);
            step_graph_destroy(proj_sg);
            return false;
        }
        if (!use_mirror_view &&
            !copy_feature_ring_range_to_tensor(feature_ring, draft_sg.target_hidden_cat,
                                                draft_start, draft_ctx)) {
            std::fprintf(stderr, "target-split-dflash draft feature copy failed\n");
            step_graph_destroy(draft_sg);
            step_graph_destroy(proj_sg);
            return false;
        }
        ggml_backend_tensor_set(draft_sg.inp_embed, noise_embed.data(), 0,
                                sizeof(float) * noise_embed.size());
        pos_k.resize((size_t)draft_ctx + q_len);
        for (int i = 0; i < q_len; i++) pos_q[i] = draft_ctx + i;
        for (int i = 0; i < draft_ctx + q_len; i++) pos_k[i] = i;
        ggml_backend_tensor_set(draft_sg.positions, pos_q.data(), 0,
                                sizeof(int32_t) * pos_q.size());
        ggml_backend_tensor_set(draft_sg.positions_k, pos_k.data(), 0,
                                sizeof(int32_t) * pos_k.size());
        auto st = ggml_backend_graph_compute(draft_backend, draft_sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "target-split-dflash draft compute %d\n", (int)st);
            step_graph_destroy(draft_sg);
            step_graph_destroy(proj_sg);
            return false;
        }

        if (!proj_sg.gf || !proj_sg.hidden_input || proj_sg.hidden_input->ne[1] != q_len) {
            if (!build_lm_head_projection_step(proj_sg, shards.back().weights,
                                                output_backend, q_len)) {
                std::fprintf(stderr, "target-split-dflash projection build failed\n");
                step_graph_destroy(draft_sg);
                step_graph_destroy(proj_sg);
                return false;
            }
        }
        const size_t hidden_bytes = ggml_nbytes(draft_sg.hidden_states);
        if (!copy_peer_async(proj_sg.hidden_input->data, output_gpu,
                             draft_sg.hidden_states->data, draft_gpu,
                             hidden_bytes)) {
            std::fprintf(stderr, "target-split-dflash hidden peer copy failed\n");
            step_graph_destroy(draft_sg);
            step_graph_destroy(proj_sg);
            return false;
        }
        cudaSetDevice(output_gpu);
        cudaDeviceSynchronize();
        st = ggml_backend_graph_compute(output_backend, proj_sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "target-split-dflash projection compute %d\n", (int)st);
            step_graph_destroy(draft_sg);
            step_graph_destroy(proj_sg);
            return false;
        }
        ggml_backend_tensor_get(proj_sg.argmax_tokens, draft_tok.data(), 0,
                                sizeof(int32_t) * q_len);
        draft_tok[0] = last_tok;

        for (auto & shard : shards) snapshot_ssm_state(shard.cache);

        int verify_last_tok = -1;
        if (!run_target_layer_split_forward(shards, shards.front().weights,
                                            draft_tok, committed, q_len,
                                            verify_last_tok, &feature_ring,
                                            &target_tok)) {
            std::fprintf(stderr, "target-split-dflash verify failed\n");
            step_graph_destroy(draft_sg);
            step_graph_destroy(proj_sg);
            return false;
        }

        int accept_n = 1;
        for (int i = 0; i < q_len - 1; i++) {
            if (draft_tok[i + 1] == target_tok[i]) accept_n++;
            else break;
        }
        int bonus_tok = (accept_n < q_len) ? target_tok[accept_n - 1] : -1;
        int commit_n = accept_n + (bonus_tok >= 0 ? 1 : 0);
        if (commit_n > need_commit_budget) {
            commit_n = need_commit_budget;
            if (commit_n <= accept_n) bonus_tok = -1;
        }

        for (auto & shard : shards) restore_ssm_state(shard.cache);

        std::vector<int32_t> replay_tok((size_t)commit_n);
        for (int i = 0; i < commit_n; i++) {
            replay_tok[i] = (i < accept_n) ? draft_tok[i] : bonus_tok;
        }
        int replay_last_tok = -1;
        if (!run_target_layer_split_forward(shards, shards.front().weights,
                                            replay_tok, committed, commit_n,
                                            replay_last_tok, &feature_ring)) {
            std::fprintf(stderr, "target-split-dflash replay failed\n");
            step_graph_destroy(draft_sg);
            step_graph_destroy(proj_sg);
            return false;
        }
        last_tok = replay_last_tok;

        bool hit_eos = false;
        for (int i = 0; i < commit_n; i++) {
            out_all.push_back(replay_tok[i]);
            if (IS_EOS_TOK(replay_tok[i], shards.front().weights)) hit_eos = true;
        }
        committed += commit_n;
        n_generated += commit_n;
        n_accept_sum += std::min(accept_n, commit_n);
        n_draft_steps++;
        if (hit_eos) break;
    }
    sync_all();
    auto t_dec1 = std::chrono::steady_clock::now();
    const double decode_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
    const int total_draft_pos = std::max(1, n_draft_steps * q_len);
    const double accept_pct = 100.0 * (double)n_accept_sum / (double)total_draft_pos;
    std::printf("[target-split-dflash] decode tokens=%d time=%.3f s speed=%.2f tok/s\n",
                n_generated, decode_s, n_generated > 0 ? n_generated / decode_s : 0.0);
    std::printf("[target-split-dflash] %d draft steps, accepted=%d/%d (%.1f%%), avg commit/step=%.2f\n",
                n_draft_steps, n_accept_sum, total_draft_pos, accept_pct,
                n_draft_steps > 0 ? (double)n_generated / (double)n_draft_steps : 0.0);
    if (out_path) write_int32_file(out_path, out_all);

    step_graph_destroy(draft_sg);
    step_graph_destroy(proj_sg);
    return true;
}

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
        int max_verify_tokens) {
    if (!prompt_path || !out_path) {
        std::fprintf(stderr, "target layer split requires prompt/n_gen/out positional args\n");
        return 2;
    }
    const int n_layer = inspect_target_layer_count(target_path);
    if (n_layer <= 0) {
        std::fprintf(stderr, "target-split could not read qwen35.block_count\n");
        return 1;
    }
    const auto ranges = compute_layer_ranges(n_layer, (int)target_gpus.size(), split_weights);
    if ((int)ranges.size() != (int)target_gpus.size()) {
        std::fprintf(stderr, "bad --target-layer-split for %zu target GPUs and %d layers\n",
                     target_gpus.size(), n_layer);
        return 2;
    }
    std::vector<TargetLayerSplitShard> shards;
    shards.resize(target_gpus.size());
    for (size_t i = 0; i < target_gpus.size(); i++) {
        shards[i].gpu = target_gpus[i];
        shards[i].layer_begin = ranges[i].first;
        shards[i].layer_end = ranges[i].second;
    }
    for (auto & shard : shards) {
        shard.backend = ggml_backend_cuda_init(shard.gpu);
        if (!shard.backend) {
            std::fprintf(stderr, "target-split cuda init failed for gpu %d\n", shard.gpu);
            free_target_layer_split_shards(shards);
            return 1;
        }
    }
    for (size_t i = 0; i < target_gpus.size(); i++) {
        for (size_t j = i + 1; j < target_gpus.size(); j++) {
            if (!enable_peer_access_pair(target_gpus[i], target_gpus[j])) {
                std::fprintf(stderr,
                             "warning: CUDA peer access not fully enabled for target gpus %d,%d\n",
                             target_gpus[i], target_gpus[j]);
            }
        }
    }
    for (auto & shard : shards) {
        TargetLoadPlan plan;
        plan.layer_begin = shard.layer_begin;
        plan.layer_end = shard.layer_end;
        plan.load_output = (&shard == &shards.back());
        if (!load_target_gguf_partial(target_path, shard.backend, plan, shard.weights)) {
            std::fprintf(stderr, "target-split load gpu=%d: %s\n",
                         shard.gpu, dflash27b_last_error());
            free_target_layer_split_shards(shards);
            return 1;
        }
        std::printf("[target-split] gpu=%d layers=[%d,%d) %s\n",
                    shard.gpu, shard.layer_begin, shard.layer_end,
                    dflash27b_last_error());
        const bool allocate_target_feat = false;
        if (!create_target_cache_partial(shard.weights, max_ctx, max_verify_tokens,
                                         shard.backend, shard.cache,
                                         /*prefill_only=*/!run_dflash,
                                         shard.layer_begin, shard.layer_end,
                                         allocate_target_feat)) {
            std::fprintf(stderr, "target-split cache gpu=%d: %s\n",
                         shard.gpu, dflash27b_last_error());
            free_target_layer_split_shards(shards);
            return 1;
        }
    }

    ggml_backend_t draft_backend = nullptr;
    DraftWeights draft_weights;
    DraftFeatureMirror feature_ring;
    bool draft_backend_owned = false;
    if (load_draft) {
        for (auto & shard : shards) {
            if (shard.gpu == draft_gpu) {
                draft_backend = shard.backend;
                break;
            }
        }
        if (!draft_backend) {
            draft_backend = ggml_backend_cuda_init(draft_gpu);
            if (!draft_backend) {
                std::fprintf(stderr, "target-split draft cuda init failed for gpu %d\n", draft_gpu);
                free_target_layer_split_shards(shards);
                return 1;
            }
            draft_backend_owned = true;
        }
        std::string dp(draft_path);
        bool draft_ok = false;
        if (dp.size() >= 5 && dp.substr(dp.size() - 5) == ".gguf") {
            draft_ok = load_draft_gguf(draft_path, draft_backend, draft_weights);
        } else {
            draft_ok = load_draft_safetensors(draft_path, draft_backend, draft_weights);
        }
        if (!draft_ok) {
            std::fprintf(stderr, "target-split draft load gpu=%d: %s\n",
                         draft_gpu, dflash27b_last_error());
            free_draft_weights(draft_weights);
            if (draft_backend_owned) ggml_backend_free(draft_backend);
            free_target_layer_split_shards(shards);
            return 1;
        }
        std::printf("[target-split] draft loaded on gpu=%d format=%s\n",
                    draft_gpu,
                    (dp.size() >= 5 && dp.substr(dp.size() - 5) == ".gguf")
                        ? "gguf" : "safetensors");
        if (g_draft_swa_window > 0) {
            draft_weights.swa_window = g_draft_swa_window;
            for (int il = 0; il < draft_weights.n_layer - 1; il++) {
                draft_weights.layers[il].is_swa = true;
            }
            std::printf("[target-split] draft SWA layers: %d/%d (window=%d)\n",
                        draft_weights.n_layer - 1, draft_weights.n_layer,
                        draft_weights.swa_window);
        }
        const int cap = std::min(max_ctx, 4096);
        if (!draft_feature_mirror_init(feature_ring, draft_backend,
                                       draft_gpu, draft_gpu, cap)) {
            std::fprintf(stderr, "target-split feature ring init failed on gpu=%d\n", draft_gpu);
            draft_feature_mirror_free(feature_ring);
            free_draft_weights(draft_weights);
            if (draft_backend_owned) ggml_backend_free(draft_backend);
            free_target_layer_split_shards(shards);
            return 1;
        }
        std::printf("[target-split] draft feature ring cap=%d gpu=%d\n", cap, draft_gpu);
    }

    auto prompt = read_int32_file(prompt_path);
    if (prompt.empty()) {
        std::fprintf(stderr, "target-split empty prompt\n");
        draft_feature_mirror_free(feature_ring);
        free_draft_weights(draft_weights);
        if (draft_backend_owned) ggml_backend_free(draft_backend);
        free_target_layer_split_shards(shards);
        return 1;
    }
    if ((int)prompt.size() + n_gen + 1 > max_ctx) {
        std::fprintf(stderr, "target-split prompt (%zu) + gen (%d) exceeds max_ctx (%d)\n",
                     prompt.size(), n_gen, max_ctx);
        draft_feature_mirror_free(feature_ring);
        free_draft_weights(draft_weights);
        if (draft_backend_owned) ggml_backend_free(draft_backend);
        free_target_layer_split_shards(shards);
        return 1;
    }

    int ubatch = (prompt.size() > 2048) ? 384 : 16;
    if (const char * s = std::getenv("DFLASH27B_PREFILL_UBATCH")) {
        ubatch = std::max(1, std::atoi(s));
    }
    std::printf("[target-split] n_gpus=%zu n_layer=%d ubatch=%d max_ctx=%d\n",
                target_gpus.size(), n_layer, ubatch, max_ctx);

    int last_tok = -1;
    auto t_pf0 = std::chrono::steady_clock::now();
    if (!run_target_layer_split_forward(shards, shards.front().weights,
                                        prompt, 0, ubatch, last_tok,
                                        load_draft ? &feature_ring : nullptr)) {
        std::fprintf(stderr, "target-split prefill failed\n");
        draft_feature_mirror_free(feature_ring);
        free_draft_weights(draft_weights);
        if (draft_backend_owned) ggml_backend_free(draft_backend);
        free_target_layer_split_shards(shards);
        return 1;
    }
    auto t_pf1 = std::chrono::steady_clock::now();
    const double prefill_s = std::chrono::duration<double>(t_pf1 - t_pf0).count();
    std::printf("[target-split] prefill tokens=%zu time=%.3f s speed=%.2f tok/s last_tok=%d\n",
                prompt.size(), prefill_s, prompt.size() / prefill_s, last_tok);

    if (run_draft_smoke) {
        const int hidden = DFLASH27B_TARGET_HIDDEN;
        const int q_len = DFLASH27B_DRAFT_BLOCK_SIZE;
        const int draft_ctx = std::min((int)prompt.size(), feature_ring.cap);
        const int draft_start = (int)prompt.size() - draft_ctx;
        StepGraph draft_sg;
        int mirror_slot0 = 0;
        const bool use_mirror_view =
            draft_feature_mirror_can_view(feature_ring, (int)prompt.size(),
                                          draft_ctx, mirror_slot0);
        if (!build_draft_step(draft_sg, draft_weights, nullptr, draft_backend,
                              draft_ctx, use_mirror_view ? &feature_ring : nullptr,
                              (int)prompt.size())) {
            std::fprintf(stderr, "target-split draft smoke build failed\n");
            step_graph_destroy(draft_sg);
            draft_feature_mirror_free(feature_ring);
            free_draft_weights(draft_weights);
            if (draft_backend_owned) ggml_backend_free(draft_backend);
            free_target_layer_split_shards(shards);
            return 1;
        }
        if (!use_mirror_view &&
            !copy_feature_ring_range_to_tensor(feature_ring,
                                                draft_sg.target_hidden_cat,
                                                draft_start, draft_ctx)) {
            std::fprintf(stderr, "target-split draft smoke feature copy failed\n");
            step_graph_destroy(draft_sg);
            draft_feature_mirror_free(feature_ring);
            free_draft_weights(draft_weights);
            if (draft_backend_owned) ggml_backend_free(draft_backend);
            free_target_layer_split_shards(shards);
            return 1;
        }
        std::vector<int32_t> noise_ids(q_len, DFLASH27B_DRAFT_MASK_TOKEN_ID);
        noise_ids[0] = last_tok;
        std::vector<float> noise_embed((size_t)hidden * q_len);
        if (!shards.front().weights.embedder.embed(noise_ids.data(), q_len, noise_embed.data())) {
            std::fprintf(stderr, "target-split draft smoke embed failed\n");
            step_graph_destroy(draft_sg);
            draft_feature_mirror_free(feature_ring);
            free_draft_weights(draft_weights);
            if (draft_backend_owned) ggml_backend_free(draft_backend);
            free_target_layer_split_shards(shards);
            return 1;
        }
        ggml_backend_tensor_set(draft_sg.inp_embed, noise_embed.data(), 0,
                                sizeof(float) * noise_embed.size());
        std::vector<int32_t> pos_q(q_len), pos_k(draft_ctx + q_len);
        for (int i = 0; i < q_len; i++) pos_q[i] = draft_ctx + i;
        for (int i = 0; i < draft_ctx + q_len; i++) pos_k[i] = i;
        ggml_backend_tensor_set(draft_sg.positions, pos_q.data(), 0,
                                sizeof(int32_t) * pos_q.size());
        ggml_backend_tensor_set(draft_sg.positions_k, pos_k.data(), 0,
                                sizeof(int32_t) * pos_k.size());
        auto t_ds0 = std::chrono::steady_clock::now();
        auto st = ggml_backend_graph_compute(draft_backend, draft_sg.gf);
        auto t_ds1 = std::chrono::steady_clock::now();
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "target-split draft smoke compute failed status=%d\n", (int)st);
            step_graph_destroy(draft_sg);
            draft_feature_mirror_free(feature_ring);
            free_draft_weights(draft_weights);
            if (draft_backend_owned) ggml_backend_free(draft_backend);
            free_target_layer_split_shards(shards);
            return 1;
        }
        std::printf("[target-split] draft smoke ctx=%d q=%d time=%.3f ms\n",
                    draft_ctx, q_len,
                    std::chrono::duration<double, std::milli>(t_ds1 - t_ds0).count());
        step_graph_destroy(draft_sg);
    }

    if (run_dflash) {
        const bool ok = run_target_layer_split_dflash_decode(
            shards, draft_weights, draft_backend, draft_gpu, feature_ring,
            prompt, n_gen, last_tok, out_path);
        draft_feature_mirror_free(feature_ring);
        free_draft_weights(draft_weights);
        if (draft_backend_owned) ggml_backend_free(draft_backend);
        free_target_layer_split_shards(shards);
        return ok ? 0 : 1;
    }

    std::vector<int32_t> out_all = prompt;
    auto t_dec0 = std::chrono::steady_clock::now();
    int generated = 0;
    for (; generated < n_gen; generated++) {
        std::vector<int32_t> one(1, last_tok);
        int next_tok = -1;
        if (!run_target_layer_split_forward(shards, shards.front().weights,
                                            one, (int)out_all.size(), 1, next_tok,
                                            load_draft ? &feature_ring : nullptr)) {
            std::fprintf(stderr, "target-split decode failed at %d\n", generated);
            draft_feature_mirror_free(feature_ring);
            free_draft_weights(draft_weights);
            if (draft_backend_owned) ggml_backend_free(draft_backend);
            free_target_layer_split_shards(shards);
            return 1;
        }
        out_all.push_back(last_tok);
        if (IS_EOS_TOK(last_tok, shards.front().weights)) {
            generated++;
            break;
        }
        last_tok = next_tok;
    }
    auto t_dec1 = std::chrono::steady_clock::now();
    const double decode_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
    std::printf("[target-split] decode tokens=%d time=%.3f s speed=%.2f tok/s\n",
                generated, decode_s, generated > 0 ? generated / decode_s : 0.0);
    if (out_path) write_int32_file(out_path, out_all);
    draft_feature_mirror_free(feature_ring);
    free_draft_weights(draft_weights);
    if (draft_backend_owned) ggml_backend_free(draft_backend);
    free_target_layer_split_shards(shards);
    return 0;
}

} // namespace dflash27b
