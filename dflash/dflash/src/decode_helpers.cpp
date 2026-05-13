// decode_helpers.cpp — standalone helpers extracted from test_dflash.cpp.
//
// File-scope globals required by the decode pipeline (and declared extern in
// decode_context.h) live here so both the library and test_dflash have a
// single definition.
//
// All functions are inside namespace dflash27b.

#include "decode_context.h"
#include "internal.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>
#include <random>

// ── File-scope globals (defined here, extern in decode_context.h) ──
static constexpr int KQ_MASK_PAD = 32;
int g_kq_stride_pad    = KQ_MASK_PAD;   // overridden to 256 when TBQ KV is active
int g_max_ctx_override = 0;             // overridden by --max-ctx=N (default 4096)
int g_fa_window        = 2048;          // overridden by DFLASH27B_FA_WINDOW=N
int g_draft_swa_window = 0;             // draft SWA window (0 = disabled); --draft-swa=N
int g_draft_ctx_max    = 4096;          // draft context cap; --draft-ctx-max=N
static int align_up(int x, int a) { return ((x + a - 1) / a) * a; }

static constexpr uint16_t F16_ZERO    = 0x0000;
static constexpr uint16_t F16_NEG_INF = 0xFC00;

namespace dflash27b {

// ── I/O helpers ────────────────────────────────────────────────────

std::vector<int32_t> read_int32_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto sz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<int32_t> out(sz / sizeof(int32_t));
    f.read((char *)out.data(), sz);
    return out;
}

bool write_int32_file(const std::string & path, const std::vector<int32_t> & v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write((const char *)v.data(), v.size() * sizeof(int32_t));
    return (bool)f;
}

int argmax_f32(const float * x, int n) {
    int best = 0;
    float bv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > bv) { bv = x[i]; best = i; }
    return best;
}

// ── Mask builders ──────────────────────────────────────────────────

void build_causal_mask(std::vector<uint16_t> & out,
                              int kv_len, int n_tokens, int kv_start,
                              int win_start) {
    const int kv_pad = align_up(kv_len, g_kq_stride_pad);
    const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
    out.assign((size_t)kv_pad * q_pad, F16_NEG_INF);
    const int abs_end = win_start + kv_len;
    for (int q = 0; q < n_tokens; q++) {
        const int abs_q = kv_start + q;
        const int min_k = std::max(0, win_start);
        const int max_k = abs_q;
        for (int k = min_k; k <= max_k && k < abs_end; k++) {
            out[(size_t)q * kv_pad + (k - win_start)] = F16_ZERO;
        }
    }
}

void build_tree_mask(const DDTree & tree, int past_length,
                            std::vector<uint16_t> & out_mask,
                            int win_start) {
    const int N      = 1 + tree.n_nodes;
    const int win_len = past_length + N - win_start;
    const int kv_pad = align_up(win_len, g_kq_stride_pad);
    const int q_pad  = align_up(N,      KQ_MASK_PAD);
    out_mask.assign((size_t)kv_pad * q_pad, F16_NEG_INF);
    for (int q = 0; q < N; q++) {
        for (int k = std::max(0, win_start); k < past_length; k++) {
            out_mask[(size_t)q * kv_pad + (k - win_start)] = F16_ZERO;
        }
        for (int j = 0; j < N; j++) {
            if (tree.visibility[(size_t)q * N + j]) {
                out_mask[(size_t)q * kv_pad + (past_length + j - win_start)] = F16_ZERO;
            }
        }
    }
}

// ── DDTree support (ported from liranringel/ddtree/ddtree.py) ──────

void extract_draft_topk(const float * logits,
                               int n_positions, int vocab, int K,
                               float * out_log_probs,
                               int32_t * out_token_ids,
                               float temperature) {
    struct Entry { float logit; int32_t id; };
    auto cmp_greater = [](const Entry & a, const Entry & b) {
        return a.logit > b.logit;
    };

    const float inv_t = 1.0f / std::max(1e-3f, temperature);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_positions; i++) {
        const float * li = logits + (size_t)i * vocab;
        std::vector<Entry> heap;
        heap.reserve(K);

        float running_max     = -INFINITY;
        float running_sum_exp = 0.0f;
        for (int j = 0; j < vocab; j++) {
            const float l = li[j] * inv_t;

            if (l > running_max) {
                if (running_max > -INFINITY) {
                    running_sum_exp = running_sum_exp * std::exp(running_max - l);
                }
                running_sum_exp += 1.0f;
                running_max = l;
            } else {
                running_sum_exp += std::exp(l - running_max);
            }

            if ((int)heap.size() < K) {
                heap.push_back({l, (int32_t)j});
                std::push_heap(heap.begin(), heap.end(), cmp_greater);
            } else if (l > heap.front().logit) {
                std::pop_heap(heap.begin(), heap.end(), cmp_greater);
                heap.back() = {l, (int32_t)j};
                std::push_heap(heap.begin(), heap.end(), cmp_greater);
            }
        }
        const float log_z = running_max + std::log(running_sum_exp);

        std::sort_heap(heap.begin(), heap.end(), cmp_greater);
        for (int k = 0; k < K; k++) {
            out_log_probs[(size_t)i * K + k] = heap[k].logit - log_z;
            out_token_ids[(size_t)i * K + k] = heap[k].id;
        }
    }
}

DDTree build_ddtree(const float * top_log_probs,
                           const int32_t * top_token_ids,
                           int L, int K, int budget,
                           bool chain_seed) {
    DDTree tree;
    if (budget <= 0 || L <= 0) {
        tree.parents.push_back(-1);
        tree.child_maps.emplace_back();
        tree.visibility.assign(1, 1);
        return tree;
    }

    struct HeapEntry {
        float                neg_logw;
        std::vector<int>     ranks;
        int                  parent_index;
        int                  depth;
        int                  rank;
        float                logw;
    };
    struct HeapCmp {
        bool operator()(const HeapEntry & a, const HeapEntry & b) const {
            return a.neg_logw > b.neg_logw;
        }
    };
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapCmp> heap;

    tree.token_ids.reserve(budget);
    tree.depths.reserve(budget);
    tree.parents.reserve(budget + 1);
    tree.parents.push_back(-1);
    tree.child_maps.emplace_back();

    if (chain_seed) {
        const int chain_depth = std::min(L, budget);
        float cum_logw = 0.0f;
        int   prev_idx = 0;
        for (int d = 1; d <= chain_depth; d++) {
            const int32_t tok_id = top_token_ids[(size_t)(d - 1) * K + 0];
            cum_logw += top_log_probs[(size_t)(d - 1) * K + 0];

            const int cur_idx = tree.n_nodes + 1;
            tree.token_ids.push_back(tok_id);
            tree.depths.push_back(d);
            tree.parents.push_back(prev_idx);
            tree.child_maps.emplace_back();
            tree.child_maps[prev_idx][tok_id] = cur_idx;
            tree.n_nodes++;

            if (K > 1) {
                const float sibling_logw = cum_logw
                    - top_log_probs[(size_t)(d - 1) * K + 0]
                    + top_log_probs[(size_t)(d - 1) * K + 1];
                heap.push({
                    /*neg_logw*/ -sibling_logw,
                    /*ranks   */ {1},
                    /*parent  */ prev_idx,
                    /*depth   */ d,
                    /*rank    */ 1,
                    /*logw    */ sibling_logw,
                });
            }
            prev_idx = cur_idx;
        }
    } else {
        const float root_logw = top_log_probs[0 * K + 0];
        heap.push({
            /*neg_logw*/ -root_logw,
            /*ranks   */ {0},
            /*parent  */ 0,
            /*depth   */ 1,
            /*rank    */ 0,
            /*logw    */ root_logw,
        });
    }

    while (!heap.empty() && tree.n_nodes < budget) {
        HeapEntry top = heap.top();
        heap.pop();

        const int    depth_minus_1 = top.depth - 1;
        const int    rank          = top.rank;
        const int32_t token_id     = top_token_ids[(size_t)depth_minus_1 * K + rank];

        const int current_index = tree.n_nodes + 1;
        tree.token_ids.push_back(token_id);
        tree.depths.push_back(top.depth);
        tree.parents.push_back(top.parent_index);
        tree.child_maps.emplace_back();
        tree.child_maps[top.parent_index][token_id] = current_index;
        tree.n_nodes++;

        if (rank + 1 < K) {
            const float sibling_logw = top.logw
                - top_log_probs[(size_t)depth_minus_1 * K + rank]
                + top_log_probs[(size_t)depth_minus_1 * K + rank + 1];
            std::vector<int> sibling_ranks = top.ranks;
            sibling_ranks.back() = rank + 1;
            heap.push({
                /*neg_logw*/ -sibling_logw,
                /*ranks   */ std::move(sibling_ranks),
                /*parent  */ top.parent_index,
                /*depth   */ top.depth,
                /*rank    */ rank + 1,
                /*logw    */ sibling_logw,
            });
        }

        if (top.depth < L) {
            const float child_logw = top.logw
                + top_log_probs[(size_t)top.depth * K + 0];
            std::vector<int> child_ranks = top.ranks;
            child_ranks.push_back(0);
            heap.push({
                /*neg_logw*/ -child_logw,
                /*ranks   */ std::move(child_ranks),
                /*parent  */ current_index,
                /*depth   */ top.depth + 1,
                /*rank    */ 0,
                /*logw    */ child_logw,
            });
        }
    }

    // Build ancestor-only visibility mask (flat row-major, (1+n)^2).
    const int N = 1 + tree.n_nodes;
    tree.visibility.assign((size_t)N * N, 0);
    tree.visibility[0 * N + 0] = 1;
    for (int i = 1; i < N; i++) {
        const int p = tree.parents[i];
        for (int j = 0; j < i; j++) {
            tree.visibility[(size_t)i * N + j] = tree.visibility[(size_t)p * N + j];
        }
        tree.visibility[(size_t)i * N + i] = 1;
    }

    return tree;
}

std::vector<int> follow_verified_tree(const DDTree & tree,
                                             const int32_t * posterior,
                                             int & out_next_token,
                                             int * out_node_idx) {
    std::vector<int> accepted;
    accepted.reserve(tree.n_nodes + 1);
    accepted.push_back(0);

    int current_index = 0;
    int next_token    = posterior[current_index];
    while (true) {
        const auto & children = tree.child_maps[current_index];
        auto it = children.find(next_token);
        if (it == children.end()) break;
        current_index = it->second;
        accepted.push_back(current_index);
        next_token = posterior[current_index];
    }
    out_next_token = next_token;
    if (out_node_idx) *out_node_idx = current_index;
    return accepted;
}

// ── CUDA peer-access helpers ───────────────────────────────────────

bool enable_peer_access_one_way(int device, int peer) {
    if (device == peer) return true;
    int can_access = 0;
    cudaError_t err = cudaDeviceCanAccessPeer(&can_access, device, peer);
    if (err != cudaSuccess || !can_access) return false;
    err = cudaSetDevice(device);
    if (err != cudaSuccess) return false;
    err = cudaDeviceEnablePeerAccess(peer, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
        return true;
    }
    return err == cudaSuccess;
}

bool enable_peer_access_pair(int a, int b) {
    if (a == b) return true;
    const bool ab = enable_peer_access_one_way(a, b);
    const bool ba = enable_peer_access_one_way(b, a);
    return ab && ba;
}

bool copy_peer_async(void * dst, int dst_device,
                            const void * src, int src_device,
                            size_t bytes,
                            cudaStream_t stream) {
    if (bytes == 0) return true;
    cudaError_t err = cudaSuccess;
    if (dst_device == src_device) {
        err = cudaSetDevice(dst_device);
        if (err != cudaSuccess) return false;
        err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
    } else {
        err = cudaSetDevice(dst_device);
        if (err != cudaSuccess) return false;
        err = cudaMemcpyPeerAsync(dst, dst_device, src, src_device, bytes, stream);
    }
    return err == cudaSuccess;
}

bool ensure_bf16_staging(DraftFeatureMirror & mirror, size_t elems) {
    if (elems <= mirror.bf16_staging_elems) return true;
    cudaError_t err = cudaSetDevice(mirror.device);
    if (err != cudaSuccess) return false;
    if (mirror.bf16_staging) {
        cudaFree(mirror.bf16_staging);
        mirror.bf16_staging = nullptr;
        mirror.bf16_staging_elems = 0;
    }
    err = cudaMalloc(&mirror.bf16_staging, elems * sizeof(uint16_t));
    if (err != cudaSuccess) return false;
    mirror.bf16_staging_elems = elems;
    return true;
}

void draft_feature_mirror_free(DraftFeatureMirror & mirror) {
    if (mirror.bf16_staging) {
        cudaSetDevice(mirror.device);
        cudaFree(mirror.bf16_staging);
        mirror.bf16_staging = nullptr;
        mirror.bf16_staging_elems = 0;
    }
    if (mirror.buf) {
        ggml_backend_buffer_free(mirror.buf);
        mirror.buf = nullptr;
    }
    if (mirror.ctx) {
        ggml_free(mirror.ctx);
        mirror.ctx = nullptr;
    }
    mirror.target_feat = nullptr;
    mirror.device = 0;
    mirror.target_device = 0;
    mirror.cap = 0;
}

bool draft_feature_mirror_init(DraftFeatureMirror & mirror,
                                      ggml_backend_t backend,
                                      int device,
                                      int target_device,
                                      int cap) {
    draft_feature_mirror_free(mirror);
    if (cap <= 0) return false;
    mirror.device = device;
    mirror.target_device = target_device;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 4 + 16 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    mirror.ctx = ggml_init(ip);
    if (!mirror.ctx) return false;

    const int fc_in = DFLASH27B_DRAFT_N_TARGET_LAYERS * DFLASH27B_TARGET_HIDDEN;
    mirror.target_feat = ggml_new_tensor_2d(mirror.ctx, GGML_TYPE_F32, fc_in, cap);
    ggml_set_name(mirror.target_feat, "draft_target_feat_mirror");
    mirror.buf = ggml_backend_alloc_ctx_tensors(mirror.ctx, backend);
    if (!mirror.buf) {
        draft_feature_mirror_free(mirror);
        return false;
    }
    const size_t bytes = (size_t)fc_in * (size_t)cap * sizeof(float);
    cudaSetDevice(device);
    cudaError_t err = cudaMemset(mirror.target_feat->data, 0, bytes);
    if (err != cudaSuccess) {
        draft_feature_mirror_free(mirror);
        return false;
    }
    mirror.cap = cap;
    return true;
}

bool draft_feature_mirror_can_view(const DraftFeatureMirror & mirror,
                                           int committed,
                                           int ctx_len,
                                           int & slot0) {
    if (!mirror.target_feat || mirror.cap <= 0) return false;
    if (ctx_len <= 0 || ctx_len > mirror.cap || committed < ctx_len) return false;
    const int start = committed - ctx_len;
    slot0 = start % mirror.cap;
    return slot0 + ctx_len <= mirror.cap;
}

bool draft_feature_mirror_sync_range(const TargetCache & cache,
                                             const DraftFeatureMirror & mirror,
                                             int start_pos,
                                             int n_tokens) {
    if (!cache.target_feat || !mirror.target_feat || mirror.cap <= 0) return false;
    if (n_tokens <= 0) return true;
    if (n_tokens > mirror.cap) return false;

    const int fc_in = DFLASH27B_DRAFT_N_TARGET_LAYERS * DFLASH27B_TARGET_HIDDEN;
    const int src_cap = cache.target_feat_cap;
    const size_t src_stride = cache.target_feat->nb[1];
    const size_t dst_stride = mirror.target_feat->nb[1];

    int done = 0;
    while (done < n_tokens) {
        const int src_slot = (start_pos + done) % src_cap;
        const int dst_slot = (start_pos + done) % mirror.cap;
        const int src_run = src_cap - src_slot;
        const int dst_run = mirror.cap - dst_slot;
        const int run = std::min(n_tokens - done, std::min(src_run, dst_run));
        const size_t elems = (size_t)run * (size_t)fc_in;
        const void * src =
            (const char *)cache.target_feat->data + (size_t)src_slot * src_stride;
        void * dst =
            (char *)mirror.target_feat->data + (size_t)dst_slot * dst_stride;
        auto bf16_to_f32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);
        if (mirror.device == mirror.target_device) {
            cudaSetDevice(mirror.device);
            bf16_to_f32(src, (float *)dst, (int64_t)elems, nullptr);
        } else {
            DraftFeatureMirror & mutable_mirror =
                const_cast<DraftFeatureMirror &>(mirror);
            if (!ensure_bf16_staging(mutable_mirror, elems)) return false;
            if (!copy_peer_async(mirror.bf16_staging, mirror.device,
                                 const_cast<void *>(src), mirror.target_device,
                                 elems * sizeof(uint16_t))) {
                return false;
            }
            cudaSetDevice(mirror.device);
            bf16_to_f32(mirror.bf16_staging, (float *)dst, (int64_t)elems, nullptr);
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return false;
        done += run;
    }
    return cudaDeviceSynchronize() == cudaSuccess;
}

bool draft_feature_mirror_sync_tail(const TargetCache & cache,
                                            const DraftFeatureMirror & mirror,
                                            int committed) {
    if (!mirror.target_feat || committed <= 0) return true;
    const int n = std::min(committed, mirror.cap);
    return draft_feature_mirror_sync_range(cache, mirror, committed - n, n);
}

} // namespace dflash27b
