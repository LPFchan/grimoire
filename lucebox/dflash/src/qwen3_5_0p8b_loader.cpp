// GGUF loader for Qwen3.5-0.8B drafter. Reads a qwen35-arch GGUF, creates
// tensors on the requested backend.
//
// Hybrid architecture: every `full_attn_interval`-th layer is full attention
// (with M-RoPE, Q||gate packed), the rest are Gated DeltaNet.
//
// Tensor layout (qwen35 key prefix, Q8_0 for weights, F32 for norms/biases):
//
//   Top-level:
//     token_embd.weight              [embedded, vocab]         Q8_0
//     output_norm.weight             [hidden]                  F32
//     output.weight                  [hidden, vocab]           Q8_0 (lm_head)
//
//   Full-attention layers (every full_attn_interval-th: (il+1)%fai == 0):
//     attn_norm.weight               [hidden]                  F32
//     attn_q.weight                  [hidden, q_dim*2]         Q8_0 (Q||gate packed)
//     attn_k.weight                  [hidden, kv_dim]          Q8_0
//     attn_v.weight                  [hidden, kv_dim]          Q8_0
//     attn_output.weight             [q_dim, hidden]           Q8_0
//     attn_q_norm.weight             [head_dim]                F32
//     attn_k_norm.weight             [head_dim]                F32
//     ffn_norm.weight                [hidden]                  F32
//     ffn_gate.weight                [hidden, n_ff]            Q8_0
//     ffn_up.weight                  [hidden, n_ff]            Q8_0
//     ffn_down.weight                [n_ff, hidden]            Q8_0
//
//   DeltaNet layers (all others):
//     attn_norm.weight               [hidden]                  F32
//     attn_qkv.weight                [hidden, key_dim*2+value_dim] Q8_0
//     attn_gate.weight               [hidden, value_dim]       Q8_0 (z proj)
//     ssm_conv1d.weight              [ssm_inner, conv_kernel]  F32
//     ssm_alpha.weight               [hidden, n_v_heads]       F32
//     ssm_beta.weight                [hidden, n_v_heads]       F32
//     ssm_a                          [dt_rank]                 F32
//     ssm_dt.bias                    [dt_rank]                 F32
//     ssm_norm.weight                [head_v_dim]              F32
//     ssm_out.weight                 [value_dim, hidden]       Q8_0
//     ffn_norm.weight                [hidden]                  F32
//     ffn_gate.weight                [hidden, n_ff]            Q8_0
//     ffn_up.weight                  [hidden, n_ff]            Q8_0
//     ffn_down.weight                [n_ff, hidden]            Q8_0
//
// Follows the copy-tensor-from-mmap pattern from qwen3_0p6b_loader.cpp.

#include "internal.h"
#include "qwen3_5_0p8b_drafter.h"

#include <cstdio>
#include <cstring>
#include <fcntl.h>
#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dflash27b {

namespace {

bool copy_tensor_from_file(gguf_context * gctx, const char * name,
                           const void * mmap_base, size_t data_offset,
                           ggml_tensor * dst) {
    int idx = gguf_find_tensor(gctx, name);
    if (idx < 0) {
        std::fprintf(stderr, "[qwen35-0.8b] missing tensor: %s\n", name);
        return false;
    }
    const size_t off = gguf_get_tensor_offset(gctx, idx);
    const size_t bytes = ggml_nbytes(dst);
    const uint8_t * src = (const uint8_t *)mmap_base + data_offset + off;
    ggml_backend_tensor_set(dst, src, 0, bytes);
    return true;
}

uint32_t get_u32(gguf_context * g, const char * key, uint32_t def) {
    int k = gguf_find_key(g, key);
    if (k < 0) return def;
    return gguf_get_val_u32(g, k);
}

float get_f32(gguf_context * g, const char * key, float def) {
    int k = gguf_find_key(g, key);
    if (k < 0) return def;
    return gguf_get_val_f32(g, k);
}

} // namespace

bool load_qwen35_0p8b_drafter(const std::string & path,
                               ggml_backend_t backend,
                               Qwen35DrafterWeights & out) {
    out.backend = backend;

    gguf_init_params iparams{ /*no_alloc=*/ false, /*ctx=*/ nullptr };
    gguf_context * gctx = gguf_init_from_file(path.c_str(), iparams);
    if (!gctx) {
        set_last_error("gguf_init_from_file failed: " + path);
        return false;
    }

    // Validate architecture
    {
        int64_t arch_id = gguf_find_key(gctx, "general.architecture");
        if (arch_id < 0) {
            set_last_error("missing general.architecture");
            gguf_free(gctx);
            return false;
        }
        const char * arch = gguf_get_val_str(gctx, arch_id);
        if (std::string(arch) != "qwen35") {
            set_last_error(std::string("unexpected arch: ") + arch + " (expected qwen35)");
            gguf_free(gctx);
            return false;
        }
    }

    // Read metadata
    out.n_embd     = (int)get_u32(gctx, "qwen35.embedding_length", 1024);
    out.n_ff       = (int)get_u32(gctx, "qwen35.feed_forward_length", 3584);
    out.n_head     = (int)get_u32(gctx, "qwen35.attention.head_count", 8);
    out.n_head_kv  = (int)get_u32(gctx, "qwen35.attention.head_count_kv", 2);
    out.n_layer    = (int)get_u32(gctx, "qwen35.block_count", 24);
    out.n_ctx_max  = (int)get_u32(gctx, "qwen35.context_length", 131072);
    out.head_dim   = (int)get_u32(gctx, "qwen35.attention.key_length", 256);
    out.rope_theta = get_f32(gctx, "qwen35.rope.freq_base", 10000000.0f);
    out.ssm_state_size  = (int)get_u32(gctx, "qwen35.ssm.state_size", 128);
    out.ssm_conv_kernel = (int)get_u32(gctx, "qwen35.ssm.conv_kernel", 4);
    out.ssm_inner_size  = (int)get_u32(gctx, "qwen35.ssm.inner_size", 2048);
    out.ssm_dt_rank     = (int)get_u32(gctx, "qwen35.ssm.time_step_rank", 16);
    out.ssm_group_count = (int)get_u32(gctx, "qwen35.ssm.group_count", 16);
    out.full_attn_interval = (int)get_u32(gctx, "qwen35.full_attention_interval", 4);
    if (out.full_attn_interval <= 0) {
        set_last_error("full_attention_interval must be > 0");
        gguf_free(gctx);
        return false;
    }

    // rope dimension_sections (array of 4 uint32)
    {
        int64_t rid = gguf_find_key(gctx, "qwen35.rope.dimension_sections");
        if (rid < 0) {
            set_last_error("missing qwen35.rope.dimension_sections");
            gguf_free(gctx);
            return false;
        }
        size_t n = gguf_get_arr_n(gctx, rid);
        if (n < 4) {
            set_last_error("qwen35.rope.dimension_sections has < 4 entries");
            gguf_free(gctx);
            return false;
        }
        const int32_t * arr = (const int32_t *)gguf_get_arr_data(gctx, rid);
        for (int k = 0; k < 4; k++) out.rope_sections[k] = arr[k];
    }

    // Derive n_vocab from token_embd.weight's byte size and block quantization.
    {
        int tid = gguf_find_tensor(gctx, "token_embd.weight");
        if (tid < 0) {
            set_last_error("token_embd.weight not found in GGUF");
            gguf_free(gctx);
            return false;
        }
        const ggml_type ttype = gguf_get_tensor_type(gctx, tid);
        const size_t total_bytes = gguf_get_tensor_size(gctx, tid);
        const size_t elem_per_block = (size_t)ggml_blck_size(ttype);
        const size_t bytes_per_block = (size_t)ggml_type_size(ttype);
        const size_t blocks_per_row = ((size_t)out.n_embd + elem_per_block - 1) / elem_per_block;
        const size_t bytes_per_row = blocks_per_row * bytes_per_block;
        out.n_vocab = (int)(total_bytes / bytes_per_row);
    }

    // Derived dimensions
    const int n_embd       = out.n_embd;
    const int n_ff         = out.n_ff;
    const int n_head       = out.n_head;
    const int n_head_kv    = out.n_head_kv;
    const int head_dim     = out.head_dim;
    const int n_layer      = out.n_layer;
    const int q_dim        = n_head * head_dim;
    const int kv_dim       = n_head_kv * head_dim;
    const int ssm_inner    = out.ssm_inner_size;
    const int conv_kernel  = out.ssm_conv_kernel;
    const int dt_rank      = out.ssm_dt_rank;
    const int fai          = out.full_attn_interval;
    const int key_dim      = n_head * head_dim;
    const int value_dim    = ssm_inner;
    const int n_v_heads    = dt_rank;  // ssm_dt_rank = number of value heads
    const int head_v_dim   = ssm_inner / dt_rank;  // per-head value dimension

    // Count tensors for context size estimation.
    // Top: 3 (tok_embd, out_norm, output)
    // Each layer: up to 14 (5 shared + 9 deltanet extras)
    const int total_tensors = 3 + n_layer * 14;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * total_tensors + 16 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);

    // Top-level tensors
    out.tok_embd = ggml_new_tensor_2d(out.ctx, GGML_TYPE_Q8_0, n_embd, out.n_vocab);
    out.out_norm = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, n_embd);
    out.output   = ggml_new_tensor_2d(out.ctx, GGML_TYPE_Q8_0, n_embd, out.n_vocab);
    ggml_set_name(out.tok_embd, "token_embd.weight");
    ggml_set_name(out.out_norm, "output_norm.weight");
    ggml_set_name(out.output,   "output.weight");

    out.layers.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        auto & L = out.layers[il];
        const bool is_full_attn = ((il + 1) % fai == 0);

        // Shared across both layer types
        L.attn_norm = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, n_embd);
        L.ffn_norm  = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, n_embd);
        L.w_gate    = ggml_new_tensor_2d(out.ctx, GGML_TYPE_Q8_0, n_embd, n_ff);
        L.w_up      = ggml_new_tensor_2d(out.ctx, GGML_TYPE_Q8_0, n_embd, n_ff);
        L.w_down    = ggml_new_tensor_2d(out.ctx, GGML_TYPE_Q8_0, n_ff, n_embd);

        if (is_full_attn) {
            L.attn_q   = ggml_new_tensor_2d(out.ctx, GGML_TYPE_Q8_0, n_embd, q_dim * 2);
            L.attn_k   = ggml_new_tensor_2d(out.ctx, GGML_TYPE_Q8_0, n_embd, kv_dim);
            L.attn_v   = ggml_new_tensor_2d(out.ctx, GGML_TYPE_Q8_0, n_embd, kv_dim);
            L.attn_o   = ggml_new_tensor_2d(out.ctx, GGML_TYPE_Q8_0, q_dim, n_embd);
            L.q_norm   = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, head_dim);
            L.k_norm   = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, head_dim);
        } else {
            L.wqkv      = ggml_new_tensor_2d(out.ctx, GGML_TYPE_Q8_0, n_embd, key_dim * 2 + value_dim);
            L.wqkv_gate = ggml_new_tensor_2d(out.ctx, GGML_TYPE_Q8_0, n_embd, value_dim);
            L.ssm_conv1d = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, ssm_inner, conv_kernel);
            L.ssm_alpha  = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, n_embd, n_v_heads);
            L.ssm_beta   = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, n_embd, n_v_heads);
            L.ssm_a      = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, dt_rank);
            L.ssm_dt_bias = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, dt_rank);
            L.ssm_norm   = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, head_v_dim);
            L.ssm_out    = ggml_new_tensor_2d(out.ctx, GGML_TYPE_Q8_0, value_dim, n_embd);
        }
    }

    // Allocate GPU memory
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        set_last_error("ggml_backend_alloc_ctx_tensors failed for Qwen3.5-0.8B drafter");
        gguf_free(gctx);
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }

    // mmap the GGUF data section
    const size_t data_off = gguf_get_data_offset(gctx);
#if defined(_WIN32)
    std::wstring wpath;
    {
        const int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        if (wlen <= 0) {
            set_last_error("MultiByteToWideChar failed for " + path);
            gguf_free(gctx);
            return false;
        }
        wpath.resize(wlen - 1);
        MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);
    }
    HANDLE hFile = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        set_last_error("CreateFileW failed for " + path);
        gguf_free(gctx);
        return false;
    }
    HANDLE hMapping = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    CloseHandle(hFile);
    if (!hMapping) {
        set_last_error("CreateFileMappingA failed for " + path);
        gguf_free(gctx);
        return false;
    }
    void * mm = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMapping);
    if (!mm) {
        set_last_error("MapViewOfFile failed for " + path);
        gguf_free(gctx);
        return false;
    }
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        set_last_error("open failed for " + path);
        gguf_free(gctx);
        return false;
    }
    struct stat st;
    if (::fstat(fd, &st) < 0) {
        set_last_error("fstat failed for " + path);
        ::close(fd);
        gguf_free(gctx);
        return false;
    }
    void * mm = ::mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (mm == MAP_FAILED) {
        set_last_error("mmap failed for " + path);
        gguf_free(gctx);
        return false;
    }
#endif

    // Copy top-level tensors
    bool ok = true;
    ok &= copy_tensor_from_file(gctx, "token_embd.weight",  mm, data_off, out.tok_embd);
    ok &= copy_tensor_from_file(gctx, "output_norm.weight", mm, data_off, out.out_norm);
    if (gguf_find_tensor(gctx, "output.weight") >= 0) {
        ok &= copy_tensor_from_file(gctx, "output.weight", mm, data_off, out.output);
    }

    // Copy per-layer tensors
    char nm[128];
    for (int il = 0; il < n_layer; ++il) {
        const auto & L = out.layers[il];
        const bool is_full_attn = ((il + 1) % fai == 0);

        std::snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", il);
        ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.attn_norm);

        if (is_full_attn) {
            std::snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight",      il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.attn_q);
            std::snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight",      il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.attn_k);
            std::snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight",      il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.attn_v);
            std::snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.attn_o);
            std::snprintf(nm, sizeof(nm), "blk.%d.attn_q_norm.weight", il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.q_norm);
            std::snprintf(nm, sizeof(nm), "blk.%d.attn_k_norm.weight", il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.k_norm);
        } else {
            std::snprintf(nm, sizeof(nm), "blk.%d.attn_qkv.weight",    il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.wqkv);
            std::snprintf(nm, sizeof(nm), "blk.%d.attn_gate.weight",   il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.wqkv_gate);
            std::snprintf(nm, sizeof(nm), "blk.%d.ssm_conv1d.weight",  il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.ssm_conv1d);
            std::snprintf(nm, sizeof(nm), "blk.%d.ssm_alpha.weight",   il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.ssm_alpha);
            std::snprintf(nm, sizeof(nm), "blk.%d.ssm_beta.weight",    il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.ssm_beta);
            std::snprintf(nm, sizeof(nm), "blk.%d.ssm_a",              il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.ssm_a);
            std::snprintf(nm, sizeof(nm), "blk.%d.ssm_dt.bias",        il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.ssm_dt_bias);
            std::snprintf(nm, sizeof(nm), "blk.%d.ssm_norm.weight",    il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.ssm_norm);
            std::snprintf(nm, sizeof(nm), "blk.%d.ssm_out.weight",     il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.ssm_out);
        }

        // FFN norm: try ffn_norm.weight, fall back to post_attention_norm.weight
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_norm.weight", il);
        if (gguf_find_tensor(gctx, nm) >= 0) {
            ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.ffn_norm);
        } else {
            std::snprintf(nm, sizeof(nm), "blk.%d.post_attention_norm.weight", il);
            ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.ffn_norm);
        }
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight", il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.w_gate);
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_up.weight",   il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.w_up);
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_down.weight", il); ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.w_down);
    }

#if defined(_WIN32)
    UnmapViewOfFile(mm);
#else
    ::munmap(mm, st.st_size);
#endif
    gguf_free(gctx);

    if (!ok) {
        set_last_error("one or more Qwen3.5-0.8B tensors failed to load");
        ggml_backend_buffer_free(out.buf);
        ggml_free(out.ctx);
        out.buf = nullptr;
        out.ctx = nullptr;
        return false;
    }
    return true;
}

void free_qwen35_0p8b_drafter(Qwen35DrafterWeights & w) {
    if (w.buf) { ggml_backend_buffer_free(w.buf); w.buf = nullptr; }
    if (w.ctx) { ggml_free(w.ctx); w.ctx = nullptr; }
    w.layers.clear();
    w.tok_embd = w.out_norm = w.output = nullptr;
    w.backend = nullptr;
}

} // namespace dflash27b
