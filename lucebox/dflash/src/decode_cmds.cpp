#include "decode_context.h"
#include "laguna_daemon.h"  // read_uncounted_i32
#include "qwen3_drafter.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace dflash27b {

bool handle_daemon_command(DecodeCtx & ctx, const std::string & line) {
    auto starts_with = [](const std::string& s, const char* pre) {
        size_t n = std::strlen(pre);
        return s.size() >= n && s.compare(0, n, pre) == 0;
    };

    // ── Park/unpark commands (additive on top of latest daemon) ─────
    if (starts_with(line, "park")) {
        bool want_draft  = (line == "park" || line == "park all" || line == "park draft");
        bool want_target = (line == "park" || line == "park all" || line == "park target");
        if (want_draft && ctx.has_draft && !ctx.draft_parked) {
            free_draft_weights(ctx.dw);
            step_graph_destroy(ctx.draft_sg);
            step_graph_destroy(ctx.proj_sg);
            ctx.draft_parked = true;
            std::printf("[park] draft released\n"); std::fflush(stdout);
        }
        if (want_target && !ctx.target_parked) {
            step_graph_destroy(ctx.proj_sg);
            free_target_weights(ctx.w);
            ctx.target_parked = true;
            std::printf("[park] target released\n"); std::fflush(stdout);
        }
        stream_emit(ctx, -1);
        return true;
    }
    if (line == "free drafter" || line == "drafter free") {
        if (ctx.drafter_loaded) {
            free_drafter(ctx.drafter_ctx);
            ctx.drafter_loaded = false;
            std::printf("[drafter] freed\n"); std::fflush(stdout);
        }
        stream_emit(ctx, -1);
        return true;
    }
    if (starts_with(line, "unpark")) {
        bool want_draft  = (line == "unpark" || line == "unpark all" || line == "unpark draft");
        bool want_target = (line == "unpark" || line == "unpark all" || line == "unpark target");
        if (want_target && ctx.target_parked) {
            if (!load_target_gguf(ctx.target_path, ctx.target_backend, ctx.w)) {
                std::fprintf(stderr, "[unpark] target: %s\n", dflash27b_last_error());
                stream_emit(ctx, -1); return true;
            }
            ctx.target_parked = false;
            std::printf("[unpark] target restored\n"); std::fflush(stdout);
        }
        if (want_draft && ctx.has_draft && ctx.draft_parked) {
            std::string dp(ctx.draft_path);
            bool draft_ok = (dp.size() >= 5 && dp.substr(dp.size() - 5) == ".gguf")
                ? load_draft_gguf(ctx.draft_path, ctx.draft_backend, ctx.dw)
                : load_draft_safetensors(ctx.draft_path, ctx.draft_backend, ctx.dw);
            if (!draft_ok) {
                std::fprintf(stderr, "[unpark] draft: %s\n", dflash27b_last_error());
                stream_emit(ctx, -1); return true;
            }
            if (ctx.draft_swa_window > 0) {
                ctx.dw.swa_window = ctx.draft_swa_window;
                for (int il = 0; il < ctx.dw.n_layer - 1; il++)
                    ctx.dw.layers[il].is_swa = true;
            }
            ctx.draft_parked = false;
            std::printf("[unpark] draft restored\n"); std::fflush(stdout);
        }
        stream_emit(ctx, -1);
        return true;
    }

    // ── Compress command (pflash speculative prefill) ───────────────
    if (starts_with(line, "compress ")) {
        char ppath[1024];
        int  keep_x1000 = 0;
        char drafter_path[1024];
        const int n = std::sscanf(line.c_str() + 9, "%1023s %d %1023s",
                                    ppath, &keep_x1000, drafter_path);
        if (n != 3) {
            std::fprintf(stderr,
                          "[compress] bad args, need: <bin> <keep_x1000> <drafter.gguf>\n");
            stream_emit(ctx, -1); return true;
        }
        if (!ctx.pflash) {
            std::fprintf(stderr, "[compress] pflash is disabled (pass --pflash to enable)\n");
            stream_emit(ctx, -1); return true;
        }

        auto src_ids = read_uncounted_i32(ppath);
        if (src_ids.empty()) {
            std::fprintf(stderr, "[compress] empty input\n");
            stream_emit(ctx, -1); return true;
        }

        const bool restore_draft = ctx.has_draft && !ctx.draft_parked;
        if (restore_draft) {
            free_draft_weights(ctx.dw);
            ctx.draft_parked = true;
            std::printf("[compress] draft parked\n"); std::fflush(stdout);
        }

        if (!ctx.drafter_loaded) {
            if (!load_drafter(drafter_path, /*gpu_layers=*/999, ctx.drafter_ctx)) {
                std::fprintf(stderr, "[compress] load_drafter failed: %s\n",
                              dflash27b_last_error());
                if (restore_draft) {
                    if (!load_draft_safetensors(ctx.draft_path, ctx.draft_backend, ctx.dw)) {
                        std::fprintf(stderr, "[compress] draft restore after drafter fail: %s\n",
                                     dflash27b_last_error());
                    } else {
                        if (ctx.draft_swa_window > 0) {
                            ctx.dw.swa_window = ctx.draft_swa_window;
                            for (int il = 0; il < ctx.dw.n_layer - 1; il++)
                                ctx.dw.layers[il].is_swa = true;
                        }
                        ctx.draft_parked = false;
                    }
                }
                stream_emit(ctx, -1); return true;
            }
            ctx.drafter_loaded = true;
            std::printf("[drafter] loaded %s vocab=%d\n",
                         drafter_path, ctx.drafter_ctx.weights.n_vocab);
            std::fflush(stdout);
        }

        const float keep = (float)keep_x1000 / 1000.0f;
        auto compressed = drafter_score_and_compress(ctx.drafter_ctx, src_ids, keep);
        std::printf("[compress] %zu -> %zu tokens (keep_ratio=%.3f)\n",
                     src_ids.size(), compressed.size(), keep);
        std::fflush(stdout);

        if (ctx.drafter_loaded) {
            free_drafter(ctx.drafter_ctx);
            ctx.drafter_loaded = false;
            std::printf("[compress] drafter parked\n"); std::fflush(stdout);
        }

        if (restore_draft) {
            if (!load_draft_safetensors(ctx.draft_path, ctx.draft_backend, ctx.dw)) {
                std::fprintf(stderr, "[compress] draft restore: %s\n",
                              dflash27b_last_error());
                stream_emit(ctx, -1); return true;
            }
            if (ctx.draft_swa_window > 0) {
                ctx.dw.swa_window = ctx.draft_swa_window;
                for (int il = 0; il < ctx.dw.n_layer - 1; il++)
                    ctx.dw.layers[il].is_swa = true;
            }
            ctx.draft_parked = false;
            std::printf("[compress] draft restored\n"); std::fflush(stdout);
        }

        for (int32_t t : compressed) stream_emit(ctx, t);
        stream_emit(ctx, -1);
        return true;
    }

    // ── Prefix-cache snapshot commands (#59) ──────────────────────
    if (!ctx.pflash && (line.rfind("SNAPSHOT", 0) == 0 || line.rfind("RESTORE", 0) == 0 || line.rfind("FREE_SNAPSHOT", 0) == 0 || line.rfind("SAVE_SNAPSHOT", 0) == 0 || line.rfind("LOAD_SNAPSHOT", 0) == 0)) {
        std::fprintf(stderr, "[snap] pflash is disabled (pass --pflash to enable)\n");
        stream_emit(ctx, -1);
        return true;
    }
    if (line.rfind("SNAPSHOT_THIN ", 0) == 0) {
        int slot = -1, kv_start = -1, kv_end = -1;
        if (std::sscanf(line.c_str() + 14, "%d %d %d", &slot, &kv_start, &kv_end) != 3
            || slot < 0 || slot >= DecodeCtx::PREFIX_CACHE_SLOTS) {
            std::fprintf(stderr, "[snap] SNAPSHOT_THIN bad args\n");
            stream_emit(ctx, -1);
            return true;
        }
        if (!snapshot_target_cache_thin(ctx.w, ctx.cache, ctx.target_backend, kv_start, kv_end,
                                         ctx.prefix_snapshots[slot])) {
            std::fprintf(stderr, "[snap] thin failed slot=%d: %s\n", slot,
                         dflash27b_last_error());
            stream_emit(ctx, -1);
            return true;
        }
        std::printf("[snap] thin slot=%d kv=%d,%d\n", slot, kv_start, kv_end);
        std::fflush(stdout);
        stream_emit(ctx, -1);
        return true;
    }
    if (line.rfind("SNAPSHOT ", 0) == 0) {
        int slot = -1;
        if (std::sscanf(line.c_str() + 9, "%d", &slot) != 1
            || slot < 0 || slot >= DecodeCtx::PREFIX_CACHE_SLOTS) {
            std::fprintf(stderr, "[snap] invalid slot %d\n", slot);
            stream_emit(ctx, -1);
            return true;
        }
        if (!snapshot_target_cache(ctx.w, ctx.cache, ctx.target_backend, ctx.prefix_snapshots[slot])) {
            std::fprintf(stderr, "[snap] failed slot=%d: %s\n", slot, dflash27b_last_error());
            stream_emit(ctx, -1);
            return true;
        }
        std::printf("[snap] slot=%d cur_pos=%d\n", slot, ctx.prefix_snapshots[slot].cur_pos);
        std::fflush(stdout);
        stream_emit(ctx, -1);
        return true;
    }
    if (line.rfind("FREE_SNAPSHOT ", 0) == 0) {
        int slot = -1;
        if (std::sscanf(line.c_str() + 14, "%d", &slot) != 1
            || slot < 0 || slot >= DecodeCtx::PREFIX_CACHE_SLOTS) {
            stream_emit(ctx, -1);
            return true;
        }
        free_prefix_snapshot(ctx.prefix_snapshots[slot]);
        std::printf("[snap] freed slot=%d\n", slot);
        std::fflush(stdout);
        stream_emit(ctx, -1);
        return true;
    }
    // ── SSD swap: SAVE_SNAPSHOT / LOAD_SNAPSHOT ────────────────
    if (line.rfind("SAVE_SNAPSHOT ", 0) == 0) {
        int slot_local = -1;
        char snap_path[1024] = {0};
        if (std::sscanf(line.c_str() + 14, "%d %1023s", &slot_local, snap_path) != 2
            || slot_local < 0 || slot_local >= DecodeCtx::PREFIX_CACHE_SLOTS) {
            std::fprintf(stderr, "[snap] SAVE_SNAPSHOT bad args\n");
            stream_emit(ctx, -1);
            return true;
        }
        PrefixSnapshot & ps = ctx.prefix_snapshots[slot_local];
        if (!ps.ctx) {
            std::fprintf(stderr, "[snap] SAVE_SNAPSHOT slot %d empty\n", slot_local);
            stream_emit(ctx, -1);
            return true;
        }
        {
            std::ofstream sf(snap_path, std::ios::binary);
            if (!sf) {
                std::fprintf(stderr, "[snap] SAVE_SNAPSHOT open %s failed\n", snap_path);
                stream_emit(ctx, -1);
                return true;
            }
            const char magic[4] = {'D', 'F', 'S', 'N'};
            const uint32_t version = 2;
            sf.write(magic, 4);
            uint32_t v = version;
            sf.write(reinterpret_cast<char *>(&v), 4);
            v = (uint32_t)ps.cur_pos;
            sf.write(reinterpret_cast<char *>(&v), 4);
            v = (uint32_t)ps.last_tok;
            sf.write(reinterpret_cast<char *>(&v), 4);
            v = (uint32_t)ps.max_ctx;
            sf.write(reinterpret_cast<char *>(&v), 4);
            v = (uint32_t)ps.kv_k_type;
            sf.write(reinterpret_cast<char *>(&v), 4);
            v = (uint32_t)ps.kv_v_type;
            sf.write(reinterpret_cast<char *>(&v), 4);
            v = (uint32_t)ps.target_feat_cap;
            sf.write(reinterpret_cast<char *>(&v), 4);
            char is_thin = (char)ps.is_thin;
            sf.write(&is_thin, 1);
            v = (uint32_t)ps.kv_start;
            sf.write(reinterpret_cast<char *>(&v), 4);
            v = (uint32_t)ps.kv_end;
            sf.write(reinterpret_cast<char *>(&v), 4);
            v = (uint32_t)ps.attn_k_snap.size();
            sf.write(reinterpret_cast<char *>(&v), 4);
            v = (uint32_t)ps.ssm_state_snap.size();
            sf.write(reinterpret_cast<char *>(&v), 4);

            auto write_tensor = [&sf, &v](ggml_tensor * t) -> bool {
                if (!t) return true;
                v = (uint32_t)t->type;
                sf.write(reinterpret_cast<char *>(&v), 4);
                for (int d = 0; d < (int)GGML_MAX_DIMS; d++) {
                    v = (uint32_t)t->ne[d];
                    sf.write(reinterpret_cast<char *>(&v), 4);
                }
                const size_t nbytes = ggml_nbytes(t);
                uint64_t nb64 = (uint64_t)nbytes;
                sf.write(reinterpret_cast<char *>(&nb64), 8);
                if (nbytes > 0) {
                    std::vector<uint8_t> buf(nbytes);
                    ggml_backend_tensor_get(t, buf.data(), 0, nbytes);
                    sf.write(reinterpret_cast<char *>(buf.data()), (std::streamsize)nbytes);
                }
                return (bool)sf;
            };

            bool ok = true;
            for (auto * t : ps.attn_k_snap) if (!write_tensor(t)) { ok = false; break; }
            if (ok) for (auto * t : ps.attn_v_snap) if (!write_tensor(t)) { ok = false; break; }
            if (ok && !ps.is_thin) {
                for (auto * t : ps.ssm_state_snap) if (!write_tensor(t)) { ok = false; break; }
                if (ok) for (auto * t : ps.conv_state_snap) if (!write_tensor(t)) { ok = false; break; }
                if (ok && !write_tensor(ps.target_feat_snap)) ok = false;
            }
            sf.close();
            if (!ok) {
                std::fprintf(stderr, "[snap] SAVE_SNAPSHOT write failed slot=%d\n", slot_local);
                std::remove(snap_path);
                stream_emit(ctx, -1);
                return true;
            }
        }
        free_prefix_snapshot(ctx.prefix_snapshots[slot_local]);
        std::printf("[snap] saved slot=%d %s\n", slot_local, snap_path);
        std::fflush(stdout);
        stream_emit(ctx, -1);
        return true;
    }
    if (line.rfind("LOAD_SNAPSHOT ", 0) == 0) {
        int slot_local = -1;
        char snap_path[1024] = {0};
        if (std::sscanf(line.c_str() + 14, "%d %1023s", &slot_local, snap_path) != 2
            || slot_local < 0 || slot_local >= DecodeCtx::PREFIX_CACHE_SLOTS) {
            std::fprintf(stderr, "[snap] LOAD_SNAPSHOT bad args\n");
            stream_emit(ctx, -1);
            return true;
        }
        PrefixSnapshot & ps = ctx.prefix_snapshots[slot_local];
        std::ifstream sf(snap_path, std::ios::binary);
        if (!sf) {
            std::fprintf(stderr, "[snap] LOAD_SNAPSHOT open %s failed\n", snap_path);
            stream_emit(ctx, -1);
            return true;
        }
        auto read_u32 = [&sf]() -> uint32_t {
            uint32_t v; sf.read(reinterpret_cast<char *>(&v), 4); return v;
        };
        auto read_u64 = [&sf]() -> uint64_t {
            uint64_t v; sf.read(reinterpret_cast<char *>(&v), 8); return v;
        };
        char hdr[4];
        sf.read(hdr, 4);
        if (hdr[0] != 'D' || hdr[1] != 'F' || hdr[2] != 'S' || hdr[3] != 'N') {
            std::fprintf(stderr, "[snap] LOAD_SNAPSHOT bad magic\n");
            stream_emit(ctx, -1);
            return true;
        }
        uint32_t version = read_u32();
        if (version != 1 && version != 2) {
            std::fprintf(stderr, "[snap] LOAD_SNAPSHOT unsupported version %u\n", version);
            stream_emit(ctx, -1);
            return true;
        }
        int32_t cur_pos_load   = (int32_t)read_u32();
        int32_t last_tok_load  = (int32_t)read_u32();
        int32_t max_ctx_load   = (int32_t)read_u32();
        ggml_type kv_k_load    = (ggml_type)read_u32();
        ggml_type kv_v_load    = GGML_TYPE_COUNT;
        if (version >= 2) {
            kv_v_load = (ggml_type)read_u32();
        }
        int32_t feat_cap_load  = (int32_t)read_u32();
        char is_thin_byte;
        sf.read(&is_thin_byte, 1);
        bool is_thin_load = (is_thin_byte != 0);
        int32_t kv_start_load = (int32_t)read_u32();
        int32_t kv_end_load   = (int32_t)read_u32();
        int n_full_attn_load  = (int)read_u32();
        int n_delta_load      = (int)read_u32();

        free_prefix_snapshot(ps);

        struct TensorMeta {
            ggml_type ttype;
            int64_t ne[GGML_MAX_DIMS] = {};
            uint64_t nbytes = 0;
            ggml_tensor * t = nullptr;
            std::vector<uint8_t> data;
        };
        std::vector<TensorMeta> metas;
        const int n_tensors = 2 * n_full_attn_load
            + (is_thin_load ? 0 : 2 * n_delta_load + 1);
        metas.reserve(n_tensors);

        for (int i = 0; i < n_tensors; i++) {
            TensorMeta m;
            m.ttype = (ggml_type)read_u32();
            for (int d = 0; d < (int)GGML_MAX_DIMS; d++)
                m.ne[d] = (int64_t)read_u32();
            m.nbytes = read_u64();
            if (m.nbytes > 0) {
                m.data.resize(m.nbytes);
                sf.read(reinterpret_cast<char *>(m.data.data()),
                        (std::streamsize)m.nbytes);
            }
            metas.push_back(std::move(m));
        }

        {
            ggml_init_params ip{};
            ip.mem_size   = (size_t)(n_tensors + 16) * ggml_tensor_overhead();
            ip.mem_buffer = nullptr;
            ip.no_alloc   = true;
            ps.ctx = ggml_init(ip);
            if (!ps.ctx) {
                std::fprintf(stderr, "[snap] LOAD_SNAPSHOT ggml_init\n");
                stream_emit(ctx, -1);
                return true;
            }
            ps.attn_k_snap.assign(n_full_attn_load, nullptr);
            ps.attn_v_snap.assign(n_full_attn_load, nullptr);

            int idx = 0;
            for (int i = 0; i < n_full_attn_load; i++) {
                ps.attn_k_snap[i] = ggml_new_tensor_3d(ps.ctx, metas[idx].ttype,
                    metas[idx].ne[0], metas[idx].ne[1], metas[idx].ne[2]);
                metas[idx].t = ps.attn_k_snap[i]; idx++;
                ps.attn_v_snap[i] = ggml_new_tensor_3d(ps.ctx, metas[idx].ttype,
                    metas[idx].ne[0], metas[idx].ne[1], metas[idx].ne[2]);
                metas[idx].t = ps.attn_v_snap[i]; idx++;
            }
            if (!is_thin_load) {
                ps.ssm_state_snap.assign(n_delta_load, nullptr);
                ps.conv_state_snap.assign(n_delta_load, nullptr);
                for (int i = 0; i < n_delta_load; i++) {
                    ps.ssm_state_snap[i] = ggml_new_tensor_3d(ps.ctx, metas[idx].ttype,
                        metas[idx].ne[0], metas[idx].ne[1], metas[idx].ne[2]);
                    metas[idx].t = ps.ssm_state_snap[i]; idx++;
                }
                for (int i = 0; i < n_delta_load; i++) {
                    ps.conv_state_snap[i] = ggml_new_tensor_2d(ps.ctx, metas[idx].ttype,
                        metas[idx].ne[0], metas[idx].ne[1]);
                    metas[idx].t = ps.conv_state_snap[i]; idx++;
                }
                ps.target_feat_snap = ggml_new_tensor_2d(ps.ctx, metas[idx].ttype,
                    metas[idx].ne[0], metas[idx].ne[1]);
                metas[idx].t = ps.target_feat_snap; idx++;
            }

            ps.buf = ggml_backend_alloc_ctx_tensors(ps.ctx, ctx.target_backend);
            if (!ps.buf) {
                std::fprintf(stderr, "[snap] LOAD_SNAPSHOT alloc failed\n");
                free_prefix_snapshot(ps);
                stream_emit(ctx, -1);
                return true;
            }

            for (auto & m : metas) {
                if (m.t && m.data.empty() == false) {
                    ggml_backend_tensor_set(m.t, m.data.data(), 0, m.data.size());
                }
            }
        }
        ps.cur_pos         = cur_pos_load;
        ps.last_tok        = last_tok_load;
        ps.max_ctx         = max_ctx_load;
        ps.kv_k_type       = kv_k_load;
        if (version >= 2) {
            ps.kv_v_type = kv_v_load;
        } else if (!ps.attn_v_snap.empty() && ps.attn_v_snap[0] != nullptr) {
            ps.kv_v_type = ps.attn_v_snap[0]->type;
        } else {
            ps.kv_v_type = kv_k_load;
        }
        ps.target_feat_cap = feat_cap_load;
        ps.is_thin         = is_thin_load;
        ps.kv_start        = kv_start_load;
        ps.kv_end          = kv_end_load;
        sf.close();
        std::printf("[snap] loaded slot=%d %s\n", slot_local, snap_path);
        std::fflush(stdout);
        stream_emit(ctx, -1);
        return true;
    }
    if (line == "LIST_SLOTS") {
        std::printf("[snap] slots=");
        bool first = true;
        for (int i = 0; i < DecodeCtx::PREFIX_CACHE_SLOTS; i++) {
            if (ctx.prefix_snapshots[i].ctx != nullptr) {
                std::printf("%s%d", first ? "" : ",", i);
                first = false;
            }
        }
        std::printf("\n");
        std::fflush(stdout);
        return true;
    }
    if (line.rfind("RESTORE_CHAIN ", 0) == 0) {
        int  thick_slot_local = -2;
        char thin_str[256]    = {0};
        char ppath[1024]      = {0};
        int  n_gen_local      = 0;
        if (std::sscanf(line.c_str() + 14, "%d %255s %1023s %d",
                        &thick_slot_local, thin_str, ppath, &n_gen_local) != 4) {
            std::fprintf(stderr, "[snap] RESTORE_CHAIN bad args\n");
            stream_emit(ctx, -1);
            return true;
        }
        if (thick_slot_local != -1
            && (thick_slot_local < 0 || thick_slot_local >= DecodeCtx::PREFIX_CACHE_SLOTS
                || ctx.prefix_snapshots[thick_slot_local].ctx == nullptr
                || ctx.prefix_snapshots[thick_slot_local].is_thin)) {
            std::fprintf(stderr, "[snap] RESTORE_CHAIN bad thick slot=%d\n", thick_slot_local);
            stream_emit(ctx, -1);
            return true;
        }
        std::vector<int> thin_ids_local;
        bool thin_parse_ok = true;
        if (std::strcmp(thin_str, "-") != 0 && thin_str[0] != '\0') {
            const char * p = thin_str;
            while (*p && thin_parse_ok) {
                char * end = nullptr;
                long id_l = std::strtol(p, &end, 10);
                if (end == p) {
                    std::fprintf(stderr,
                        "[snap] RESTORE_CHAIN malformed thin list near '%s'\n", p);
                    thin_parse_ok = false; break;
                }
                int id = (int)id_l;
                if (id < 0 || id >= DecodeCtx::PREFIX_CACHE_SLOTS
                    || ctx.prefix_snapshots[id].ctx == nullptr
                    || !ctx.prefix_snapshots[id].is_thin) {
                    std::fprintf(stderr, "[snap] RESTORE_CHAIN bad thin slot=%d\n", id);
                    thin_parse_ok = false; break;
                }
                thin_ids_local.push_back(id);
                if (*end == '\0') break;
                if (*end != ',') {
                    std::fprintf(stderr,
                        "[snap] RESTORE_CHAIN expected ',' after slot %d, got '%c'\n",
                        id, *end);
                    thin_parse_ok = false; break;
                }
                p = end + 1;
                if (*p == '\0' || *p == ',') {
                    std::fprintf(stderr,
                        "[snap] RESTORE_CHAIN empty thin slot entry\n");
                    thin_parse_ok = false; break;
                }
            }
        }
        if (!thin_parse_ok) {
            stream_emit(ctx, -1);
            return true;
        }
        ctx.n_gen                    = n_gen_local;
        ctx.prompt_file_str          = ppath;
        ctx.chain_restore_requested  = true;
        ctx.chain_thick_slot         = thick_slot_local;
        ctx.chain_thin_ids           = std::move(thin_ids_local);
        return false;  // fall through to generation
    }
    if (line.rfind("RESTORE ", 0) == 0) {
        int slot = -1;
        char ppath[1024];
        if (std::sscanf(line.c_str() + 8, "%d %1023s %d", &slot, ppath, &ctx.n_gen) != 3
            || slot < 0 || slot >= DecodeCtx::PREFIX_CACHE_SLOTS
            || ctx.prefix_snapshots[slot].ctx == nullptr) {
            std::fprintf(stderr, "[snap] RESTORE bad args or empty slot %d\n", slot);
            stream_emit(ctx, -1);
            return true;
        }
        ctx.prompt_file_str = ppath;
        ctx.restore_from_slot = true;
        ctx.restore_slot_id   = slot;
        if (const char * sp = std::strstr(line.c_str(), "snap=")) {
            if (std::sscanf(sp, "snap=%d:%d", &ctx.snap_pos, &ctx.snap_slot) != 2
                || ctx.snap_slot < 0 || ctx.snap_slot >= DecodeCtx::PREFIX_CACHE_SLOTS) {
                std::fprintf(stderr, "[snap] bad inline-snap arg\n");
                ctx.snap_pos = -1; ctx.snap_slot = -1;
            }
        }
        return false;  // fall through to generation
    }

    // Legacy: bare `<prompt_file> <n_gen>` line — full reset path.
    char ppath[1024];
    if (std::sscanf(line.c_str(), "%1023s %d", ppath, &ctx.n_gen) != 2) return true;
    ctx.prompt_file_str = ppath;
    if (const char * sp = std::strstr(line.c_str(), "snap=")) {
        if (std::sscanf(sp, "snap=%d:%d", &ctx.snap_pos, &ctx.snap_slot) != 2
            || ctx.snap_slot < 0 || ctx.snap_slot >= DecodeCtx::PREFIX_CACHE_SLOTS) {
            std::fprintf(stderr, "[snap] bad inline-snap arg\n");
            ctx.snap_pos = -1; ctx.snap_slot = -1;
        }
    }
    return false;  // fall through to generation
}

} // namespace dflash27b
