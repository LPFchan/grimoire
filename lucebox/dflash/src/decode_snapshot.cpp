// decode_snapshot.cpp — prefix-cache snapshot commands and restore logic
// extracted from test/dflash_entrypoint.cpp.
//
// All functions are inside namespace dflash27b.

#include "decode_snapshot.h"
#include "internal.h"
#include "decode_context.h"
#include "dflash27b.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace dflash27b {

// ── SAVE_SNAPSHOT: serialize a PrefixSnapshot to disk ───────────────
//
// Header format (all little-endian):
//   magic(4) version(4) cur_pos(4) last_tok(4) max_ctx(4)
//   kv_k_type(4) kv_v_type(4) target_feat_cap(4) is_thin(1)
//   kv_start(4) kv_end(4) n_attn_k(4) n_ssm(4)
// Followed by the raw tensor bytes for attn_k, attn_v, and (if not thin)
// ssm_state, conv_state, target_feat.
static bool save_snapshot_to_disk(const PrefixSnapshot & ps,
                                   const char * snap_path) {
    std::ofstream sf(snap_path, std::ios::binary);
    if (!sf) {
        std::fprintf(stderr, "[snap] SAVE_SNAPSHOT open %s failed\n", snap_path);
        return false;
    }
    const char magic[4] = {'D', 'F', 'S', 'N'};
    const uint32_t version = 2;
    sf.write(magic, 4);
    auto write_u32 = [&sf](uint32_t v) {
        sf.write(reinterpret_cast<const char *>(&v), 4);
    };
    write_u32(version);
    write_u32((uint32_t)ps.cur_pos);
    write_u32((uint32_t)ps.last_tok);
    write_u32((uint32_t)ps.max_ctx);
    write_u32((uint32_t)ps.kv_k_type);
    write_u32((uint32_t)ps.kv_v_type);
    write_u32((uint32_t)ps.target_feat_cap);
    char is_thin = (char)ps.is_thin;
    sf.write(&is_thin, 1);
    write_u32((uint32_t)ps.kv_start);
    write_u32((uint32_t)ps.kv_end);
    write_u32((uint32_t)ps.attn_k_snap.size());
    write_u32((uint32_t)ps.ssm_state_snap.size());

    auto write_tensor = [&sf](ggml_tensor * t) -> bool {
        if (!t) return true;
        uint32_t v = (uint32_t)t->type;
        sf.write(reinterpret_cast<const char *>(&v), 4);
        for (int d = 0; d < (int)GGML_MAX_DIMS; d++) {
            v = (uint32_t)t->ne[d];
            sf.write(reinterpret_cast<const char *>(&v), 4);
        }
        const size_t nbytes = ggml_nbytes(t);
        uint64_t nb64 = (uint64_t)nbytes;
        sf.write(reinterpret_cast<const char *>(&nb64), 8);
        if (nbytes > 0) {
            std::vector<uint8_t> buf(nbytes);
            ggml_backend_tensor_get(t, buf.data(), 0, nbytes);
            sf.write(reinterpret_cast<const char *>(buf.data()),
                     (std::streamsize)nbytes);
        }
        return (bool)sf;
    };

    bool ok = true;
    for (auto * t : ps.attn_k_snap)
        if (!write_tensor(t)) { ok = false; break; }
    if (ok)
        for (auto * t : ps.attn_v_snap)
            if (!write_tensor(t)) { ok = false; break; }
    if (ok && !ps.is_thin) {
        for (auto * t : ps.ssm_state_snap)
            if (!write_tensor(t)) { ok = false; break; }
        if (ok)
            for (auto * t : ps.conv_state_snap)
                if (!write_tensor(t)) { ok = false; break; }
        if (ok && !write_tensor(ps.target_feat_snap)) ok = false;
    }
    sf.close();
    if (!ok) {
        std::fprintf(stderr, "[snap] SAVE_SNAPSHOT write failed\n");
        std::remove(snap_path);
        return false;
    }
    return true;
}

// ── LOAD_SNAPSHOT: deserialize a PrefixSnapshot from disk ───────────
//
// Reads the format written by save_snapshot_to_disk, allocates GPU
// buffers on `backend`, and places the result in `ps`.  Any prior
// contents of `ps` are freed first.
static bool load_snapshot_from_disk(PrefixSnapshot & ps,
                                     const char * snap_path,
                                     ggml_backend_t backend) {
    std::ifstream sf(snap_path, std::ios::binary);
    if (!sf) {
        std::fprintf(stderr, "[snap] LOAD_SNAPSHOT open %s failed\n", snap_path);
        return false;
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
        return false;
    }
    uint32_t version = read_u32();
    if (version != 1 && version != 2) {
        std::fprintf(stderr, "[snap] LOAD_SNAPSHOT unsupported version %u\n", version);
        return false;
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

    // Free any existing snapshot in this slot.
    free_prefix_snapshot(ps);

    // Phase 1: read metadata for all tensors.
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

    // Phase 2: create ggml tensors from metadata.
    {
        ggml_init_params ip{};
        ip.mem_size   = (size_t)(n_tensors + 16) * ggml_tensor_overhead();
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        ps.ctx = ggml_init(ip);
        if (!ps.ctx) {
            std::fprintf(stderr, "[snap] LOAD_SNAPSHOT ggml_init\n");
            return false;
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

        // Phase 3: allocate backend buffer.
        ps.buf = ggml_backend_alloc_ctx_tensors(ps.ctx, backend);
        if (!ps.buf) {
            std::fprintf(stderr, "[snap] LOAD_SNAPSHOT alloc failed\n");
            free_prefix_snapshot(ps);
            return false;
        }

        // Phase 4: copy data from temp buffers to GPU tensors.
        for (auto & m : metas) {
            if (m.t && !m.data.empty()) {
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
    return true;
}

// ── Snapshot command dispatch ──────────────────────────────────────

bool handle_snapshot_command(DecodeCtx & ctx,
                             PrefixSnapshot * prefix_snapshots,
                             int n_slots,
                             const std::string & line,
                             ggml_backend_t backend,
                             const TargetWeights & w,
                             TargetCache & cache) {
    // pflash guard: snapshot commands need pflash enabled.
    if (!ctx.pflash && (line.rfind("SNAPSHOT", 0) == 0 ||
                        line.rfind("RESTORE", 0) == 0 ||
                        line.rfind("FREE_SNAPSHOT", 0) == 0 ||
                        line.rfind("SAVE_SNAPSHOT", 0) == 0 ||
                        line.rfind("LOAD_SNAPSHOT", 0) == 0)) {
        std::fprintf(stderr, "[snap] pflash is disabled (pass --pflash to enable)\n");
        stream_emit(ctx, -1);
        return true; // handled (error)
    }

    // SNAPSHOT_THIN <slot> <kv_start> <kv_end>
    if (line.rfind("SNAPSHOT_THIN ", 0) == 0) {
        int slot = -1, kv_start = -1, kv_end = -1;
        if (std::sscanf(line.c_str() + 14, "%d %d %d", &slot, &kv_start, &kv_end) != 3
            || slot < 0 || slot >= n_slots) {
            std::fprintf(stderr, "[snap] SNAPSHOT_THIN bad args\n");
            stream_emit(ctx, -1);
            return true;
        }
        if (!snapshot_target_cache_thin(w, cache, backend, kv_start, kv_end,
                                         prefix_snapshots[slot])) {
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

    // SNAPSHOT <slot>
    if (line.rfind("SNAPSHOT ", 0) == 0) {
        int slot = -1;
        if (std::sscanf(line.c_str() + 9, "%d", &slot) != 1
            || slot < 0 || slot >= n_slots) {
            std::fprintf(stderr, "[snap] invalid slot %d\n", slot);
            stream_emit(ctx, -1);
            return true;
        }
        if (!snapshot_target_cache(w, cache, backend, prefix_snapshots[slot])) {
            std::fprintf(stderr, "[snap] failed slot=%d: %s\n", slot, dflash27b_last_error());
            stream_emit(ctx, -1);
            return true;
        }
        std::printf("[snap] slot=%d cur_pos=%d\n", slot, prefix_snapshots[slot].cur_pos);
        std::fflush(stdout);
        stream_emit(ctx, -1);
        return true;
    }

    // FREE_SNAPSHOT <slot>
    if (line.rfind("FREE_SNAPSHOT ", 0) == 0) {
        int slot = -1;
        if (std::sscanf(line.c_str() + 14, "%d", &slot) != 1
            || slot < 0 || slot >= n_slots) {
            stream_emit(ctx, -1);
            return true;
        }
        free_prefix_snapshot(prefix_snapshots[slot]);
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
            || slot_local < 0 || slot_local >= n_slots) {
            std::fprintf(stderr, "[snap] SAVE_SNAPSHOT bad args\n");
            stream_emit(ctx, -1);
            return true;
        }
        const PrefixSnapshot & ps = prefix_snapshots[slot_local];
        if (!ps.ctx) {
            std::fprintf(stderr, "[snap] SAVE_SNAPSHOT slot %d empty\n", slot_local);
            stream_emit(ctx, -1);
            return true;
        }
        if (!save_snapshot_to_disk(ps, snap_path)) {
            stream_emit(ctx, -1);
            return true;
        }
        // Free VRAM for the slot after saving.
        free_prefix_snapshot(prefix_snapshots[slot_local]);
        std::printf("[snap] saved slot=%d %s\n", slot_local, snap_path);
        std::fflush(stdout);
        stream_emit(ctx, -1);
        return true;
    }

    if (line.rfind("LOAD_SNAPSHOT ", 0) == 0) {
        int slot_local = -1;
        char snap_path[1024] = {0};
        if (std::sscanf(line.c_str() + 14, "%d %1023s", &slot_local, snap_path) != 2
            || slot_local < 0 || slot_local >= n_slots) {
            std::fprintf(stderr, "[snap] LOAD_SNAPSHOT bad args\n");
            stream_emit(ctx, -1);
            return true;
        }
        if (!load_snapshot_from_disk(prefix_snapshots[slot_local], snap_path, backend)) {
            stream_emit(ctx, -1);
            return true;
        }
        std::printf("[snap] loaded slot=%d %s\n", slot_local, snap_path);
        std::fflush(stdout);
        stream_emit(ctx, -1);
        return true;
    }

    // LIST_SLOTS
    if (line == "LIST_SLOTS") {
        std::printf("[snap] slots=");
        bool first = true;
        for (int i = 0; i < n_slots; i++) {
            if (prefix_snapshots[i].ctx != nullptr) {
                std::printf("%s%d", first ? "" : ",", i);
                first = false;
            }
        }
        std::printf("\n");
        std::fflush(stdout);
        return true;
    }

    // ── Fall-through commands ──────────────────────────────────
    //
    // RESTORE_CHAIN, RESTORE, and bare-promt lines are NOT handled
    // here — they set state on ctx and return false so the caller
    // can proceed to prefill.

    // RESTORE_CHAIN <thick_slot> <thin_slot_list> <prompt_file> <n_gen>
    if (line.rfind("RESTORE_CHAIN ", 0) == 0) {
        int  thick_slot_local = -2;
        char thin_str[256]    = {0};
        char ppath[1024]      = {0};
        int  n_gen_local      = 0;
        if (std::sscanf(line.c_str() + 14, "%d %255s %1023s %d",
                        &thick_slot_local, thin_str, ppath, &n_gen_local) != 4) {
            std::fprintf(stderr, "[snap] RESTORE_CHAIN bad args\n");
            stream_emit(ctx, -1);
            return true; // error, skip
        }
        // Validate thick_slot (-1 = none).
        if (thick_slot_local != -1
            && (thick_slot_local < 0 || thick_slot_local >= n_slots
                || prefix_snapshots[thick_slot_local].ctx == nullptr
                || prefix_snapshots[thick_slot_local].is_thin)) {
            std::fprintf(stderr, "[snap] RESTORE_CHAIN bad thick slot=%d\n", thick_slot_local);
            stream_emit(ctx, -1);
            return true;
        }
        // Parse thin slot list.
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
                if (id < 0 || id >= n_slots
                    || prefix_snapshots[id].ctx == nullptr
                    || !prefix_snapshots[id].is_thin) {
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
        // Set state on ctx for caller to process (fall-through to prefill).
        ctx.n_gen                   = n_gen_local;
        ctx.prompt_file_str         = ppath;
        ctx.chain_restore_requested = true;
        ctx.chain_thick_slot        = thick_slot_local;
        ctx.chain_thin_ids          = std::move(thin_ids_local);
        return false; // not handled — fall through to prefill
    }

    // RESTORE <slot> <prompt_file> <n_gen>
    if (line.rfind("RESTORE ", 0) == 0) {
        int slot = -1;
        char ppath[1024];
        if (std::sscanf(line.c_str() + 8, "%d %1023s %d", &slot, ppath, &ctx.n_gen) != 3
            || slot < 0 || slot >= n_slots
            || prefix_snapshots[slot].ctx == nullptr) {
            std::fprintf(stderr, "[snap] RESTORE bad args or empty slot %d\n", slot);
            stream_emit(ctx, -1);
            return true;
        }
        ctx.prompt_file_str  = ppath;
        ctx.restore_from_slot = true;
        ctx.restore_slot_id   = slot;
        // Parse optional inline-snap suffix: snap=<pos>:<slot_id>
        ctx.snap_pos = -1;
        ctx.snap_slot = -1;
        if (const char * sp = std::strstr(line.c_str(), "snap=")) {
            if (std::sscanf(sp, "snap=%d:%d", &ctx.snap_pos, &ctx.snap_slot) != 2
                || ctx.snap_slot < 0 || ctx.snap_slot >= n_slots) {
                ctx.snap_pos = -1; ctx.snap_slot = -1;
            }
        }
        return false; // not handled — fall through to prefill
    }

    // Not a snapshot command at all.
    return false;
}

// ── Post-reset restore ─────────────────────────────────────────────

bool apply_snapshot_restore(DecodeCtx & ctx,
                            PrefixSnapshot * prefix_snapshots,
                            TargetCache & cache) {
    // After cache is fresh, optionally restore from snapshot.
    if (ctx.restore_from_slot) {
        if (!restore_target_cache(prefix_snapshots[ctx.restore_slot_id], cache)) {
            std::fprintf(stderr, "[snap] restore failed: %s\n", dflash27b_last_error());
            stream_emit(ctx, -1);
            ctx.restore_from_slot = false;
            return false;
        }
        std::printf("[snap] restored slot=%d cur_pos=%d\n",
                    ctx.restore_slot_id, cache.cur_pos);
        std::fflush(stdout);
        free_prefix_snapshot(prefix_snapshots[ctx.restore_slot_id]);
        ctx.restore_from_slot = false;
    }

    // After cache is fresh, optionally apply chain restore.
    if (ctx.chain_restore_requested) {
        const PrefixSnapshot * thick_ptr =
            (ctx.chain_thick_slot == -1) ? nullptr : &prefix_snapshots[ctx.chain_thick_slot];
        std::vector<const PrefixSnapshot *> thin_ptrs;
        for (int id : ctx.chain_thin_ids)
            thin_ptrs.push_back(&prefix_snapshots[id]);
        if (!restore_target_cache_chain(thick_ptr,
                                         thin_ptrs.empty() ? nullptr : thin_ptrs.data(),
                                         (int)thin_ptrs.size(),
                                         cache)) {
            std::fprintf(stderr, "[snap] RESTORE_CHAIN failed: %s\n", dflash27b_last_error());
            stream_emit(ctx, -1);
            ctx.chain_restore_requested = false;
            ctx.chain_thin_ids.clear();
            return false;
        }
        std::printf("[snap] chain restored thick=%d thins=%zu cur_pos=%d\n",
                    ctx.chain_thick_slot, thin_ptrs.size(), cache.cur_pos);
        std::fflush(stdout);
        if (ctx.chain_thick_slot >= 0) {
            free_prefix_snapshot(prefix_snapshots[ctx.chain_thick_slot]);
        }
        for (int id : ctx.chain_thin_ids) {
            free_prefix_snapshot(prefix_snapshots[id]);
        }
        ctx.chain_restore_requested = false;
        ctx.chain_thin_ids.clear();
    }

    return true;
}

} // namespace dflash27b
