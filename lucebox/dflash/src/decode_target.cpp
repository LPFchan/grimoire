// decode_target.cpp — target-only autoregressive generation (no draft).
//
// Used when the daemon is launched without a draft model (--pflash mode with
// no DFlash drafter).  Prefill happens inside the caller (test_dflash.cpp /
// the daemon monolith); this module provides the single-token decode loop
// that replaces the speculative (dflash) decode loop.

#include "decode_context.h"
#include "sampler.h"

#include <cstdio>
#include <chrono>

namespace dflash27b {

bool run_target_only_decode(DecodeCtx & ctx, const std::vector<int32_t> & prompt, int n_gen, std::vector<int32_t> & out_all) {
    const int hidden = ctx.hidden();
    const int vocab  = ctx.vocab();
    int committed = (int)prompt.size();
    int32_t last_tok = out_all.empty() ? -1 : out_all.back();
    if (last_tok < 0) {
        std::fprintf(stderr, "[target] no last_tok available (empty prompt or prefill failure)\n");
        return false;
    }

    auto t0 = std::chrono::steady_clock::now();
    int n_generated = 0;

    while (n_generated < n_gen) {
        if (!build_target_step(ctx.sg, ctx.w, ctx.cache, ctx.target_backend,
                                committed, /*n_tokens=*/1,
                                /*with_mask=*/false, /*capture=*/true,
                                /*capture_delta_intermediate=*/false,
                                /*fa_window=*/ctx.fa_window,
                                /*last_token_logits_only=*/false)) {
            std::fprintf(stderr, "[target] decode build failed\n");
            return false;
        }

        std::vector<float> embed_buf(hidden);
        if (!ctx.w.embedder.embed(&last_tok, 1, embed_buf.data())) {
            std::fprintf(stderr, "[target] embed failed\n");
            return false;
        }
        ggml_backend_tensor_set(ctx.sg.inp_embed, embed_buf.data(), 0,
                                sizeof(float) * hidden);

        int32_t pos4[4] = {committed, committed, committed, 0};
        ggml_backend_tensor_set(ctx.sg.positions, pos4, 0, sizeof(int32_t) * 4);

        auto st = ggml_backend_graph_compute(ctx.target_backend, ctx.sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "[target] decode compute failed code=%d\n", (int)st);
            return false;
        }

        std::vector<float> logits_buf(vocab);
        ggml_backend_tensor_get(ctx.sg.logits, logits_buf.data(), 0,
                                sizeof(float) * vocab);

        last_tok = (ctx.sampler.temp > 0.0f)
            ? sample_logits(logits_buf.data(), vocab, ctx.sampler, out_all, ctx.sampler_rng)
            : argmax_f32(logits_buf.data(), vocab);

        out_all.push_back(last_tok);
        stream_emit(ctx, last_tok);
        committed++;
        n_generated++;

        if (IS_EOS_TOK(last_tok, ctx.w)) break;
    }

    auto t1 = std::chrono::steady_clock::now();
    double gen_s = std::chrono::duration<double>(t1 - t0).count();
    double tps = n_generated / std::max(1e-9, gen_s);
    std::printf("[target] decoded %d tokens in %.3f s  ->  %.2f tok/s\n",
                n_generated, gen_s, tps);
    return true;
}

} // namespace dflash27b
