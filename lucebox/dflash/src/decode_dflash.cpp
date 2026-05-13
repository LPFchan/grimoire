// decode_dflash.cpp — speculative decode loop (dflash daemon pipeline).
//
// THIS FILE IS A STUB.
//
// The full speculative decode loop (~900 lines) is currently embedded
// directly in test_dflash.cpp because it shares ~40 local variables and
// lambdas with the containing main() function.  Extracting it here is a
// planned follow-up once the daemon monolith's per-request state has been
// fully moved into DecodeCtx.
//
// For now, all callers go through the inline loop in test_dflash.cpp.
// This file exists so the build system can link decode_*.cpp sources
// consistently, and so decode_context.h's forward declaration of
// run_dflash_decode() has a translation unit to land in.

#include "decode_context.h"

#include <cstdio>

namespace dflash27b {

bool run_dflash_decode(DecodeCtx & ctx, const std::vector<int32_t> & prompt, int n_gen, std::vector<int32_t> & out_all) {
    (void)ctx;
    (void)prompt;
    (void)n_gen;
    (void)out_all;
    std::fprintf(stderr, "[dflash] run_dflash_decode stub called — "
                         "the real loop is inline in test_dflash.cpp\n");
    return false;
}

} // namespace dflash27b
