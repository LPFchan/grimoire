// decode_snapshot.h — prefix-cache snapshot commands extracted from
// test/dflash_entrypoint.cpp.
//
// Provides the daemon-loop command dispatch for SNAPSHOT / SNAPSHOT_THIN /
// FREE_SNAPSHOT / SAVE_SNAPSHOT / LOAD_SNAPSHOT / LIST_SLOTS and the
// post-cache-reset restore logic for RESTORE / RESTORE_CHAIN.
//
// Each function returns bool and prints diagnostic / error messages directly
// to stderr/stdout (matching the daemon convention).

#pragma once

#include "internal.h"  // PrefixSnapshot, snapshot_target_cache, etc.
#include "decode_context.h"

#include <string>

namespace dflash27b {

// ── Snapshot command dispatch ──────────────────────────────────────
//
// Handle one line of daemon input that starts with a snapshot-related
// command prefix.  Returns true if the line was a self-contained snapshot
// command and the caller should `continue` to the next daemon iteration.
// Returns false if the line is NOT a snapshot command, or is a fall-through
// command (RESTORE_CHAIN / RESTORE) that sets state on `ctx` for the caller
// to process.
//
// Fall-through commands set the following ctx fields:
//   ctx.n_gen, ctx.prompt_file_str          — generation parameters
//   ctx.restore_from_slot, ctx.restore_slot_id  — single-slot restore
//   ctx.chain_restore_requested, ctx.chain_thick_slot, ctx.chain_thin_ids
//                                            — chain restore
//   ctx.snap_pos, ctx.snap_slot             — inline-snap at prefill boundary
//
// The caller should call apply_snapshot_restore() after cache reset to
// execute the pending restore.  Bare `<prompt> <n_gen>` lines are NOT
// handled here — the caller should fall back to legacy parsing if both
// restore flags are false.
bool handle_snapshot_command(DecodeCtx & ctx,
                             PrefixSnapshot * prefix_snapshots,
                             int n_slots,
                             const std::string & line,
                             ggml_backend_t backend,
                             const TargetWeights & w,
                             TargetCache & cache);

// ── Post-reset restore ─────────────────────────────────────────────
//
// After the caller has reset the target cache (via reset_target_cache()),
// apply any pending single-slot restore (ctx.restore_from_slot) or chain
// restore (ctx.chain_restore_requested).
//
// Returns true on success.  On failure prints an error and returns false;
// the caller should `continue` to the next daemon iteration.
//
// Clears the request flags on ctx after applying (so they aren't
// accidentally re-applied on a subsequent iteration).
bool apply_snapshot_restore(DecodeCtx & ctx,
                            PrefixSnapshot * prefix_snapshots,
                            TargetCache & cache);

} // namespace dflash27b
