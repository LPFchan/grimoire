# DEC-20260518-001: PFlash Stale-Thread Deadlock — Dedicated Compressor Thread

Opened: 2026-05-18 17-00-00 KST
Recorded by agent: opencode

## Status

- Status: implemented and verified
- Deciders: operator, opencode
- Related ids: LOG-20260518-164224-encode, LOG-20260518-150423-encode

## Decision

Replace the `loop.run_in_executor` thread pool approach for `PflashDaemon.compress()` with a **dedicated background compressor thread** that owns exclusive access to the daemon's pipes. `compress()` becomes an async method that puts a job on a `Queue` and awaits a `Future`; the worker thread processes jobs one-at-a-time.

## Context

The PflashDaemon communicates with a C++ child process (`pflash_daemon`) via a single shared pipe (`_pipe_r`) for compressed token output. The original code called `daemon.compress()` via `loop.run_in_executor(None, ...)` — a thread pool executor.

Bug: When an asyncio task is cancelled (e.g., HTTP client disconnects), `run_in_executor` cancels the Future but does **not** kill the running thread. The orphan thread continues reading from the shared `_pipe_r`. The next request spawns a new executor thread that also reads from the same pipe — two threads reading the same fd concurrently corrupt each other's data, the daemon sees garbage on its write attempts, and the entire gateway deadlocks on `pipe_write`.

Observed in practice: after killing the soak client mid-run, the gateway permanently hung. All subsequent requests timed out. The pflash daemon was stuck in `pipe_write`, gateway threads in `pipe_read` on a pipe nobody could drain correctly.

## Options Considered

### A. Threading Lock + Timeout

Add a `threading.Lock` around `compress()` with 30s timeout.

- Upside: minimal code change (~5 lines)
- Downside: cancelled request's orphan thread holds the lock for up to 30s, blocking all new requests. No recovery if daemon itself hangs.

### B. Lock + Signal-Interrupted `os.read`

Use `signal.alarm` to interrupt the blocking read so the orphan thread releases the lock.

- Upside: bounded wait time
- Downside: `SIGALRM` is process-wide, not thread-safe. Fragile.

### C. Close Pipe on Staleness + Restart Daemon

Detect orphan executor future, close `_pipe_r` to unstick threads, restart daemon.

- Upside: immediate unstall ~2s
- Downside: corrupts thread pool executor (threads see `BrokenPipeError`). Requires daemon restart logic.

### D. Per-Request Pipe via SCM_RIGHTS

Each `compress()` call creates a new pipe pair. Pass the write-end fd to the daemon via a Unix socket (`os.send_fds`/`os.recv_fds`).

- Upside: zero shared state. Thread-safe by construction.
- Downside: requires ~50 lines of C++ changes in `pflash_daemon.cpp` and ~20 lines of Python. Requires rebuild + redeploy. We **tried** passing fd as integer in the command string — this doesn't work because file descriptors are process-local.

### E. Dedicated Compressor Thread + Async Queue (Chosen)

Single worker thread owns all daemon access. `compress()` is `async def` — creates a `Future`, puts a `_CompressJob` on a `Queue`, awaits the future. Worker processes jobs serially; cancelled futures are simply dropped.

- Upside: no stale thread problem (only one thread ever touches the pipes). Pure Python, no C++ changes, no rebuild. No lock contention. Immediate cancellation (no 30s wait).
- Downside: moderate refactor (~60 lines). Still vulnerable to daemon process hang (worker thread blocks).

## Rationale

Approach E was chosen because it eliminates the entire class of bug (orphan threads competing on a shared fd) with zero C++ changes and zero deployment overhead. The only files changed were `daemon.py` and `prefill.py` — both pure Python and live-reloaded via bind mount.

The soak-until-deadlock bug has been eliminated: even if the HTTP client disconnects mid-request, the worker thread finishes the current compress call, then processes the next queued job cleanly. No pipe corruption, no deadlock.

## Consequences

- `PflashDaemon.compress()` is now `async def` — callers must `await` it rather than wrapping in `run_in_executor`
- `prefill.py`'s `maybe_compress` was updated to `await daemon.compress(chunk)` directly
- The old `_lock`, `_pipe_r` state sharing, and `run_in_executor` pattern are removed
- Verified: 100/100 consecutive PFlash-compressing iterations from inside the container, zero failures, zero VRAM drift, zero deadlocks

## Related

- All fix candidates analyzed in detail: `78cd53b` (commit)

