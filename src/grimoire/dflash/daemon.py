"""DFlash daemon process manager.

Spawns the DFlash binary, communicates via stdin/stdout protocol,
and provides the generate() and compress() interfaces used by the gateway.
"""

import logging
import os
import struct
import subprocess
import tempfile
import time
from typing import Optional

from grimoire import config

logger = logging.getLogger(__name__)

DFLASH_BIN = os.path.join(config.DFLASH_HOME, "dflash")

# VRAM delta (MiB) above the pre-spawn baseline that signals the daemon
# has loaded weights. Target+draft for a 27B model is ~18 GB; we use a
# conservative threshold so other processes on the same GPU can't satisfy
# it.
LOADED_VRAM_DELTA_MIB = 12000


def _prepend_library_dir(env: dict[str, str], path: str) -> None:
    existing = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p and p != path]
    env["LD_LIBRARY_PATH"] = ":".join([path, *existing]) if path else ":".join(existing)


class DflashDaemon:
    """Manage a DFlash daemon subprocess.

    Handles:
      - Spawning the daemon with target + draft models
      - stdin/stdout protocol communication
      - VRAM-delta-based health check
      - Token streaming via pipe
      - PFlash compress commands
      - Park/unpark for VRAM management
    """

    def __init__(
        self,
        target_path: str,
        draft_path: Optional[str] = None,
        max_ctx: int = 16384,
        budget: int = 22,
        gpu_id: int = 0,
        pflash: bool = True,
        dflash: bool = True,
        prefill_threshold: int = 32000,
        prefill_keep_ratio: float = 0.05,
        kv_k_type: str = "q8_0",
        kv_v_type: str = "q8_0",
        fa_window: int = 2048,
    ):
        self.target_path = target_path
        self.draft_path = draft_path
        self.max_ctx = max_ctx
        self.budget = budget
        self.gpu_id = gpu_id
        self.pflash = pflash
        self.dflash = dflash
        self.prefill_threshold = prefill_threshold
        self.prefill_keep_ratio = prefill_keep_ratio
        self.kv_k_type = kv_k_type
        self.kv_v_type = kv_v_type
        self.fa_window = fa_window

        self.proc: Optional[subprocess.Popen] = None
        self.r_pipe: int = -1

    def _read_pipe_int32(self) -> Optional[int]:
        b = os.read(self.r_pipe, 4)
        if not b or len(b) < 4:
            return None
        return struct.unpack("<i", b)[0]

    def _send_expect_ack(self, cmd: str) -> None:
        self._send(cmd)
        ack = self._read_pipe_int32()
        if ack != -1:
            raise RuntimeError(f"dflash daemon command failed or desynced: {cmd.strip()}")

    def spawn(self, timeout: float = 600.0) -> None:
        """Spawn the dflash daemon and wait for it to load.

        Raises RuntimeError if the daemon fails to load within timeout.
        """
        baseline_vram = self._read_gpu_vram_mib()

        r_pipe, w_pipe = os.pipe()
        stream_fd = w_pipe

        cmd = [DFLASH_BIN, self.target_path]
        if self.draft_path:
            cmd.append(self.draft_path)
        cmd += [
            "--daemon",
            f"--max-ctx={self.max_ctx}",
            f"--stream-fd={stream_fd}",
        ]
        if self.pflash:
            cmd.append("--pflash")
        if self.dflash:
            # Fast rollback caches a second KV copy for undoing failed
            # speculative steps. Not needed in pflash-only mode where
            # there's no draft speculation to roll back.
            cmd.append("--fast-rollback")
            cmd += ["--ddtree", f"--ddtree-budget={self.budget}"]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        _prepend_library_dir(env, config.DFLASH_HOME)
        env["DFLASH27B_KV_K"] = self.kv_k_type
        env["DFLASH27B_KV_V"] = self.kv_v_type
        env["DFLASH27B_FA_WINDOW"] = str(self.fa_window)

        logger.info(f"Starting dflash daemon on GPU {self.gpu_id}")
        logger.info(f"Command: {' '.join(cmd)}")

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            pass_fds=(w_pipe,),
            env=env,
            preexec_fn=_spawn_preexec,
        )
        os.close(w_pipe)
        self.r_pipe = r_pipe

        try:
            self._wait_until_loaded(timeout=timeout, baseline_mib=baseline_vram)
        except Exception:
            self.stop()
            raise

    def _read_gpu_vram_mib(self) -> int:
        """Return MiB used on this daemon's GPU, or 0 if nvidia-smi fails."""
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(self.gpu_id),
                ]
            ).decode()
            return int(output.strip().split("\n")[0])
        except Exception:
            return 0

    def _wait_until_loaded(self, timeout: float, baseline_mib: int) -> None:
        """Wait for daemon to load by checking VRAM delta above baseline."""
        if baseline_mib <= 0:
            logger.warning("nvidia-smi unavailable; skipping VRAM-based dflash readiness check")
            time.sleep(2)
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"dflash daemon exited before loading (code {self.proc.returncode})"
                )
            return
        boot = time.time()
        while time.time() - boot < timeout:
            time.sleep(1)
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"dflash daemon exited before loading (code {self.proc.returncode})"
                )
            current = self._read_gpu_vram_mib()
            if current - baseline_mib >= LOADED_VRAM_DELTA_MIB:
                logger.info(
                    f"dflash daemon loaded, VRAM={current} MiB "
                    f"(delta {current - baseline_mib} MiB)"
                )
                return
        raise RuntimeError(
            f"dflash daemon failed to load within {timeout:.0f}s"
        )

    def is_running(self) -> bool:
        """Check if the daemon process is alive."""
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        """Stop the daemon process."""
        if not self.proc:
            return
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
        finally:
            if self.r_pipe >= 0:
                os.close(self.r_pipe)
            self.r_pipe = -1
            self.proc = None
            logger.info("dflash daemon stopped")

    def _send(self, cmd: str) -> None:
        """Send a command to the daemon's stdin."""
        if not self.is_running():
            raise RuntimeError("dflash daemon has exited")
        self.proc.stdin.write(cmd.encode("utf-8"))
        self.proc.stdin.flush()

    def _drain_sentinel(self) -> None:
        """Read the token stream pipe until -1 sentinel (command ACK)."""
        while True:
            tok = self._read_pipe_int32()
            if tok is None:
                break
            if tok == -1:
                break

    def _write_prompt_file(self, prompt_ids: list) -> str:
        """Write prompt token ids to a temp file and return its path."""
        fd, path = tempfile.mkstemp(suffix=".bin")
        try:
            with os.fdopen(fd, "wb") as f:
                for t in prompt_ids:
                    f.write(struct.pack("<i", int(t)))
        except Exception:
            try:
                os.unlink(path)
            except OSError:
                pass
            raise
        return path

    def _sampler_suffix(
        self,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        seed: Optional[int],
    ) -> str:
        """Build the daemon's shared `samp=` tail."""
        if temperature is None and top_p is None and top_k is None and seed is None:
            return ""
        temp = 0.0 if temperature is None else float(temperature)
        samp_top_p = 1.0 if top_p is None else float(top_p)
        samp_top_k = 0 if top_k is None else int(top_k)
        samp_seed = 0 if seed is None else int(seed)
        return f" samp={temp:.4f},{samp_top_p:.4f},{samp_top_k},1.0000,{samp_seed}"

    def read_next_token(self) -> Optional[int]:
        """Read one int32 token from the stream pipe.

        Blocking. Returns the token id, or None when the daemon emits the
        -1 sentinel or the pipe is closed. Callers iterating this from an
        asyncio context should wrap each call in `asyncio.to_thread`.
        """
        tok = self._read_pipe_int32()
        if tok is None:
            return None
        if tok == -1:
            return None
        return tok

    def send_generate_cmd(
        self,
        prompt_ids: list,
        n_gen: int,
        prefix_cache_slot: Optional[int] = None,
        snap_slot: Optional[int] = None,
        snap_pos: Optional[int] = None,
        temperature: Optional[float] = 0.8,
        top_p: Optional[float] = 0.9,
        top_k: Optional[int] = 40,
        seed: Optional[int] = None,
    ) -> str:
        """Send a generate command to the daemon and return the temp prompt path.

        The caller owns the returned path and must unlink it after the
        stream finishes (recommended: after read_next_token returns None).
        """
        path = self._write_prompt_file(prompt_ids)

        if prefix_cache_slot is not None:
            cmd = f"RESTORE {prefix_cache_slot} {path} {n_gen}"
        else:
            cmd = f"{path} {n_gen}"

        if snap_slot is not None and snap_pos is not None:
            cmd += f" snap={snap_pos}:{snap_slot}"

        cmd += self._sampler_suffix(temperature, top_p, top_k, seed)

        self._send(cmd + "\n")
        return path

    def restore_chain(
        self,
        thick_slot: int,
        thin_slots: list[int],
        prompt_path: str,
        n_gen: int,
        temperature: Optional[float] = 0.8,
        top_p: Optional[float] = 0.9,
        top_k: Optional[int] = 40,
        seed: Optional[int] = None,
    ) -> None:
        """Restore from a thick base plus zero or more thin layers."""
        if thick_slot < -1 or thick_slot >= 8:
            raise ValueError(f"Invalid thick snapshot slot: {thick_slot}")
        thin_str = "-"
        if thin_slots:
            for slot in thin_slots:
                if slot < 0 or slot >= 8:
                    raise ValueError(f"Invalid thin snapshot slot: {slot}")
            thin_str = ",".join(str(slot) for slot in thin_slots)
        cmd = f"RESTORE_CHAIN {thick_slot} {thin_str} {prompt_path} {n_gen}"
        cmd += self._sampler_suffix(temperature, top_p, top_k, seed)
        self._send(cmd + "\n")

    def send_restore_chain_cmd(
        self,
        prompt_ids: list,
        n_gen: int,
        thick_slot: int,
        thin_slots: list[int],
        temperature: Optional[float] = 0.8,
        top_p: Optional[float] = 0.9,
        top_k: Optional[int] = 40,
        seed: Optional[int] = None,
    ) -> str:
        """Send a RESTORE_CHAIN command and return the temp prompt path."""
        path = self._write_prompt_file(prompt_ids)
        self.restore_chain(
            thick_slot=thick_slot,
            thin_slots=thin_slots,
            prompt_path=path,
            n_gen=n_gen,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        )
        return path

    def generate(
        self,
        prompt_ids: list,
        n_gen: int,
        stop_ids: set = None,
        prefix_cache_slot: Optional[int] = None,
        prefix_cache_prefix_len: Optional[int] = None,
        snap_slot: Optional[int] = None,
        snap_pos: Optional[int] = None,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 40,
        seed: Optional[int] = None,
    ) -> list:
        """Generate tokens via the daemon (blocking, full-response).

        Prefer send_generate_cmd + read_next_token for streaming. This helper
        is kept for non-streaming callers and tests.
        """
        path = self.send_generate_cmd(
            prompt_ids=prompt_ids,
            n_gen=n_gen,
            prefix_cache_slot=prefix_cache_slot,
            snap_slot=snap_slot,
            snap_pos=snap_pos,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        )
        try:
            tokens = []
            while True:
                tok = self.read_next_token()
                if tok is None:
                    break
                if stop_ids and tok in stop_ids:
                    break
                tokens.append(tok)
                if len(tokens) >= n_gen:
                    break
            return tokens
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def compress(
        self,
        prompt_ids: list,
        drafter_path: str,
        keep_ratio: float = 0.05,
    ) -> list:
        """Compress prompt via PFlash drafter scoring.

        Parks target+draft, loads drafter, scores+compresses, then restores.

        Args:
            prompt_ids: Original prompt token IDs (drafter tokenizer)
            drafter_path: Path to drafter GGUF model
            keep_ratio: Fraction of tokens to keep (0.01-1.0)

        Returns:
            Compressed token IDs
        """
        fd, path = tempfile.mkstemp(suffix=".bin")
        try:
            with os.fdopen(fd, "wb") as f:
                for t in prompt_ids:
                    f.write(struct.pack("<i", int(t)))

            keep_x1000 = int(round(keep_ratio * 1000))
            self._send(f"compress {path} {keep_x1000} {drafter_path}\n")

            tokens = []
            while True:
                tok = self.read_next_token()
                if tok is None:
                    break
                tokens.append(tok)
            return tokens
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def restore(self, slot: int) -> bool:
        """Restore KV cache from a snapshot slot.

        Args:
            slot: Daemon slot ID (0-7)

        Returns:
            True if restore succeeded
        """
        self._send_expect_ack(f"RESTORE_SLOT {slot}\n")
        return True

    def free_snapshot(self, slot: int) -> None:
        """Free a snapshot slot.

        Args:
            slot: Daemon slot ID (0-7)
        """
        self._send_expect_ack(f"FREE_SNAPSHOT {slot}\n")

    def park_draft(self) -> None:
        """Park draft model to free ~3.3 GB VRAM."""
        self._send("park draft\n")
        self._drain_sentinel()

    def unpark_draft(self) -> None:
        """Restore parked draft model."""
        self._send("unpark draft\n")
        self._drain_sentinel()

    def park_target(self) -> None:
        """Park target model to free ~15 GB VRAM."""
        self._send("park target\n")
        self._drain_sentinel()

    def unpark_target(self) -> None:
        """Restore parked target model."""
        self._send("unpark target\n")
        self._drain_sentinel()

    def free_drafter(self) -> None:
        """Free the drafter model."""
        self._send("free drafter\n")
        self._drain_sentinel()

    def snapshot(self, slot: int) -> None:
        """Take a KV cache snapshot at current position.

        Sends a SNAPSHOT command. The daemon captures at cache.cur_pos.
        For prompt-boundary snapshots, prefer the inline `snap=pos:slot`
        suffix on send_generate_cmd so decode cannot advance cache.cur_pos
        before the snapshot is taken.

        Args:
            slot: Daemon slot ID (0-7)
        """
        if slot < 0 or slot >= 8:
            raise ValueError(f"Invalid snapshot slot: {slot}")
        self._send_expect_ack(f"SNAPSHOT {slot}\n")

    def snapshot_thin(self, slot: int, kv_start: int, kv_end: int) -> None:
        """Take a thin KV-only snapshot of range [kv_start, kv_end)."""
        if slot < 0 or slot >= 8:
            raise ValueError(f"Invalid snapshot slot: {slot}")
        self._send_expect_ack(f"SNAPSHOT_THIN {slot} {kv_start} {kv_end}\n")

    def save_snapshot(self, slot: int, path: str) -> None:
        """Serialize a snapshot slot to disk or tmpfs.

        The daemon frees the snapshot slot after a successful save.

        Args:
            slot: Daemon slot ID (0-7)
            path: Absolute disk path to write (e.g., /var/lib/grimoire/swap/slot-0.dfsn)
        """
        if slot < 0 or slot >= 8:
            raise ValueError(f"Invalid snapshot slot: {slot}")
        self._send_expect_ack(f"SAVE_SNAPSHOT {slot} {path}\n")

    def load_snapshot(self, slot: int, path: str) -> None:
        """Load a snapshot from disk into a slot (allocates VRAM).

        Args:
            slot: Daemon slot ID (0-7)
            path: Absolute disk path to read (e.g., /var/lib/grimoire/swap/slot-0.dfsn)
        """
        if slot < 0 or slot >= 8:
            raise ValueError(f"Invalid snapshot slot: {slot}")
        self._send_expect_ack(f"LOAD_SNAPSHOT {slot} {path}\n")


class PflashDaemon:
    """Manage a standalone pflash_daemon subprocess for compression-only.

    The pflash_daemon binary loads only the Qwen3.5-0.8B drafter GGUF
    (no target model). It accepts compress commands on stdin and emits
    compressed token IDs via a stream fd.
    """

    PFLASH_BIN = os.path.join(config.DFLASH_HOME, "pflash_daemon")

    def __init__(self, drafter_path: str, gpu_id: int = 0):
        self.drafter_path = drafter_path
        self.gpu_id = gpu_id
        self._proc: Optional[subprocess.Popen] = None
        self._pipe_r: Optional[int] = None  # read end of stream pipe
        self._pipe_w: Optional[int] = None  # write end (closed after spawn)
        self._temp_files: list[str] = []

    def start(self) -> None:
        """Spawn the pflash daemon. Creates a pipe for compressed token output."""
        pr, pw = os.pipe()
        self._pipe_r = pr

        cmd = [
            self.PFLASH_BIN,
            self.drafter_path,
            f"--stream-fd={pw}",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        _prepend_library_dir(env, config.DFLASH_HOME)

        logger.info(f"Starting pflash daemon on GPU {self.gpu_id}")
        logger.info(f"Command: {' '.join(cmd)}")

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(pw,),
            env=env,
            preexec_fn=_spawn_preexec,
        )
        os.close(pw)

        # Persistent stderr drain thread — without it the daemon blocks
        # on a full stderr pipe buffer during compress().
        import select
        import threading as _threading
        self._stderr_stop = _threading.Event()
        self._stderr_lines: list[str] = []

        def _drain_stderr():
            while not self._stderr_stop.is_set():
                r, _, _ = select.select([self._proc.stderr], [], [], 0.5)
                if r:
                    line = self._proc.stderr.readline()
                    if not line:
                        break
                    decoded = line.decode(errors="replace").strip()
                    self._stderr_lines.append(decoded)
                    logger.warning(f"[pflash-daemon] {decoded}")

        self._stderr_thread = _threading.Thread(target=_drain_stderr, daemon=True)
        self._stderr_thread.start()

        # Wait for ready line (with timeout)
        ready = False
        for _ in range(120):
            line = self._proc.stdout.readline()
            if not line:
                break
            decoded = line.decode(errors="replace").strip()
            logger.info(f"[pflash-daemon] {decoded}")
            if "ready" in decoded:
                ready = True
                break

        if not ready:
            # Log any stderr output captured before failure
            for err in self._stderr_lines:
                if "error" in err.lower() or "fail" in err.lower() or "oom" in err.lower() \
                   or "cudamalloc" in err.lower() or "out of memory" in err.lower():
                    logger.error(f"[pflash-daemon] {err}")
                else:
                    logger.warning(f"[pflash-daemon] {err}")
            raise RuntimeError(f"pflash daemon failed to start (see logs for details)")

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def compress(self, prompt_ids: list[int], drafter_path: str = "",
                 keep_ratio: float = 0.05) -> list[int]:
        if not self.is_running():
            raise RuntimeError("pflash daemon not running")
        if self._pipe_r is None:
            raise RuntimeError("pflash daemon pipe not initialized")

        fd, path = tempfile.mkstemp(suffix=".bin")
        self._temp_files.append(path)
        try:
            with os.fdopen(fd, "wb") as f:
                for t in prompt_ids:
                    f.write(struct.pack("<i", int(t)))

            keep_x1000 = int(round(keep_ratio * 1000))
            self._proc.stdin.write(f"compress {path} {keep_x1000}\n".encode())
            self._proc.stdin.flush()

            tokens: list[int] = []
            while True:
                raw = os.read(self._pipe_r, 4)
                if not raw or len(raw) < 4:
                    break
                tok = struct.unpack("<i", raw)[0]
                if tok == -1:
                    break
                tokens.append(tok)
            return tokens
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def stop(self) -> None:
        if self._proc:
            try:
                self._proc.stdin.write(b"quit\n")
                self._proc.stdin.flush()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None
        if self._pipe_r is not None:
            try:
                os.close(self._pipe_r)
            except OSError:
                pass
            self._pipe_r = None
        if hasattr(self, '_stderr_stop') and self._stderr_stop is not None:
            self._stderr_stop.set()
        if hasattr(self, '_stderr_thread') and self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2)
        for path in self._temp_files:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._temp_files.clear()

    def __del__(self):
        self.stop()


def _spawn_preexec():
    """Pre-exec function for daemon subprocess.

    Detaches into new session and sets PR_SET_PDEATHSIG so the kernel
    kills the daemon if the parent gateway crashes.
    """
    import ctypes
    import signal

    os.setsid()
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(
            1, signal.SIGTERM, 0, 0, 0
        )
    except Exception:
        pass
