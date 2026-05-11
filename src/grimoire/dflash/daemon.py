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

logger = logging.getLogger(__name__)

DFLASH_BIN = "/opt/dflash/dflash"

# VRAM delta (MiB) above the pre-spawn baseline that signals the daemon
# has loaded weights. Target+draft for a 27B model is ~18 GB; we use a
# conservative threshold so other processes on the same GPU can't satisfy
# it.
LOADED_VRAM_DELTA_MIB = 12000


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
        draft_path: str,
        max_ctx: int = 16384,
        budget: int = 22,
        gpu_id: int = 0,
        prefill_threshold: int = 32000,
        prefill_keep_ratio: float = 0.05,
        kv_k_type: str = "q8_0",
        fa_window: int = 2048,
    ):
        self.target_path = target_path
        self.draft_path = draft_path
        self.max_ctx = max_ctx
        self.budget = budget
        self.gpu_id = gpu_id
        self.prefill_threshold = prefill_threshold
        self.prefill_keep_ratio = prefill_keep_ratio
        self.kv_k_type = kv_k_type
        self.fa_window = fa_window

        self.proc: Optional[subprocess.Popen] = None
        self.r_pipe: int = -1

    def spawn(self, timeout: float = 600.0) -> None:
        """Spawn the dflash daemon and wait for it to load.

        Raises RuntimeError if the daemon fails to load within timeout.
        """
        baseline_vram = self._read_gpu_vram_mib()

        r_pipe, w_pipe = os.pipe()
        stream_fd = w_pipe

        cmd = [
            DFLASH_BIN,
            self.target_path,
            self.draft_path,
            "--daemon",
            "--fast-rollback",
            "--ddtree",
            f"--ddtree-budget={self.budget}",
            f"--max-ctx={self.max_ctx}",
            f"--stream-fd={stream_fd}",
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        env["DFLASH27B_KV_K"] = self.kv_k_type
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

        self._wait_until_loaded(timeout=timeout, baseline_mib=baseline_vram)

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
            b = os.read(self.r_pipe, 4)
            if not b or len(b) < 4:
                break
            if struct.unpack("<i", b)[0] == -1:
                break

    def read_next_token(self) -> Optional[int]:
        """Read one int32 token from the stream pipe.

        Blocking. Returns the token id, or None when the daemon emits the
        -1 sentinel or the pipe is closed. Callers iterating this from an
        asyncio context should wrap each call in `asyncio.to_thread`.
        """
        b = os.read(self.r_pipe, 4)
        if not b or len(b) < 4:
            return None
        tok = struct.unpack("<i", b)[0]
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

        if prefix_cache_slot is not None:
            cmd = f"RESTORE {prefix_cache_slot} {path} {n_gen}"
        else:
            cmd = f"{path} {n_gen}"

        if snap_slot is not None and snap_pos is not None:
            cmd += f" snap={snap_pos}:{snap_slot}"

        samp_parts = []
        if temperature is not None:
            samp_parts.append(f"temp={temperature:.4f}")
        if top_p is not None:
            samp_parts.append(f"top_p={top_p:.4f}")
        if top_k is not None:
            samp_parts.append(f"top_k={top_k}")
        if seed is not None:
            samp_parts.append(f"seed={seed}")
        if samp_parts:
            cmd += " " + " ".join(samp_parts)

        self._send(cmd + "\n")
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
        self._send(f"RESTORE_SLOT {slot}\n")
        return True

    def free_snapshot(self, slot: int) -> None:
        """Free a snapshot slot.

        Args:
            slot: Daemon slot ID (0-7)
        """
        self._send(f"FREE_SNAPSHOT {slot}\n")

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
        self._send(f"SNAPSHOT {slot}\n")

    def save_snapshot(self, slot: int, path: str) -> None:
        """Serialize a snapshot slot to disk, freeing the VRAM.

        Args:
            slot: Daemon slot ID (0-7)
            path: Absolute disk path to write (e.g., /var/lib/grimoire/swap/slot-0.dfsn)
        """
        if slot < 0 or slot >= 8:
            raise ValueError(f"Invalid snapshot slot: {slot}")
        self._send(f"SAVE_SNAPSHOT {slot} {path}\n")

    def load_snapshot(self, slot: int, path: str) -> None:
        """Load a snapshot from disk into a slot (allocates VRAM).

        Args:
            slot: Daemon slot ID (0-7)
            path: Absolute disk path to read (e.g., /var/lib/grimoire/swap/slot-0.dfsn)
        """
        if slot < 0 or slot >= 8:
            raise ValueError(f"Invalid snapshot slot: {slot}")
        self._send(f"LOAD_SNAPSHOT {slot} {path}\n")


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
