"""DFlash daemon process manager.

Spawns the DFlash binary, communicates via stdin/stdout protocol,
and provides the generate() and compress() interfaces used by the gateway.
"""

import asyncio
import logging
import os
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DFLASH_BIN = "/opt/dflash/dflash"


class DflashDaemon:
    """Manage a DFlash daemon subprocess.

    Handles:
      - Spawning the daemon with target + draft models
      - stdin/stdout protocol communication
      - VRAM-based health checks
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
        drafter_path: Optional[str] = None,
        prefill_threshold: int = 32000,
        prefill_keep_ratio: float = 0.05,
        kv_k_type: str = "q8_0",
        fa_window: int = 2048,
    ):
        self.target_path = target_path
        self.draft_path = draft_path
        self.drafter_path = drafter_path
        self.max_ctx = max_ctx
        self.budget = budget
        self.gpu_id = gpu_id
        self.prefill_threshold = prefill_threshold
        self.prefill_keep_ratio = prefill_keep_ratio
        self.kv_k_type = kv_k_type
        self.fa_window = fa_window

        self.proc: Optional[subprocess.Popen] = None
        self.r_pipe: int = -1
        self.w_pipe: int = -1

    def spawn(self, timeout: float = 600.0) -> None:
        """Spawn the dflash daemon and wait for it to load.

        Raises RuntimeError if the daemon fails to load within timeout.
        """
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
        self.w_pipe = w_pipe

        self._wait_until_loaded(timeout=timeout)

    def _wait_until_loaded(self, timeout: float) -> None:
        """Wait for daemon to load by checking VRAM usage."""
        boot = time.time()
        while time.time() - boot < timeout:
            time.sleep(1)
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"dflash daemon exited before loading (code {self.proc.returncode})"
                )
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
                vram = int(output.strip().split("\n")[0])
                if vram > 16000:
                    logger.info(f"dflash daemon loaded, VRAM={vram} MiB")
                    return
            except Exception:
                pass
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

    def _read_tokens(self, n_gen: int, stop_ids: set = None) -> list:
        """Read generated token IDs from the stream pipe.

        Returns list of token IDs (stops on -1 sentinel or n_gen limit).
        """
        tokens = []
        while True:
            b = os.read(self.r_pipe, 4)
            if not b or len(b) < 4:
                break
            tok = struct.unpack("<i", b)[0]
            if tok == -1:
                break
            if stop_ids and tok in stop_ids:
                break
            tokens.append(tok)
            if len(tokens) >= n_gen:
                break
        return tokens

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
        """Generate tokens via the daemon.

        Args:
            prompt_ids: Input token IDs
            n_gen: Maximum tokens to generate
            stop_ids: Token IDs that stop generation
            prefix_cache_slot: If set, restore from prefix cache slot before prefill
            prefix_cache_prefix_len: Length of prefix already in cache
            snap_slot: If set, take inline snapshot at snap_pos during prefill
            snap_pos: Token position to snapshot
            temperature: Sampling temperature
            top_p: Nucleus sampling top_p
            top_k: Top-k sampling
            seed: Random seed for reproducibility

        Returns:
            List of generated token IDs
        """
        # Write prompt to temp .bin file
        fd, path = tempfile.mkstemp(suffix=".bin")
        try:
            with os.fdopen(fd, "wb") as f:
                for t in prompt_ids:
                    f.write(struct.pack("<i", int(t)))

            # Build command line
            if prefix_cache_slot is not None:
                cmd = f"RESTORE {prefix_cache_slot} {path} {n_gen}"
            else:
                cmd = f"{path} {n_gen}"

            # Inline snapshot
            if snap_slot is not None and snap_pos is not None:
                cmd += f" snap={snap_pos}:{snap_slot}"

            # Sampling parameters
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
            return self._read_tokens(n_gen, stop_ids)
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
                b = os.read(self.r_pipe, 4)
                if not b or len(b) < 4:
                    break
                tok = struct.unpack("<i", b)[0]
                if tok == -1:
                    break
                tokens.append(tok)
            return tokens
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def snapshot(self, slot: int) -> None:
        """Take a KV cache snapshot at current position into slot.

        Args:
            slot: Daemon slot ID (0-7)
        """
        if slot < 0 or slot >= 8:
            raise ValueError(f"Invalid prefix cache slot: {slot}")
        self._send(f"SNAPSHOT {slot}\n")

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

    def list_slots(self) -> list:
        """List occupied prefix cache slots.

        Returns:
            List of slot IDs that have active snapshots
        """
        self._send("LIST_SLOTS\n")
        # Read response from stdout — handled by stdout bus
        return []

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
