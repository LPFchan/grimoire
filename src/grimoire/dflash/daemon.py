"""PFlash daemon process manager.

Manages the standalone pflash_daemon subprocess for compression-only
prompt prefill. The daemon loads a Qwen3.5-0.8B drafter GGUF and
accepts compress commands on stdin.
"""

import logging
import os
import struct
import subprocess
import tempfile
from typing import Optional

from grimoire import config

logger = logging.getLogger(__name__)


def _prepend_library_dir(env: dict[str, str], path: str) -> None:
    existing = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p and p != path]
    env["LD_LIBRARY_PATH"] = ":".join([path, *existing]) if path else ":".join(existing)


class PflashDaemon:
    """Manage a standalone pflash_daemon subprocess for compression-only.

    The pflash_daemon binary loads only the Qwen3.5-0.8B drafter GGUF
    (no target model). It accepts compress commands on stdin and emits
    compressed token IDs via a stream fd.
    """

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
            config.PFLASH_DAEMON_BIN,
            self.drafter_path,
            f"--stream-fd={pw}",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        _prepend_library_dir(env, config.PFLASH_LIB_DIR)

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
