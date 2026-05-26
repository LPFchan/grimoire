"""Safe model download helpers."""

import ipaddress
import os
import socket
import urllib.parse
import urllib.request

DEFAULT_MAX_BYTES = 80 * 1024 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name, default):
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


ALLOW_HTTP = _env_bool("GRIMOIRE_ALLOW_HTTP_INGEST", False)
ALLOW_PRIVATE = _env_bool("GRIMOIRE_ALLOW_PRIVATE_INGEST", False)
MAX_BYTES = _env_int("GRIMOIRE_INGEST_MAX_BYTES", DEFAULT_MAX_BYTES)


def model_filename_from_url(model_url):
    """Return a safe local filename derived from a model URL path."""
    parsed = urllib.parse.urlparse(model_url)
    filename = os.path.basename(urllib.parse.unquote(parsed.path))
    if not filename or filename in {".", ".."} or filename != os.path.basename(filename):
        raise ValueError("Model URL must end with a valid filename")
    return filename


def validate_ingest_url(model_url):
    """Validate an ingest URL before opening a network connection."""
    parsed = urllib.parse.urlparse(model_url)
    allowed_schemes = {"https"}
    if ALLOW_HTTP:
        allowed_schemes.add("http")
    if parsed.scheme not in allowed_schemes:
        raise ValueError("Model URL must use https")
    if not parsed.hostname:
        raise ValueError("Model URL must include a hostname")

    if not ALLOW_PRIVATE:
        for result in socket.getaddrinfo(parsed.hostname, parsed.port, type=socket.SOCK_STREAM):
            address = result[4][0]
            ip = ipaddress.ip_address(address)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
                raise ValueError("Model URL resolves to a private or non-routable address")

    model_filename_from_url(model_url)
    return parsed


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validate redirect targets before urllib follows them."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_ingest_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_model_file(model_url, target_path, max_bytes=MAX_BYTES, timeout=30):
    """Download a model URL to target_path atomically with size and URL checks."""
    validate_ingest_url(model_url)

    request = urllib.request.Request(model_url, headers={"User-Agent": "grimoire/0.1"})
    opener = urllib.request.build_opener(SafeRedirectHandler)
    tmp_path = f"{target_path}.part"

    try:
        with opener.open(request, timeout=timeout) as response:
            final_url = response.geturl()
            validate_ingest_url(final_url)

            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise ValueError(f"Model download exceeds limit of {max_bytes} bytes")

            total = 0
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"Model download exceeds limit of {max_bytes} bytes")
                    f.write(chunk)

        os.replace(tmp_path, target_path)
        return target_path
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


# ---------------------------------------------------------------------------
# HuggingFace URL parsing
# ---------------------------------------------------------------------------


def parse_hf_url(raw_url: str):
    """Resolve a HuggingFace URL to a concrete download URL and filename."""
    parsed = urllib.parse.urlparse(raw_url)
    params = urllib.parse.parse_qs(parsed.query)
    segments = [s for s in parsed.path.strip("/").split("/") if s]

    # Case 1: ?show_file_info=foo.gguf OR ?download=foo.gguf — extract filename, branch from path
    filename = (params.get("show_file_info") or [None])[0]
    maybe_download = (params.get("download") or [None])[0]
    if not filename and maybe_download and maybe_download.lower().endswith(".gguf"):
        filename = maybe_download
    if filename and filename.lower().endswith(".gguf"):
        if len(segments) >= 2:
            user, repo = segments[0], segments[1]
            branch = "main"
            for i, s in enumerate(segments):
                if s in ("blob", "tree") and i + 1 < len(segments):
                    branch = segments[i + 1]
                    break
            dl_url = f"https://huggingface.co/{user}/{repo}/resolve/{branch}/{filename}"
            return dl_url, filename

    # Case 2: /user/repo/blob/<branch>/<subpath...>/<filename>.gguf or .../resolve/...
    fn_check = parsed.path.lower().rstrip("/")
    if fn_check.endswith(".gguf") or any(seg.lower().endswith(".gguf") for seg in fn_check.split("/")):
        idx = next((i for i, p in enumerate(segments) if p in ("blob", "tree", "resolve")), None)
        if idx is not None and len(segments) > idx + 2 and segments[-1].lower().endswith(".gguf"):
            user, repo = segments[0], segments[1]
            branch = segments[idx + 1]
            filename = segments[-1]
            subpath = "/".join(segments[idx + 2:-1])
            if subpath:
                dl_url = f"https://huggingface.co/{user}/{repo}/resolve/{branch}/{subpath}/{filename}"
            else:
                dl_url = f"https://huggingface.co/{user}/{repo}/resolve/{branch}/{filename}"
            return dl_url, filename

    # Case 3: Direct URL ending in .gguf — use as-is
    fn = os.path.basename(urllib.parse.unquote(parsed.path))
    if fn.lower().endswith(".gguf"):
        return raw_url, fn

    raise ValueError("URL does not reference a .gguf file")


# ---------------------------------------------------------------------------
# Download with progress tracking
# ---------------------------------------------------------------------------


class _DownloadCancelled(Exception):
    pass


def download_model_file_with_progress(model_url, target_path, progress_dict, max_bytes=MAX_BYTES, timeout=30):
    """Download a model URL with progress tracking and cancellation support."""
    validate_ingest_url(model_url)

    request = urllib.request.Request(model_url, headers={"User-Agent": "grimoire/0.1"})
    opener = urllib.request.build_opener(SafeRedirectHandler)
    tmp_path = f"{target_path}.part"

    progress_dict["tmp_path"] = tmp_path
    progress_dict["target_path"] = target_path

    try:
        with opener.open(request, timeout=timeout) as response:
            final_url = response.geturl()
            validate_ingest_url(final_url)

            content_length = response.headers.get("Content-Length")
            total_bytes = int(content_length) if content_length else None
            if content_length and total_bytes and total_bytes > max_bytes:
                raise ValueError(f"Model download exceeds limit of {max_bytes} bytes")

            progress_dict["total_bytes"] = total_bytes
            progress_dict["downloaded_bytes"] = 0
            total = 0

            with open(tmp_path, "wb") as f:
                while True:
                    if progress_dict.get("cancelled"):
                        raise _DownloadCancelled()

                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"Model download exceeds limit of {max_bytes} bytes")
                    f.write(chunk)
                    progress_dict["downloaded_bytes"] = total

        os.replace(tmp_path, target_path)
    except _DownloadCancelled:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise
