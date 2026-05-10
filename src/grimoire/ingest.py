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
