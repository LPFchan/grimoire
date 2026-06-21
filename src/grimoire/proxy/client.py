"""Shared, connection-pooled httpx client for backend proxying.

Creating a fresh ``httpx.AsyncClient`` per request opens and tears down a TCP
connection to the local llama-server every time (no keepalive), which caps
high-RPS endpoints (rerank, embeddings) at ~30 req/s. A single pooled client
reused across requests is ~3x faster for those workloads. Initialized in the
app lifespan; both the entrypoint proxies and proxy.llama use it.
"""

import httpx

_client: httpx.AsyncClient | None = None


def init_proxy_client() -> httpx.AsyncClient:
    """Create the shared client (idempotent). Call once at app startup."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=None,
            limits=httpx.Limits(
                max_connections=256,
                max_keepalive_connections=128,
                keepalive_expiry=30.0,
            ),
        )
    return _client


async def close_proxy_client() -> None:
    """Close the shared client at app shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def get_proxy_client() -> httpx.AsyncClient:
    """Return the shared client. Raises if not initialized."""
    if _client is None:
        raise RuntimeError("proxy client not initialized")
    return _client
