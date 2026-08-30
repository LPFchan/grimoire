"""Cross-origin access for the standalone dashboard at dash.lost.plus.

The dashboard used to be a page inside the chat webui, so it was same-origin
with the gateway. Served from its own host it is not, and the browser blocks the
request unless the gateway names that origin. These tests pin the allowlist
behaviour — especially that it stays closed by default and never echoes an
origin it was not given.
"""

import importlib

import pytest
from fastapi.testclient import TestClient

ALLOWED = "https://dash.lost.plus"
DENIED = "https://evil.example"


def _client(monkeypatch, origins):
    """Reimport config + proxy_app so the module-level allowlist is re-read."""
    if origins is None:
        monkeypatch.delenv("GRIMOIRE_CORS_ORIGINS", raising=False)
    else:
        monkeypatch.setenv("GRIMOIRE_CORS_ORIGINS", origins)

    config = importlib.reload(importlib.import_module("grimoire.config"))
    proxy_app = importlib.reload(importlib.import_module("grimoire.proxy_app"))
    # Without CORS configured a preflight falls through to the catch-all
    # forwarder, which has no live manager here. We only care about the headers,
    # so let that surface as a 500 rather than an exception.
    return TestClient(proxy_app.app, raise_server_exceptions=False), config


def _preflight(client, origin, method="GET"):
    return client.options(
        "/stats/dashboard",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": method,
            "Access-Control-Request-Headers": "authorization",
        },
    )


@pytest.fixture(autouse=True)
def _restore_modules():
    """Leave the imported modules matching the ambient env for other tests."""
    yield
    importlib.reload(importlib.import_module("grimoire.config"))
    importlib.reload(importlib.import_module("grimoire.proxy_app"))


def test_origins_parse_and_strip(monkeypatch):
    _, config = _client(monkeypatch, " https://dash.lost.plus/ , https://other.test ")
    assert config.CORS_ORIGINS == ["https://dash.lost.plus", "https://other.test"]


def test_unset_means_no_origins(monkeypatch):
    _, config = _client(monkeypatch, None)
    assert config.CORS_ORIGINS == []


def test_closed_by_default(monkeypatch):
    """Without the env var the gateway stays same-origin only, as before.

    Nothing answers the preflight, so no permission is granted to anyone.
    """
    client, _ = _client(monkeypatch, None)
    response = _preflight(client, ALLOWED)
    assert "access-control-allow-origin" not in response.headers


def test_allowed_origin_gets_permission(monkeypatch):
    client, _ = _client(monkeypatch, ALLOWED)
    response = _preflight(client, ALLOWED)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED


def test_unlisted_origin_is_refused(monkeypatch):
    client, _ = _client(monkeypatch, ALLOWED)
    response = _preflight(client, DENIED)
    assert response.headers.get("access-control-allow-origin") != DENIED


def test_card_order_put_is_permitted(monkeypatch):
    """The dashboard saves card arrangement with PUT /stats/card-order."""
    client, _ = _client(monkeypatch, ALLOWED)
    response = _preflight(client, ALLOWED, method="PUT")
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED


def test_cookies_are_not_invited(monkeypatch):
    """Auth is a bearer token; the gw_session cookie must not ride along."""
    client, _ = _client(monkeypatch, ALLOWED)
    response = _preflight(client, ALLOWED)
    assert "access-control-allow-credentials" not in response.headers
