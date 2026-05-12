"""Grimoire authentication — token validation and login endpoints."""

import hmac
import json
import logging
import urllib.parse

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from grimoire import config
from grimoire.history import identity_hash

logger = logging.getLogger(__name__)

router = APIRouter()

LOGIN_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grimoire Login</title><style>
body{margin:0;min-height:100vh;display:grid;place-items:center;background:#101014;color:#f6f3ea;font-family:system-ui,sans-serif}
form{display:grid;gap:14px;width:min(360px,calc(100vw - 32px));padding:28px;border:1px solid #2f2d3a;border-radius:18px;background:#191821}
input,button{font:inherit;border-radius:10px;padding:11px 13px}input{border:1px solid #403d4d;background:#111018;color:#fff}button{border:0;background:#e89b41;color:#15100a;font-weight:700;cursor:pointer}.err{color:#ff8c8c}
</style></head><body><form method="post" action="/login"><h1>Grimoire</h1><input name="key" type="password" placeholder="API key" autofocus><button>Login</button>{error}</form></body></html>"""


WEBUI_LOCALSTORAGE_CONFIG_KEY = "LlamaCppWebui.config"

LOGIN_BRIDGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Grimoire</title></head><body>
<noscript>Logged in. <a href="/">Open chat</a>.</noscript>
<script>
try {{
  var k = "{storage_key}";
  var c = {{}};
  try {{ c = JSON.parse(localStorage.getItem(k) || "{{}}") || {{}}; }} catch (e) {{ c = {{}}; }}
  c.apiKey = {key_json};
  localStorage.setItem(k, JSON.stringify(c));
}} catch (e) {{}}
location.replace("/");
</script></body></html>"""


def _request_token(request):
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return request.headers.get("x-grimoire-token")


def _valid_cookie(request):
    token = request.cookies.get(config.COOKIE_NAME, "")
    return bool(config.API_KEY and token and hmac.compare_digest(token, config.API_KEY))


def require_api(request):
    """Require the shared API key for public API and history endpoints."""
    if not config.API_KEY:
        if not config.ALLOW_ANONYMOUS:
            raise HTTPException(status_code=503, detail="GRIMOIRE_API_KEY is required")
        return "anonymous", identity_hash("anonymous")
    token = _request_token(request)
    if token and hmac.compare_digest(token, config.API_KEY):
        return token, identity_hash(token)
    if _valid_cookie(request):
        return config.API_KEY, identity_hash(config.API_KEY)
    raise HTTPException(status_code=401, detail="Invalid API token")


def require_admin(request):
    """Require the shared admin token for mutating management endpoints."""
    if not config.ADMIN_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="GRIMOIRE_ADMIN_TOKEN is required for management endpoints",
        )
    token = _request_token(request)
    if not token or not hmac.compare_digest(token, config.ADMIN_TOKEN):
        cookie = request.cookies.get(config.COOKIE_NAME, "")
        if cookie and hmac.compare_digest(cookie, config.ADMIN_TOKEN):
            return cookie, identity_hash(cookie)
        raise HTTPException(status_code=401, detail="Invalid admin token")
    return token, identity_hash(token)


def _require_login_enabled():
    if not config.API_KEY and not config.ALLOW_ANONYMOUS:
        raise HTTPException(status_code=503, detail="GRIMOIRE_API_KEY is required")


def _render_login_html(error=""):
    return LOGIN_HTML.replace("{error}", error)


def _render_login_bridge_html(key):
    return LOGIN_BRIDGE_HTML.format(
        storage_key=WEBUI_LOCALSTORAGE_CONFIG_KEY,
        key_json=json.dumps(key),
    )


@router.get("/login")
async def login_page():
    if not config.API_KEY and not config.ALLOW_ANONYMOUS:
        return HTMLResponse(
            _render_login_html('<p class="err">Set GRIMOIRE_API_KEY or GATEWAY_API_KEY before login.</p>'),
            status_code=503,
        )
    return HTMLResponse(_render_login_html(""))


@router.post("/login")
async def login_submit(request: Request):
    _require_login_enabled()
    form = urllib.parse.parse_qs((await request.body()).decode("utf-8"))
    key = (form.get("key") or [""])[0]
    if config.API_KEY and not hmac.compare_digest(key, config.API_KEY):
        return HTMLResponse(_render_login_html('<p class="err">Invalid key</p>'), status_code=401)
    response = HTMLResponse(_render_login_bridge_html(key))
    response.set_cookie(config.COOKIE_NAME, key, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 30)
    return response
