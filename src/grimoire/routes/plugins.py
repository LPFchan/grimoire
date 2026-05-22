"""Plugin management routes — JSON API + standalone MCP-style page."""

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from grimoire.auth import require_api
from grimoire.history import identity_hash
from grimoire.plugins import plugin_manager, restore_plugin_states
from grimoire.settings import settings_store

logger = logging.getLogger(__name__)

router = APIRouter()

PLUGINS_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Plugins — Grimoire</title>
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  :root{--bg:#0e0e12;--surface:#16161e;--surface-hover:#1e1e2a;--border:#2a2a3a;--text:#e2e2ea;--text-muted:#8888a0;--accent:#ff8234;--accent-hover:#ff9a50;--green:#4ade80;--red:#f87171}
  body{background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,sans-serif;min-height:100vh}
  .header{display:flex;align-items:center;gap:12px;padding:24px 32px 0}
  .header svg{width:28px;height:28px}
  .header h1{font-size:1.6rem;font-weight:600}
  .content{max-width:1200px;padding:24px 32px}
  .grid{display:grid;gap:16px;grid-template-columns:repeat(auto-fill,minmax(min(32rem,calc(100dvw-4rem)),1fr))}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px;transition:border-color .15s}
  .card:hover{border-color:var(--accent)}
  .card-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
  .card-name{font-size:1.1rem;font-weight:600}
  .badge{font-size:.75rem;padding:3px 10px;border-radius:999px;font-weight:500}
  .badge-on{background:rgba(74,222,128,.15);color:var(--green)}
  .badge-off{background:rgba(248,113,113,.15);color:var(--red)}
  .card-desc{color:var(--text-muted);font-size:.9rem;line-height:1.5;margin-bottom:12px}
  .card-key{font-size:.8rem;color:var(--text-muted);font-family:ui-monospace,monospace;background:var(--bg);padding:4px 8px;border-radius:6px;display:inline-block}
  .empty{grid-column:1/-1;border:1px dashed var(--border);border-radius:12px;padding:32px;text-align:center;color:var(--text-muted);font-size:.95rem}
</style>
</head>
<body>
<div class="header">
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
  <h1>Plugins</h1>
</div>
<div class="content">
  <div class="grid" id="plugin-grid"></div>
</div>
<script>
async function load(){try{const r=await fetch('/stats/plugins');if(!r.ok)throw new Error(r.status);const p=await r.json(),g=document.getElementById('plugin-grid');g.innerHTML='';if(p.length===0){g.innerHTML='<div class="empty">No plugins loaded.</div>';return}p.forEach(pl=>{const c=document.createElement('div');c.className='card';c.innerHTML=\`
  <div class="card-header">
    <span class="card-name">\${pl.name}</span>
    <span class="badge \${pl.enabled?'badge-on':'badge-off'}">\${pl.enabled?'Active':'Disabled'}</span>
  </div>
  <div class="card-desc">\${pl.description}</div>
  <span class="card-key">\${pl.key}</span>
\`;g.appendChild(c)})}catch(e){document.getElementById('plugin-grid').innerHTML='<div class="empty">Failed to load plugins: '+e.message+'</div>'}}load()
</script>
</body>
</html>"""


@router.get("/stats/plugins")
async def plugins_stats(request: Request):
    _, user_hash = require_api(request)
    restore_plugin_states(user_hash, settings_store)
    return plugin_manager.get_all_info()


@router.get("/plugins")
async def plugins_page():
    return HTMLResponse(PLUGINS_HTML)


@router.patch("/stats/plugins/{key}")
async def plugins_toggle(key: str, request: Request):
    _, user_hash = require_api(request)
    body = await request.json()
    enabled = body.get("enabled")
    if enabled is None or not isinstance(enabled, bool):
        raise HTTPException(status_code=400, detail="'enabled' must be a boolean")
    result = plugin_manager.set_enabled(key, enabled)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Plugin '{key}' not found")
    settings_store.set(user_hash, f"plugin.{key}", json.dumps(enabled))
    return result
