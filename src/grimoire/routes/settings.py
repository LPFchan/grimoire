"""Settings management routes — server-backed config storage."""

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from grimoire.auth import require_api
from grimoire.history import identity_hash
from grimoire.settings import settings_store

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/settings")
async def get_settings(request: Request):
    _, user_hash = require_api(request)
    return settings_store.get_all(user_hash)


@router.put("/settings")
async def put_settings(request: Request):
    _, user_hash = require_api(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")
    kv = {k: (json.dumps(v) if not isinstance(v, str) else v) for k, v in body.items()}
    settings_store.set_many(user_hash, kv)
    return {"status": "ok"}


@router.post("/settings/migrate")
async def migrate_settings(request: Request):
    _, user_hash = require_api(request)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")
    kv = {k: (json.dumps(v) if not isinstance(v, str) else v) for k, v in body.items()}
    settings_store.import_bulk(user_hash, kv)
    return {"status": "migrated", "count": len(kv)}


@router.delete("/settings/{key:path}")
async def delete_setting(key: str, request: Request):
    _, user_hash = require_api(request)
    settings_store.delete(user_hash, key)
    return {"status": "deleted"}
