"""Preset management routes."""

import logging

from fastapi import APIRouter, HTTPException, Request

from grimoire.auth import require_admin, require_api
from grimoire.presets import presets
from grimoire.registry import registry

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_manager():
    from grimoire.entrypoint import manager
    return manager


@router.get("/presets")
async def list_presets(request: Request):
    require_api(request)
    return {"presets": presets.list(), "active": presets.get_active_name()}


@router.get("/presets/{name}")
async def get_preset(name: str, request: Request):
    require_api(request)
    preset = presets.get(name)
    if preset is None:
        raise HTTPException(status_code=404, detail=f"Preset '{name}' not found")
    mgr = _get_manager()
    loaded = [m for m in preset.get("models", []) if mgr.get_active(m)]
    return {
        "name": name,
        "description": preset.get("description", ""),
        "models": preset.get("models", []),
        "fixed": preset.get("fixed", {}),
        "gpus": preset.get("gpus"),
        "loaded": loaded,
        "active": presets.get_active_name() == name,
    }


@router.put("/presets/{name}")
async def upsert_preset(name: str, request: Request):
    require_admin(request)
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")

    description = data.get("description", "")
    if not isinstance(description, str):
        raise HTTPException(status_code=400, detail="'description' must be a string")

    models = data.get("models")
    if models is None or not isinstance(models, list):
        raise HTTPException(status_code=400, detail="'models' is required and must be an array")
    if not all(isinstance(m, str) for m in models):
        raise HTTPException(status_code=400, detail="'models' must be an array of strings")

    fixed = data.get("fixed")
    if fixed is not None:
        if not isinstance(fixed, dict):
            raise HTTPException(status_code=400, detail="'fixed' must be an object")
        mgr = _get_manager()
        for k, v in fixed.items():
            if not isinstance(k, str):
                raise HTTPException(status_code=400, detail=f"Fixed key '{k}' must be a string")
            if not isinstance(v, int) or v < 0 or v >= mgr.gpu_count:
                raise HTTPException(status_code=400, detail=f"Fixed GPU ID '{v}' for '{k}' is invalid")

    for model_name in models:
        if not registry.get(model_name):
            raise HTTPException(status_code=400, detail=f"Model '{model_name}' not found in registry")

    gpus = data.get("gpus")
    if gpus is not None:
        if not isinstance(gpus, list):
            raise HTTPException(status_code=400, detail="'gpus' must be an array")
        if not all(isinstance(g, int) for g in gpus):
            raise HTTPException(status_code=400, detail="'gpus' must be an array of integers")
        mgr = _get_manager()
        for g in gpus:
            if g < 0 or g >= mgr.gpu_count:
                raise HTTPException(status_code=400, detail=f"GPU ID {g} in 'gpus' is out of range")
        if fixed:
            for model, gpu_id in fixed.items():
                if gpu_id not in gpus:
                    raise HTTPException(status_code=400,
                        detail=f"Model '{model}' pinned to GPU {gpu_id} not in 'gpus' {gpus}")

    presets.upsert(name, description, models, fixed if fixed else {}, gpus=gpus)
    return {"status": "saved", "name": name}


@router.delete("/presets/{name}")
async def delete_preset(name: str, request: Request):
    require_admin(request)
    try:
        ok = presets.delete(name)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if not ok:
        raise HTTPException(status_code=404, detail=f"Preset '{name}' not found")
    return {"status": "deleted", "name": name}


@router.post("/presets/{name}/activate")
async def activate_preset(name: str, request: Request):
    require_admin(request)
    mgr = _get_manager()
    try:
        result = await presets.activate(name, mgr, registry)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Preset '{name}' not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result
