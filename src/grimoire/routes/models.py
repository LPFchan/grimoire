"""Model management route handlers (/v1/models, /models, /switch, /stop, /props, /ingest)."""

import asyncio
import logging
import os
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from grimoire.auth import require_api, require_admin
from grimoire.config import DEFAULT_CTX_SIZE, DEFAULT_GENERATION_PARAMS
from grimoire.ingest import download_model_file, model_filename_from_url
from grimoire.registry import MODELS_DIR, registry

_REFERENCE_FIELDS = ("file", "mmproj", "mtp-head", "spec-draft-model", "draft", "drafter")
_MAX_GGUF_UPLOAD_BYTES = int(os.environ.get("GRIMOIRE_MAX_GGUF_UPLOAD_BYTES", 40 * 1024**3))

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_manager():
    from grimoire.entrypoint import manager
    return manager


def _synthetic_props(model_name=None):
    cfg = registry.get(model_name) if model_name else None
    capabilities = (cfg or {}).get("capabilities", []) or []
    has_vision = "multimodal" in capabilities or "vision" in capabilities
    return {
        "default_generation_settings": {
            "id": 0,
            "id_task": -1,
            "n_ctx": (cfg or {}).get("ctx-size", DEFAULT_CTX_SIZE),
            "speculative": False,
            "is_processing": False,
            "params": dict(DEFAULT_GENERATION_PARAMS),
            "prompt": "",
            "next_token": {
                "has_next_token": False,
                "has_new_line": False,
                "n_remain": 0,
                "n_decoded": 0,
                "stopping_word": "",
            },
        },
        "total_slots": (cfg or {}).get("parallel", 1),
        "model_path": (cfg or {}).get("file", ""),
        "role": "router",
        "modalities": {"vision": bool(has_vision), "audio": False},
        "chat_template": "",
        "bos_token": "",
        "eos_token": "",
        "build_info": "grimoire",
    }


def _model_payload_name(payload):
    if not isinstance(payload, dict):
        return None
    name = payload.get("model")
    return name if isinstance(name, str) and name else None


def _containment_check(filename: str) -> str:
    real_models = os.path.realpath(MODELS_DIR)
    resolved = os.path.realpath(os.path.join(MODELS_DIR, filename))
    if os.path.commonpath([real_models, resolved]) != real_models:
        raise HTTPException(status_code=400, detail="Path escapes models directory")
    return resolved


def _file_referenced_by(filename: str) -> list[str]:
    models = registry.snapshot().get("models", {})
    refs = []
    for name, cfg in models.items():
        for field in _REFERENCE_FIELDS:
            val = (cfg or {}).get(field)
            if val and str(val) == filename:
                refs.append(name)
                break
    return refs


def _scan_gguf_files() -> list[dict]:
    gguf_dir = os.path.join(MODELS_DIR, "gguf")
    if not os.path.isdir(gguf_dir):
        return []
    models = registry.snapshot().get("models", {})
    used_map: dict[str, list[str]] = {}
    for name, cfg in models.items():
        for field in _REFERENCE_FIELDS:
            val = (cfg or {}).get(field)
            if val and isinstance(val, str):
                used_map.setdefault(val, []).append(name)
    files = []
    for entry in sorted(os.listdir(gguf_dir)):
        if not entry.lower().endswith(".gguf"):
            continue
        rel_path = f"gguf/{entry}"
        full_path = os.path.join(gguf_dir, entry)
        try:
            size = os.path.getsize(full_path)
        except OSError:
            size = 0
        used_by = used_map.get(rel_path)
        files.append({"filename": rel_path, "size_bytes": size, "used_by": used_by or None})
    return files


def _validate_model_config(data: dict) -> None:
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")
    file_val = data.get("file")
    if not file_val or not isinstance(file_val, str):
        raise HTTPException(status_code=400, detail="'file' is required and must be a string")
    for field in ("ctx-size", "predict", "parallel", "n-gpu-layers",
                  "image-min-tokens", "image-max-tokens"):
        val = data.get(field)
        if val is not None and (not isinstance(val, int) or val <= 0):
            raise HTTPException(status_code=400, detail=f"'{field}' must be a positive integer")
    for field in ("cache-type-k", "cache-type-v"):
        val = data.get(field)
        if val is not None and val not in ("q8_0", "q4_0", "turbo4", "f16"):
            raise HTTPException(status_code=400,
                                detail=f"'{field}' must be one of: q8_0, q4_0, turbo4, f16")
    cost = data.get("cost")
    if cost is not None:
        if not isinstance(cost, dict):
            raise HTTPException(status_code=400, detail="'cost' must be an object")
        for key in ("input", "output", "cache_read"):
            val = cost.get(key)
            if val is not None and (not isinstance(val, (int, float)) or val < 0):
                raise HTTPException(status_code=400,
                                    detail=f"'cost.{key}' must be a non-negative number")
    extra_args = data.get("extra-args")
    if extra_args is not None:
        if not isinstance(extra_args, list) or not all(isinstance(a, str) for a in extra_args):
            raise HTTPException(status_code=400, detail="'extra-args' must be an array of strings")
    spec_type = data.get("speculative-type")
    if spec_type == "mtp":
        if not data.get("mtp-head"):
            raise HTTPException(status_code=400, detail="'mtp-head' is required when speculative-type is 'mtp'")
    if spec_type == "dflash":
        draft = data.get("spec-draft-model") or data.get("draft")
        if not draft:
            raise HTTPException(status_code=400, detail="'spec-draft-model' is required when speculative-type is 'dflash'")
    if data.get("pflash") and not data.get("drafter"):
        raise HTTPException(status_code=400, detail="'drafter' is required when pflash is enabled")


@router.get("/v1/models")
async def get_v1_models(request: Request):
    """Return all registry models in OpenAI-compatible + llama.cpp router shape."""
    require_api(request)
    manager = _get_manager()
    data = registry.list_metadata()
    active_names = set(manager.list_active())
    for item in data:
        name = item["id"]
        cfg = registry.get(name) or {}
        item["active"] = name in active_names
        item["status"] = {"value": manager.get_status(name)}
        item["in_cache"] = True
        # llama backends carry `file`; dflash carries `target`.
        item["path"] = cfg.get("file") or cfg.get("target") or ""
        item["context_window"] = cfg.get("ctx-size", DEFAULT_CTX_SIZE)
    return {"object": "list", "data": data}


@router.get("/models")
async def get_models(request: Request):
    """Return registry and active model info."""
    require_api(request)
    manager = _get_manager()
    return {
        "models": registry.list_all(),
        "metadata": registry.list_metadata(),
        "fixed": registry.list_fixed(),
        "active": manager.list_active(),
        "gpu_count": manager.gpu_count
    }


@router.get("/status")
async def status(request: Request):
    """Return system status."""
    require_api(request)
    manager = _get_manager()
    active_info = {}
    for name in manager.list_active():
        active = manager.get_active(name)
        if not active:
            continue
        active_info[name] = {
            "gpu": active.gpu,
            "port": active.port,
            "started": active.started.isoformat(),
            "pinned": registry.is_fixed(name),
            "running": active.is_running(),
        }
    return {
        "models": registry.list_all(),
        "fixed": registry.list_fixed(),
        "active": active_info,
        "gpu_count": manager.gpu_count
    }


@router.post("/switch/{model_name}")
async def switch_model(model_name: str, request: Request):
    """Start a model with GPU allocation."""
    require_admin(request)
    manager = _get_manager()
    try:
        active = await manager.start_model(model_name)
        return {
            "status": "started",
            "model": model_name,
            "gpu": active.gpu,
            "port": active.port
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop/{model_name}")
async def stop_model_endpoint(model_name: str, request: Request):
    """Stop an active model."""
    require_admin(request)
    manager = _get_manager()
    if not manager.get_active(model_name):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' is not active")
    await manager.stop_model(model_name)
    return {"status": "stopped", "model": model_name}


@router.post("/models/load")
async def models_load(request: Request):
    """Router-mode alias of /switch/{name}, called by stock llama.cpp webui."""
    require_admin(request)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    name = _model_payload_name(payload)
    if not name:
        raise HTTPException(status_code=400, detail="Missing 'model' in body")
    return await switch_model(name, request)


@router.post("/models/unload")
async def models_unload(request: Request):
    """Router-mode alias of /stop/{name}, called by stock llama.cpp webui."""
    require_admin(request)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    name = _model_payload_name(payload)
    if not name:
        raise HTTPException(status_code=400, detail="Missing 'model' in body")
    return await stop_model_endpoint(name, request)


@router.post("/ingest")
async def ingest_model(request: Request):
    """Download and register a new model."""
    require_admin(request)
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    model_alias = data.get("alias")
    model_url = data.get("url")
    ctx_size = data.get("ctx-size", DEFAULT_CTX_SIZE)

    if not model_alias or not model_url:
        raise HTTPException(status_code=400, detail="Missing 'alias' or 'url'")

    try:
        model_filename = model_filename_from_url(model_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    model_dir = os.path.join(MODELS_DIR, "gguf")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_filename)

    if os.path.exists(model_path):
        raise HTTPException(status_code=409, detail=f"Model file already exists at {model_path}")

    try:
        logger.info(f"Downloading model from {model_url} to {model_path}")
        await asyncio.to_thread(download_model_file, model_url, model_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download model: {str(e)}")

    try:
        registry.add(model_alias, {
            "file": f"gguf/{model_filename}",
            "mmproj": None,
            "ctx-size": ctx_size,
        })
        logger.info(f"Added model {model_alias} to registry")
        return {"status": "added", "model": model_alias}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/props")
async def props(request: Request):
    """Router-mode /props for the stock llama.cpp webui.

    Without ?model=<id> returns server-wide router props.
    With ?model=<id>&autoload=false returns synthetic per-model props from registry.
    With ?model=<id> (autoload not false) starts the model and proxies its real /props.
    """
    require_api(request)
    model_name = request.query_params.get("model")
    autoload = request.query_params.get("autoload", "true").lower() not in {"false", "0", "no", "off"}

    if not model_name:
        return _synthetic_props()

    resolved = registry.resolve(model_name)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not in registry")

    manager = _get_manager()
    if not autoload and not manager.get_active(resolved):
        return _synthetic_props(resolved)

    try:
        active = await manager.start_model(resolved)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start {resolved} for /props: {e}")
        raise HTTPException(status_code=502, detail="Model server unavailable")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"http://127.0.0.1:{active.port}/props")
        if resp.status_code == 200:
            data = resp.json()
            data["role"] = "router"
            return data
    except Exception as e:
        logger.info(f"Falling back to synthetic /props for {resolved}: {e}")
    return _synthetic_props(resolved)


# ---------------------------------------------------------------------------
# Registry management endpoints
# ---------------------------------------------------------------------------


@router.get("/registry")
async def get_registry(request: Request):
    """Return full models.json snapshot + GGUF directory scan."""
    require_api(request)
    manager = _get_manager()
    snapshot = registry.snapshot()
    return {
        "models": snapshot.get("models", {}),
        "fixed": snapshot.get("fixed", {}),
        "family_defaults": snapshot.get("family_defaults", {}),
        "active": manager.list_active(),
        "gpu_count": manager.gpu_count,
        "gguf_files": _scan_gguf_files(),
        "mmproj_files": [
            f for f in _scan_gguf_files()
            if (isinstance(f.get("used_by"), list) and len(f["used_by"]) >= 1
                and any(
                    (registry.get(m) or {}).get("mmproj") == f["filename"]
                    for m in f["used_by"]
                ))
        ],
        "gguf_dir": os.path.join(MODELS_DIR, "gguf"),
    }


@router.get("/registry/model/{name}")
async def get_registry_model(name: str, request: Request):
    """Return a single model config from the registry."""
    require_api(request)
    cfg = registry.get(name)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    return cfg


@router.put("/registry/model/{name}")
async def put_registry_model(name: str, request: Request):
    """Upsert a model entry."""
    require_api(request)
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    _validate_model_config(data)

    created = False
    try:
        result = registry.add(name, data)
        created = True
    except ValueError:
        result = registry.update(name, data)

    manager = _get_manager()
    valid, msg = registry.validate(name, gpu_count=manager.gpu_count)
    return JSONResponse(
        content={"model": result, "valid": valid, "message": msg},
        status_code=201 if created else 200,
    )


@router.delete("/registry/model/{name}")
async def delete_registry_model(
    name: str,
    request: Request,
    delete_gguf: bool = False,
    delete_mtp_head: bool = False,
    force: bool = False,
):
    """Remove a model entry with optional shared-file gate."""
    require_api(request)
    cfg = registry.get(name)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")

    # Collect files to check / delete
    files_to_maybe_delete: list[tuple[str, str]] = []
    if delete_gguf and cfg.get("file"):
        files_to_maybe_delete.append(("file", cfg["file"]))
    if delete_mtp_head and cfg.get("mtp-head"):
        files_to_maybe_delete.append(("mtp-head", cfg["mtp-head"]))
    # mmproj is never deleted (frequently shared)

    # Remove the registry entry first
    registry.remove(name)

    # Check shared references
    shared = {}
    for _, rel_path in files_to_maybe_delete:
        other_refs = _file_referenced_by(rel_path)
        if other_refs:
            shared[rel_path] = other_refs

    if shared and not force:
        return JSONResponse(
            content={"status": "removed", "shared_files": shared},
            status_code=200,
        )

    # Delete files
    deleted = []
    for _, rel_path in files_to_maybe_delete:
        try:
            real_path = _containment_check(rel_path)
            if os.path.isfile(real_path):
                os.remove(real_path)
                deleted.append(rel_path)
        except (HTTPException, OSError) as e:
            logger.warning(f"Failed to delete {rel_path}: {e}")

    return {"status": "removed", "deleted_files": deleted}


@router.delete("/registry/gguf")
async def delete_registry_gguf(request: Request, filename: str = ""):
    """Delete an orphaned GGUF file."""
    require_api(request)
    if not filename:
        raise HTTPException(status_code=400, detail="'filename' query parameter is required")
    real_path = _containment_check(filename)
    if not os.path.isfile(real_path):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    refs = _file_referenced_by(filename)
    if refs:
        raise HTTPException(
            status_code=409,
            detail=f"File is still referenced by: {', '.join(refs)}",
        )
    try:
        os.remove(real_path)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {e}")
    return {"status": "deleted"}


@router.post("/registry/upload")
async def upload_registry_gguf(request: Request, file: UploadFile):
    """Upload a .gguf file to the models directory."""
    require_api(request)
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    original = file.filename
    if not original.lower().endswith(".gguf"):
        raise HTTPException(status_code=400, detail="Only .gguf files are accepted")

    safe_name = Path(original).name
    if not safe_name or safe_name != original or ".." in safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    gguf_dir = os.path.join(MODELS_DIR, "gguf")
    os.makedirs(gguf_dir, exist_ok=True)
    target_path = os.path.join(gguf_dir, safe_name)

    if os.path.exists(target_path):
        raise HTTPException(status_code=409, detail=f"File already exists: gguf/{safe_name}")

    try:
        total = 0
        with open(target_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > _MAX_GGUF_UPLOAD_BYTES:
                    f.close()
                    os.remove(target_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds max size of {_MAX_GGUF_UPLOAD_BYTES} bytes",
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except OSError as e:
        try:
            os.remove(target_path)
        except OSError:
            pass
        raise HTTPException(status_code=507, detail=f"Disk write failed: {e}")
    except Exception as e:
        try:
            os.remove(target_path)
        except OSError:
            pass
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    return {"filename": f"gguf/{safe_name}", "size_bytes": total}
