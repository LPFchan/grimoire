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
from grimoire.ingest import download_model_file, model_filename_from_url, parse_hf_url, download_model_file_with_progress, _DownloadCancelled, MAX_BYTES as INGEST_MAX_BYTES
from grimoire.registry import MODELS_DIR, registry

_REFERENCE_FIELDS = ("file", "mmproj", "mtp-head", "spec-draft-model", "draft", "drafter")
_MAX_GGUF_UPLOAD_BYTES = int(os.environ.get("GRIMOIRE_MAX_GGUF_UPLOAD_BYTES", 40 * 1024**3))

router = APIRouter()
logger = logging.getLogger(__name__)


@router.on_event("startup")
async def _on_startup():
    asyncio.create_task(prune_old_ingest_tasks())


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
                  "image-min-tokens", "image-max-tokens", "vram-budget-mib"):
        val = data.get(field)
        # bool is an int subclass; reject it so `true` isn't read as 1.
        if val is not None and (isinstance(val, bool) or not isinstance(val, int) or val <= 0):
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


# ---------------------------------------------------------------------------
# Ingest infrastructure (background downloads with progress)
# ---------------------------------------------------------------------------

import time as _time

_ingest_tasks: dict[str, dict] = {}
_ingest_lock = asyncio.Lock()


async def prune_old_ingest_tasks():
    """Periodically remove old ingest task entries and orphaned files."""
    while True:
        await asyncio.sleep(600)  # every 10 minutes
        cutoff = _time.time() - 3600  # 1 hour ago
        async with _ingest_lock:
            stale_ids = [
                tid for tid, t in _ingest_tasks.items()
                if t.get("_completed_at", _time.time()) < cutoff
            ]
            for tid in stale_ids:
                task = _ingest_tasks.pop(tid, None)
                if task:
                    for path_key in ("tmp_path", "target_path"):
                        p = task.get(path_key)
                        if p and os.path.isfile(p):
                            try:
                                os.remove(p)
                            except OSError:
                                pass
            # Clean orphaned .part files older than 1 hour
            gguf_dir = os.path.join(MODELS_DIR, "gguf")
            if os.path.isdir(gguf_dir):
                for entry in os.listdir(gguf_dir):
                    if not entry.endswith(".part"):
                        continue
                    part_path = os.path.join(gguf_dir, entry)
                    try:
                        if _time.time() - os.path.getmtime(part_path) > 3600:
                            os.remove(part_path)
                    except OSError:
                        pass


# ---------------------------------------------------------------------------
# Ingest endpoints (HuggingFace URL → download → register)
# ---------------------------------------------------------------------------


@router.post("/registry/ingest-start")
async def ingest_start(request: Request):
    """Parse a HuggingFace URL and start a background download."""
    require_api(request)
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    url = data.get("url")
    if not url or not isinstance(url, str):
        raise HTTPException(status_code=400, detail="'url' is required")

    try:
        download_url, filename = parse_hf_url(url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    suggested_alias = os.path.splitext(os.path.basename(filename))[0].lower()

    task_id = os.urandom(8).hex()
    gguf_dir = os.path.join(MODELS_DIR, "gguf")
    os.makedirs(gguf_dir, exist_ok=True)
    target_path = os.path.join(gguf_dir, os.path.basename(filename))

    task = {
        "task_id": task_id,
        "status": "downloading",
        "filename": os.path.basename(filename),
        "suggested_alias": suggested_alias,
        "downloaded_bytes": 0,
        "total_bytes": None,
        "target_path": target_path,
        "cancelled": False,
        "configured": False,
        "alias": None,
        "load_settings_from": None,
        "error": None,
        "last_poll_at": None,
        "last_poll_bytes": None,
    }

    async with _ingest_lock:
        _ingest_tasks[task_id] = task

    async def _run_download():
        try:
            await asyncio.to_thread(
                download_model_file_with_progress,
                download_url, target_path, task, max_bytes=INGEST_MAX_BYTES
            )
            async with _ingest_lock:
                if task["configured"]:
                    _register_model(task_id)
                    task["status"] = "done"
                else:
                    task["status"] = "awaiting_configure"
                task["_completed_at"] = _time.time()
        except _DownloadCancelled:
            async with _ingest_lock:
                task["status"] = "cancelled"
                task["_completed_at"] = _time.time()
                if os.path.isfile(target_path):
                    try:
                        os.remove(target_path)
                    except OSError:
                        pass
        except Exception as e:
            async with _ingest_lock:
                task["status"] = "failed"
                task["error"] = str(e)
                task["_completed_at"] = _time.time()

    asyncio.create_task(_run_download())

    return {
        "status": "started",
        "task_id": task_id,
        "filename": os.path.basename(filename),
        "suggested_alias": suggested_alias,
    }


@router.post("/registry/ingest-configure")
async def ingest_configure(request: Request):
    """Save config for a download task. Model is registered when download completes."""
    require_api(request)
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    task_id = data.get("task_id")
    if not task_id:
        raise HTTPException(status_code=400, detail="'task_id' is required")

    alias = data.get("alias")
    if not alias or not isinstance(alias, str):
        raise HTTPException(status_code=400, detail="'alias' is required")

    load_settings_from = data.get("load_settings_from")
    if load_settings_from and not isinstance(load_settings_from, str):
        raise HTTPException(status_code=400, detail="'load_settings_from' must be a string or null")

    async with _ingest_lock:
        task = _ingest_tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if task["status"] in ("cancelled", "failed", "done"):
            raise HTTPException(
                status_code=400,
                detail=f"Task is in terminal state: {task['status']}",
            )

        task["configured"] = True
        task["alias"] = alias
        task["load_settings_from"] = load_settings_from

        # Validate merged config if settings source specified
        if load_settings_from:
            src = registry.get(load_settings_from)
            if not src:
                raise HTTPException(status_code=404, detail=f"Source model '{load_settings_from}' not found")

        if task["status"] in ("awaiting_configure", "downloading"):
            # If download is already done, register immediately
            if task["status"] == "awaiting_configure":
                _register_model(task_id)
                task["status"] = "done"

    return {"status": "configured"}


@router.get("/registry/ingest-status/{task_id}")
async def ingest_status(task_id: str, request: Request):
    """Poll the status of a background download."""
    require_api(request)
    async with _ingest_lock:
        task = _ingest_tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        now = _time.time()

        # Compute speed from last poll
        speed_bytes_per_sec = None
        if task.get("last_poll_at") and task.get("last_poll_bytes") is not None:
            delta_t = now - task["last_poll_at"]
            delta_b = task["downloaded_bytes"] - task["last_poll_bytes"]
            if delta_t > 0:
                speed_bytes_per_sec = int(delta_b / delta_t)

        task["last_poll_at"] = now
        task["last_poll_bytes"] = task["downloaded_bytes"]

        response = {
            "status": task["status"],
            "filename": task["filename"],
            "suggested_alias": task["suggested_alias"],
            "downloaded_bytes": task["downloaded_bytes"],
            "total_bytes": task["total_bytes"],
            "speed_bytes_per_sec": speed_bytes_per_sec,
        }

        if task["status"] == "done":
            response["alias"] = task["alias"]
        if task["status"] == "failed":
            response["error"] = task["error"]

        return response


@router.delete("/registry/ingest-status/{task_id}")
async def ingest_cancel(task_id: str, request: Request):
    """Cancel a background download."""
    require_api(request)
    async with _ingest_lock:
        task = _ingest_tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if task["status"] not in ("downloading", "awaiting_configure"):
            return {"status": "already_finished"}

        task["cancelled"] = True

    # Wait briefly for the chunk loop to notice the flag
    await asyncio.sleep(1)

    async with _ingest_lock:
        if os.path.isfile(task.get("target_path", "")):
            try:
                os.remove(task["target_path"])
            except OSError:
                pass
        tmp = task.get("tmp_path")
        if tmp and os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        task["status"] = "cancelled"
        task["_completed_at"] = _time.time()

    return {"status": "cancelled"}


# ---------------------------------------------------------------------------
# GGUF file management
# ---------------------------------------------------------------------------


@router.patch("/registry/gguf")
async def rename_gguf(request: Request, filename: str = "", new_filename: str = ""):
    """Rename a GGUF file and update all model configs referencing it."""
    require_api(request)
    if not filename or not new_filename:
        raise HTTPException(status_code=400, detail="'filename' and 'new_filename' are required")

    # Containment check on old file
    real_old = _containment_check(filename)
    if not os.path.isfile(real_old):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    # New filename must end with .gguf
    if not new_filename.lower().endswith(".gguf"):
        raise HTTPException(status_code=400, detail="'new_filename' must end with .gguf")

    # Containment check on parent of new filename (file doesn't exist yet)
    new_parent = os.path.realpath(os.path.join(MODELS_DIR, os.path.dirname(new_filename) or "."))
    real_models = os.path.realpath(MODELS_DIR)
    if os.path.commonpath([real_models, new_parent]) != real_models:
        raise HTTPException(status_code=400, detail="Path escapes models directory")

    real_new = os.path.join(new_parent, os.path.basename(new_filename))

    # Collision check
    if os.path.exists(real_new):
        raise HTTPException(status_code=409, detail=f"'{new_filename}' already exists")

    try:
        os.rename(real_old, real_new)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Rename failed: {e}")

    # Update all model configs referencing the old filename
    from grimoire.registry import registry as reg
    updated = []
    for model_name in reg.list_all():
        cfg = reg.get(model_name)
        changed = False
        for field in _REFERENCE_FIELDS:
            if cfg and cfg.get(field) == filename:
                cfg[field] = new_filename
                changed = True
        if changed:
            reg.update(model_name, cfg)
            updated.append(model_name)

    return {"status": "renamed", "updated_models": updated}


def _register_model(task_id: str):
    """Register a completed download into the registry. Caller must hold _ingest_lock."""
    task = _ingest_tasks.get(task_id)
    if not task:
        return

    alias = task.get("alias")
    filename = task.get("filename")
    if not alias or not filename:
        return

    config = {
        "file": f"gguf/{filename}",
        "capabilities": ["completion"],
        "ctx-size": 262144,
        "predict": 16384,
        "parallel": 1,
        "n-gpu-layers": 999,
        "cache-type-k": "q8_0",
        "cache-type-v": "q8_0",
        "cost": {"input": 0, "output": 0, "cache_read": 0},
    }

    # Merge settings from source model if specified
    load_from = task.get("load_settings_from")
    if load_from:
        src = registry.get(load_from)
        if src:
            for k, v in src.items():
                if k not in ("file", "mmproj", "alias", "added", "backend",
                             "mtp-head", "spec-draft-model", "draft", "drafter"):
                    config[k] = v

    # Strip incompatible combos
    if config.get("speculative-type") == "mtp" and not config.get("mtp-head"):
        config.pop("speculative-type", None)
    if config.get("speculative-type") == "dflash" and not config.get("spec-draft-model"):
        config.pop("speculative-type", None)
    if config.get("pflash") and not config.get("drafter"):
        config.pop("pflash", None)
        config.pop("park-unpark", None)
    caps = config.get("capabilities") or []
    if ("multimodal" in caps or "vision" in caps) and not config.get("mmproj"):
        config["capabilities"] = [c for c in caps if c not in ("multimodal", "vision")]

    registry.add(alias, config)
    task["alias"] = alias
