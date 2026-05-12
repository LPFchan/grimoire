"""History route handlers (/history/*)."""

from fastapi import APIRouter, HTTPException, Request

from grimoire.auth import require_api

router = APIRouter()


def _history_store():
    from grimoire.entrypoint import history_store
    return history_store


@router.get("/history")
async def list_history(request: Request):
    """List conversations for the authenticated API key (tree-aware shape)."""
    _, user_hash = require_api(request)
    conversations = _history_store().list_conversations_tree(user_hash)
    return {"conversations": conversations}


@router.post("/history")
async def create_history(request: Request):
    """Create a conversation for the authenticated API key.

    Webui upsert path: pass {id, name, lastModified, currNode, ...}.
    Legacy gateway path: pass {title, model, messages: [...]}.
    """
    _, user_hash = require_api(request)
    data = await request.json()
    if data.get("id") or data.get("name") is not None or data.get("lastModified") is not None:
        try:
            return _history_store().upsert_conversation_tree(user_hash, data)
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e))
    return _history_store().create_conversation(
        user_hash,
        title=data.get("title") or "New chat",
        model=data.get("model"),
        messages=data.get("messages") or [],
    )


@router.get("/history/{conversation_id}")
async def get_history(conversation_id: str, request: Request):
    """Return one server-side conversation with tree-shaped messages."""
    _, user_hash = require_api(request)
    try:
        return _history_store().get_conversation_tree(user_hash, conversation_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/history/{conversation_id}")
async def update_history(conversation_id: str, request: Request):
    """Replace metadata/messages for one server-side conversation."""
    _, user_hash = require_api(request)
    data = await request.json()
    try:
        return _history_store().replace_conversation(
            user_hash,
            conversation_id,
            title=data.get("title"),
            model=data.get("model"),
            messages=data.get("messages"),
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.patch("/history/{conversation_id}")
async def patch_history(conversation_id: str, request: Request):
    """Partial-update conversation metadata (webui's updateConversation)."""
    _, user_hash = require_api(request)
    data = await request.json()
    try:
        return _history_store().patch_conversation_tree(user_hash, conversation_id, data)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/history/{conversation_id}")
async def delete_history(conversation_id: str, request: Request):
    """Delete one conversation; pass ?with_forks=true to cascade through forks."""
    _, user_hash = require_api(request)
    with_forks = request.query_params.get("with_forks", "").lower() in {"1", "true", "yes", "on"}
    try:
        _history_store().delete_conversation_with_options(user_hash, conversation_id, delete_with_forks=with_forks)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"deleted": conversation_id}


@router.patch("/history/messages/{message_id}")
async def patch_history_message_by_id(message_id: str, request: Request):
    """Webui's updateMessage doesn't carry convId; resolve it from the message row."""
    _, user_hash = require_api(request)
    conv_id = _history_store().find_message_conversation(user_hash, message_id)
    if not conv_id:
        raise HTTPException(status_code=404, detail=f"Message '{message_id}' not found")
    data = await request.json()
    _history_store().update_message_tree(user_hash, conv_id, message_id, data)
    return {"updated": message_id}


@router.delete("/history/messages/{message_id}")
async def delete_history_message_by_id(message_id: str, request: Request):
    """Webui's deleteMessage doesn't carry convId; resolve it from the message row."""
    _, user_hash = require_api(request)
    conv_id = _history_store().find_message_conversation(user_hash, message_id)
    if not conv_id:
        raise HTTPException(status_code=404, detail=f"Message '{message_id}' not found")
    cascade = request.query_params.get("cascade", "").lower() in {"1", "true", "yes", "on"}
    deleted = _history_store().delete_message_tree(user_hash, conv_id, message_id, cascade=cascade)
    return {"deleted": deleted}


@router.post("/history/{conversation_id}/messages")
async def create_history_message(conversation_id: str, request: Request):
    """Create a message branch under parent_id and update the conversation's currNode."""
    _, user_hash = require_api(request)
    data = await request.json()
    try:
        return _history_store().create_message_branch(user_hash, conversation_id, data)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.patch("/history/{conversation_id}/messages/{message_id}")
async def patch_history_message(conversation_id: str, message_id: str, request: Request):
    """Partial-update a message (webui's updateMessage)."""
    _, user_hash = require_api(request)
    data = await request.json()
    try:
        _history_store().update_message_tree(user_hash, conversation_id, message_id, data)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"updated": message_id}


@router.delete("/history/{conversation_id}/messages/{message_id}")
async def delete_history_message(conversation_id: str, message_id: str, request: Request):
    """Delete a message; pass ?cascade=true to delete the whole subtree."""
    _, user_hash = require_api(request)
    cascade = request.query_params.get("cascade", "").lower() in {"1", "true", "yes", "on"}
    try:
        deleted = _history_store().delete_message_tree(user_hash, conversation_id, message_id, cascade=cascade)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"deleted": deleted}


@router.post("/history/{conversation_id}/fork")
async def fork_history(conversation_id: str, request: Request):
    """Fork a conversation at a specific message into a new conversation."""
    _, user_hash = require_api(request)
    data = await request.json()
    at_message_id = data.get("at_message_id") or data.get("atMessageId")
    name = data.get("name") or "Forked chat"
    include_attachments = data.get("include_attachments", data.get("includeAttachments", True))
    if not at_message_id:
        raise HTTPException(status_code=400, detail="Missing 'at_message_id'")
    try:
        return _history_store().fork_conversation(
            user_hash, conversation_id, at_message_id, name, include_attachments
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/history/import")
async def import_history(request: Request):
    """Bulk-import conversations in the webui's exported shape."""
    _, user_hash = require_api(request)
    data = await request.json()
    return _history_store().import_conversations_tree(user_hash, data)
