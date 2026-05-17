"""Grimoire SSE frame builders and response text/usage extractors."""

import json


def _extract_assistant_text(raw_bytes):
    text = raw_bytes.decode("utf-8", errors="ignore")
    pieces = []

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            continue
        for choice in parsed.get("choices", []) or []:
            delta = choice.get("delta") or {}
            message = choice.get("message") or {}
            content = delta.get("content") or message.get("content") or choice.get("text")
            if isinstance(content, str):
                pieces.append(content)

    if pieces:
        return "".join(pieces)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return ""
    choices = parsed.get("choices", []) if isinstance(parsed, dict) else []
    if not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message") or {}
    if isinstance(message.get("content"), str):
        return message["content"]
    if isinstance(first.get("text"), str):
        return first["text"]
    return ""


def _usage_from_object(data):
    if not isinstance(data, dict):
        return None
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
    cached_tokens = usage.get("cached_tokens")
    try:
        input_tokens = int(input_tokens or 0)
        output_tokens = int(output_tokens or 0)
        if cached_tokens is not None:
            cached_tokens = int(cached_tokens)
    except (TypeError, ValueError):
        return None
    if input_tokens <= 0 and output_tokens <= 0:
        return None
    result = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    if cached_tokens is not None and cached_tokens > 0:
        result["cached_tokens"] = cached_tokens
    return result


def _extract_usage(raw_bytes):
    """Extract token usage from JSON or final SSE chunks."""
    text = raw_bytes.decode("utf-8", errors="ignore")

    try:
        parsed = json.loads(text)
        usage = _usage_from_object(parsed)
        if usage:
            return usage
    except json.JSONDecodeError:
        pass

    found = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            continue
        usage = _usage_from_object(parsed)
        if usage:
            found = usage
    return found


def _extract_error_message(raw_bytes):
    """Extract an SSE or JSON error message if present."""
    text = raw_bytes.decode("utf-8", errors="ignore")

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            continue
        error = parsed.get("error") if isinstance(parsed, dict) else None
        if isinstance(error, dict) and isinstance(error.get("message"), str) and error["message"]:
            return error["message"]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict) and isinstance(error.get("message"), str) and error["message"]:
        return error["message"]
    return None


def _extract_tokens_per_sec(raw_bytes):
    """Extract predicted_per_second from the last timing chunk in SSE data."""
    text = raw_bytes.decode("utf-8", errors="ignore")
    best = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            continue
        timings = parsed.get("timings") if isinstance(parsed, dict) else None
        if isinstance(timings, dict):
            tps = timings.get("predicted_per_second")
            if isinstance(tps, (int, float)) and tps > 0:
                best = float(tps)
    return best


def _extract_chunk_tokens_per_sec(chunk):
    """Like _extract_tokens_per_sec but scans a single raw chunk for live updating."""
    return _extract_tokens_per_sec(chunk)


def _sse_error_frames(completion_id, created, message):
    """SSE error payload plus the [DONE] terminator clients expect."""
    err = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "error": {"message": message, "type": "server_error"},
    }
    return (
        f"data: {json.dumps(err)}\n\n".encode()
        + b"data: [DONE]\n\n"
    )


def _delta_sse(completion_id, created, content, index=0):
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "",
        "choices": [{
            "index": index,
            "delta": {"role": "assistant", "content": content},
            "finish_reason": None,
        }]
    }


def _final_sse(completion_id, created, prompt_tokens, completion_tokens, content, ctx_size, cached_tokens=None):
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
        "usage": usage,
        "context_window": ctx_size,
    }
    if cached_tokens is not None:
        payload["usage"]["cached_tokens"] = cached_tokens
    return payload
