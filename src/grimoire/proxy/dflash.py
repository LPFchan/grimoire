"""DFlash backend proxy path — chat completions via daemon stdin/stdout."""

import asyncio
import copy
import json
import shutil
import os
import time
import uuid
import logging

from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from grimoire import config
from grimoire.dflash.prefill import materialize_blocks, maybe_compress
from grimoire.plugins import plugin_manager
from grimoire.prompt.generic import _prompt_layout_from_messages, _prefix_cache_boundaries
from grimoire.proxy.sse import _sse_error_frames, _delta_sse, _final_sse

logger = logging.getLogger(__name__)


async def _proxy_dflash(requested_model, payload, active, user_hash, conversation_id):
    """Handle chat completions for the dflash backend.

    Streams via the daemon's stdin/stdout protocol while reusing the same
    history/telemetry/plugin pipeline as the llama path so dashboards work.
    """
    # Local import avoids circular dependency with entrypoint.
    from grimoire.entrypoint import _record_response_stream

    model_cfg = active.cfg
    payload = copy.deepcopy(payload)
    payload = plugin_manager.before_request(payload, active.name, model_cfg)

    messages = payload.get("messages", [])
    want_stream = payload.get("stream", True)
    max_tokens = int(payload.get("max_tokens", model_cfg.get("predict", config.DEFAULT_PREDICT)) or config.DEFAULT_PREDICT)
    temperature = payload.get("temperature", 0.8)
    top_p = payload.get("top_p", 0.9)
    top_k = payload.get("top_k", 40)
    seed = payload.get("seed")

    for name in config.DFLASH_IGNORED_SAMPLING:
        if payload.get(name) is not None:
            logger.warning(
                f"dflash: ignoring unsupported sampling param '{name}' on model {active.name}"
            )

    try:
        tokenizer = active.get_tokenizer()
    except Exception as e:
        logger.error(f"Failed to load tokenizer for {active.name}: {e}")
        raise HTTPException(status_code=503, detail=f"Tokenizer unavailable: {e}")

    try:
        prompt_ids, prompt_blocks = _prompt_layout_from_messages(
            tokenizer,
            messages,
            add_generation_prompt=True,
            model_cfg=model_cfg,
            active=active,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to render chat template: {e}")

    ctx_size = int(model_cfg.get("ctx-size", config.DEFAULT_CTX_SIZE))
    max_effective_context = model_cfg.get(
        "max-effective-context",
        model_cfg.get(
            "max_effective_context",
            model_cfg.get("max-raw-context", model_cfg.get("max_raw_context")),
        ),
    )
    # Cheap raw-size reject before we take the daemon lock — a pathologically
    # large prompt shouldn't tie up the drafter scoring uncompressible tokens.
    max_raw_ceiling = int(model_cfg.get(
        "max-raw-ceiling",
        model_cfg.get("max_raw_ceiling", ctx_size),
    ))
    if max_raw_ceiling and len(prompt_ids) > max_raw_ceiling:
        raise HTTPException(
            status_code=400,
            detail=(
                f"raw prompt ({len(prompt_ids)} tokens) exceeds max raw ceiling "
                f"{max_raw_ceiling}"
            ),
        )

    prefill_config = active.prefill_config
    daemon = active.dflash_daemon

    # Hold the daemon lock across compression AND generation as one critical
    # section, so concurrent requests serialize the full pipeline (compress →
    # daemon work) instead of interleaving compressions. The streaming
    # generator's finally releases on exit; the outer except is the backstop
    # for the pre-stream path.
    lock = active.dflash_lock()
    await lock.acquire()
    _released = False
    def _release_once():
        nonlocal _released
        if not _released:
            _released = True
            lock.release()

    lock_handoff = False
    try:
        effective_ids = prompt_ids
        effective_blocks = materialize_blocks(prompt_ids, prompt_blocks)
        if daemon is not None and daemon.is_running() and prefill_config and prefill_config.enabled:
            try:
                effective_ids, _, effective_blocks = await maybe_compress(
                    prompt_ids,
                    daemon,
                    prefill_config,
                    blocks=prompt_blocks,
                )
            except Exception as e:
                logger.error(f"pflash compression failed: {e}")
                effective_ids = prompt_ids
                effective_blocks = materialize_blocks(prompt_ids, prompt_blocks)

        if max_effective_context is not None:
            max_effective_context = int(max_effective_context)
            if len(effective_ids) > max_effective_context:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"effective prompt ({len(effective_ids)} tokens) exceeds max effective context "
                        f"{max_effective_context}"
                    ),
                )

        if len(effective_ids) + max_tokens > ctx_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"effective prompt ({len(effective_ids)} tokens) + max_tokens ({max_tokens}) "
                    f"exceeds context size {ctx_size}"
                ),
            )

        effective_boundaries = _prefix_cache_boundaries(effective_blocks)

        stop_ids, stop_seqs = _dflash_collect_stop_ids(tokenizer, payload.get("stop"), model_cfg)

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        async def sse_stream():
            loaded_staging = False
            had_session = False
            pc = None
            prepared_prefix = None
            prefix_snapshot_key = None
            prefix_snapshot_len = None
            prefix_snapshot_confirmed = False
            try:
                if daemon is None or not daemon.is_running():
                    yield _sse_error_frames(completion_id, created, "dflash daemon not running")
                    return

                prefix_hit = None
                sk = active.session_kv
                pc = active.prefix_cache
                store = getattr(active, "snapshot_swap", None)
                staging_slot = getattr(active, "snapshot_staging_slot", 7)
                if store is not None:
                    store.bind_loop()

                async def _load_into_staging(snapshot_key, prefix_len):
                    ok = await asyncio.to_thread(store.load, daemon, snapshot_key, staging_slot)
                    if not ok:
                        raise KeyError(snapshot_key)
                    return (staging_slot, prefix_len)

                if conversation_id and sk and store:
                    prior_session_keys = sk.all_keys(conversation_id) if sk.has_session(conversation_id) else []
                    session_hit = sk.get_session(conversation_id, effective_ids)
                    if session_hit is None and prior_session_keys and not sk.has_session(conversation_id):
                        for key in prior_session_keys:
                            await asyncio.to_thread(store.discard, key)
                    if session_hit is not None:
                        had_session = True
                        session_key, session_prefix_len = session_hit
                        try:
                            prefix_hit = await _load_into_staging(session_key, session_prefix_len)
                            loaded_staging = True
                        except KeyError:
                            logger.warning("session snapshot missing from store; evicting session")
                            await asyncio.to_thread(store.discard, session_key)
                            sk.evict(conversation_id)

                if prefix_hit is None and pc and not pc.disabled and store:
                    prefix_cached = pc.lookup(effective_ids, boundaries=effective_boundaries)
                    if prefix_cached is not None:
                        prefix_key, prefix_len = prefix_cached
                        try:
                            prefix_hit = await _load_into_staging(prefix_key, prefix_len)
                            loaded_staging = True
                        except KeyError:
                            logger.warning("prefix snapshot missing from store; dropping cache entry")
                            await asyncio.to_thread(store.discard, prefix_key)
                            pc.discard(prefix_key)

                restored_prefix_len = prefix_hit[1] if prefix_hit else 0
                if pc and not pc.disabled and store and effective_boundaries:
                    boundary = effective_boundaries[0]
                    if boundary > restored_prefix_len:
                        prepared_prefix = pc.prepare_inline_snap(effective_ids, boundary)
                        if prepared_prefix is not None:
                            prefix_snapshot_key, prefix_snapshot_len = prepared_prefix

                cmd_path = await asyncio.to_thread(
                    daemon.send_generate_cmd,
                    effective_ids,
                    max_tokens,
                    prefix_hit[0] if prefix_hit else None,
                    staging_slot if prepared_prefix is not None else None,
                    prefix_snapshot_len,
                    temperature,
                    top_p,
                    top_k,
                    seed,
                )

                decoded_prefix = ""
                tokens_emitted = []
                # Sliding-window detokenize state: `read_offset` is the count of
                # tokens whose decoded text has been emitted; `prefix_offset` is
                # the earliest token still inside the decode window. Both reset
                # to the current tail after every successful emit, so each
                # decode is bounded to the most recent few tokens — overall O(n)
                # over the response instead of O(n²) over `decode(full_list)`
                # per token.
                prefix_offset = 0
                read_offset = 0
                stop_seq_lens = sorted({len(seq) for seq in stop_seqs}, reverse=True)
                t0 = time.monotonic()

                def _decode(start, stop):
                    return tokenizer.decode(
                        tokens_emitted[start:stop],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                try:
                    index = 0
                    hit_stop = False
                    while True:
                        tok = await asyncio.to_thread(daemon.read_next_token)
                        if tok is None:
                            break
                        if tok in stop_ids:
                            hit_stop = True
                            break

                        # Collect tokens up to max_tokens, but keep draining
                        # the daemon stream so the -1 sentinel is consumed and
                        # the pipe is clean for the next request.
                        if len(tokens_emitted) < max_tokens:
                            tokens_emitted.append(tok)
                            stop_hit = False
                            for seq_len in stop_seq_lens:
                                if len(tokens_emitted) < seq_len:
                                    continue
                                if tuple(tokens_emitted[-seq_len:]) in stop_seqs:
                                    del tokens_emitted[-seq_len:]
                                    stop_hit = True
                                    break

                            if stop_hit:
                                # The trim may have removed tokens we already emitted
                                # bytes for; re-decode against the shortened buffer
                                # and reconcile decoded_prefix so the final usage
                                # text matches what's actually in tokens_emitted.
                                read_offset = min(read_offset, len(tokens_emitted))
                                prefix_offset = min(prefix_offset, read_offset)
                                final_text = _decode(0, len(tokens_emitted))
                                if len(final_text) > len(decoded_prefix):
                                    delta = final_text[len(decoded_prefix):]
                                    decoded_prefix = final_text
                                    frame = _delta_sse(completion_id, created, delta, index)
                                    index += 1
                                    yield f"data: {json.dumps(frame)}\n\n".encode()
                                else:
                                    decoded_prefix = final_text
                                read_offset = len(tokens_emitted)
                                prefix_offset = read_offset
                                hit_stop = True
                                break

                            # Wait for a complete UTF-8 sequence before emitting:
                            # multi-byte chars can straddle BPE/SentencePiece pieces,
                            # so we keep accumulating while the window still ends in
                            # the replacement char U+FFFD.
                            prefix_text = _decode(prefix_offset, read_offset)
                            new_text = _decode(prefix_offset, len(tokens_emitted))
                            if not new_text.endswith("�") and len(new_text) > len(prefix_text):
                                delta = new_text[len(prefix_text):]
                                decoded_prefix += delta
                                prefix_offset = read_offset
                                read_offset = len(tokens_emitted)
                                frame = _delta_sse(completion_id, created, delta, index)
                                index += 1
                                yield f"data: {json.dumps(frame)}\n\n".encode()

                    # If we broke early (stop hit, max_tokens), drain the
                    # daemon's remaining tokens and the -1 sentinel so the
                    # pipe is clean for the next request.
                    if hit_stop:
                        while True:
                            leftover = await asyncio.to_thread(daemon.read_next_token)
                            if leftover is None:
                                break

                    # Flush any complete bytes still trapped in the sliding
                    # window (loop exited via EOS, max_tokens, or pipe close —
                    # not via the stop-hit branch, which already reconciled).
                    if read_offset < len(tokens_emitted):
                        prefix_text = _decode(prefix_offset, read_offset)
                        final_text = _decode(prefix_offset, len(tokens_emitted))
                        if len(final_text) > len(prefix_text):
                            delta = final_text[len(prefix_text):].rstrip("�")
                            if delta:
                                decoded_prefix += delta
                                frame = _delta_sse(completion_id, created, delta, index)
                                index += 1
                                yield f"data: {json.dumps(frame)}\n\n".encode()
                finally:
                    try:
                        os.unlink(cmd_path)
                    except OSError:
                        pass

                elapsed = max(time.monotonic() - t0, 1e-6)
                tps = len(tokens_emitted) / elapsed

                # Save snapshot before the zero-token guard so existing sessions
                # always persist even when no new tokens are generated (e.g. a
                # stop-token-only follow-up turn).
                try:
                    prefix_snapshot_written = False
                    if prepared_prefix is not None:
                        await asyncio.to_thread(store.save, daemon, prefix_snapshot_key, staging_slot)
                        prefix_snapshot_written = True

                    if tokens_emitted or had_session:
                        await asyncio.to_thread(daemon.snapshot, staging_slot)
                        if conversation_id and sk and store:
                            evicted_id = sk.evict_lru_if_full(conversation_id)
                            if evicted_id is not None:
                                for key in sk.all_keys(evicted_id):
                                    await asyncio.to_thread(store.discard, key)
                            session_key = sk.update(conversation_id, len(effective_ids), effective_ids)
                            if session_key is not None:
                                await asyncio.to_thread(store.save, daemon, session_key, staging_slot)

                        if prepared_prefix is not None:
                            if prefix_snapshot_written and prefix_snapshot_key is not None and prefix_snapshot_len is not None:
                                evicted_prefix_key = pc.confirm_inline_snap(
                                    prefix_snapshot_key,
                                    prefix_snapshot_len,
                                    effective_ids,
                                )
                                prefix_snapshot_confirmed = True
                                if evicted_prefix_key is not None:
                                    await asyncio.to_thread(store.discard, evicted_prefix_key)
                            else:
                                pc.abort_inline_snap(prefix_snapshot_key)
                except Exception as e:
                    logger.warning(f"snapshot save failed: {e}")
                    if store and prefix_snapshot_written and prefix_snapshot_key is not None:
                        await asyncio.to_thread(store.discard, prefix_snapshot_key)
                    if pc and prepared_prefix is not None and prefix_snapshot_key is not None:
                        pc.abort_inline_snap(prefix_snapshot_key)
                    if conversation_id and sk:
                        if store:
                            for key in sk.all_keys(conversation_id):
                                await asyncio.to_thread(store.discard, key)
                        sk.evict(conversation_id)

                # Detect silent failures where no tokens were emitted.
                if not tokens_emitted and max_tokens > 0:
                    if store and prefix_snapshot_written and prefix_snapshot_key is not None:
                        await asyncio.to_thread(store.discard, prefix_snapshot_key)
                    yield _sse_error_frames(
                        completion_id, created,
                        "DFlash generation failed: model produced zero tokens. "
                        "This usually means the context size + budget combination "
                        "exceeds available VRAM. Try reducing ctx-size or budget.",
                    )
                    return

                final = _final_sse(
                    completion_id, created,
                    len(effective_ids), len(tokens_emitted),
                    decoded_prefix, ctx_size,
                )
                final["timings"] = {
                    "predicted_n": len(tokens_emitted),
                    "predicted_ms": elapsed * 1000.0,
                    "predicted_per_second": tps,
                }
                yield f"data: {json.dumps(final)}\n\n".encode()
                yield b"data: [DONE]\n\n"
            finally:
                if pc and prepared_prefix is not None and prefix_snapshot_key is not None and not prefix_snapshot_confirmed:
                    pc.abort_inline_snap(prefix_snapshot_key)
                if loaded_staging:
                    try:
                        await asyncio.to_thread(daemon.free_snapshot, staging_slot)
                    except Exception:
                        pass
                _release_once()

        async def safe_stream():
            try:
                async for chunk in sse_stream():
                    yield chunk
            except Exception as e:
                logger.exception(f"dflash generation error: {e}")
                yield _sse_error_frames(completion_id, created, str(e))

        stream = safe_stream()
        stream = plugin_manager.wrap_response_stream(stream, active.name, model_cfg)
        if user_hash:
            stream = _record_response_stream(
                stream, user_hash, conversation_id, active.name,
                model_cfg, payload, gpu_index=active.gpu, record_history=True,
            )

        if want_stream:
            lock_handoff = True
            return StreamingResponse(
                stream,
                media_type="text/event-stream",
                headers={"x-request-id": requested_model},
            )

        body = bytearray()
        async for chunk in stream:
            body.extend(chunk)

        from grimoire.proxy.sse import _extract_assistant_text, _extract_usage
        text = _extract_assistant_text(bytes(body))
        usage = _extract_usage(bytes(body)) or {"input_tokens": len(effective_ids), "output_tokens": 0}

        # Detect silent dflash failures (e.g. OOM during decode) that produce
        # no tokens and empty content but don't raise an exception.
        if not text and usage["output_tokens"] == 0 and max_tokens > 0:
            raise HTTPException(
                status_code=503,
                detail="DFlash generation failed: model produced zero tokens. "
                       "This usually means the context size + budget combination "
                       "exceeds available VRAM. Try reducing ctx-size or budget.",
            )

        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": requested_model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": usage["input_tokens"],
                "completion_tokens": usage["output_tokens"],
                "total_tokens": usage["input_tokens"] + usage["output_tokens"],
            },
            "context_window": ctx_size,
        })
    finally:
        if not lock_handoff:
            _release_once()


def _dflash_collect_stop_ids(tokenizer, payload_stop, cfg):
    """Build the daemon's stop-id set from EOS, chat-template ends, and user stop."""
    stop_ids = set()
    stop_seqs = []
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    # Common chat-template assistant terminators across model families.
    for candidate in ("<|im_end|>", "<end_of_turn>", "<|eot_id|>", "<|end_of_text|>"):
        try:
            tok_id = tokenizer.convert_tokens_to_ids(candidate)
        except Exception:
            tok_id = None
        if isinstance(tok_id, int) and tok_id >= 0 and tok_id != tokenizer.unk_token_id:
            stop_ids.add(tok_id)

    # Operator-supplied stop strings on the model config.
    for s in cfg.get("stop-strings", []) or []:
        if not isinstance(s, str) or not s:
            continue
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            stop_ids.add(ids[0])
        elif ids:
            stop_seqs.append(tuple(ids))

    # Request-level stop strings.
    raw_stops = [payload_stop] if isinstance(payload_stop, str) else (payload_stop or [])
    for s in raw_stops:
        if isinstance(s, str) and s:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                stop_ids.add(ids[0])
            elif ids:
                stop_seqs.append(tuple(ids))

    return stop_ids, stop_seqs
