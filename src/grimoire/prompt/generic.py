"""Generic prompt layout, block building, and tokenization dispatcher."""

from grimoire import config
from grimoire.dflash.prefill import PromptBlock
from grimoire.prompt import _tool_name_from_message
from grimoire.prompt.qwen import (
    _encode_qwen_prompt_blocks,
    _prompt_block_cache_for,
    _qwen_prompt_block_specs,
    _qwen_prompt_blocks,
    _tokenize_qwen_prompt_blocks,
)


def _generic_prompt_blocks(messages, tokenizer, prompt_ids, add_generation_prompt=False):
    if not messages:
        return []

    blocks = []
    prev = 0
    tool_call_names = {}
    prev_rendered = ""
    prev_encoded: list = []
    for index, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id")
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                fn_name = fn.get("name")
                if isinstance(tc_id, str) and isinstance(fn_name, str):
                    tool_call_names[tc_id] = fn_name
        rendered = tokenizer.apply_chat_template(
            messages[: index + 1], tokenize=False, add_generation_prompt=False
        )
        # Chat templates generally produce stable append-only renderings, so
        # tokenize just the new suffix and concatenate against the previously
        # encoded prefix. Falls back to a full encode if the template wasn't
        # append-stable (e.g., a message edited an earlier turn's framing).
        encoded = None
        if prev_rendered and rendered.startswith(prev_rendered):
            suffix = rendered[len(prev_rendered):]
            suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False) if suffix else []
            candidate = prev_encoded + suffix_tokens
            if len(candidate) <= len(prompt_ids) and prompt_ids[:len(candidate)] == candidate:
                encoded = candidate
        if encoded is None:
            encoded = tokenizer.encode(rendered, add_special_tokens=False)
        end = len(encoded)
        if end <= prev or end > len(prompt_ids) or prompt_ids[:end] != encoded:
            raise ValueError(f"Unable to build prompt block span for message {index}")
        prev_rendered = rendered
        prev_encoded = encoded

        tool_name = _tool_name_from_message(msg, tool_call_names)
        role = str(msg.get("role") or "unknown")
        kind = role
        if role == "assistant" and msg.get("tool_calls"):
            kind = "assistant"
        if role == "tool":
            kind = "tool"

        metadata = {"message_index": index}
        if isinstance(tool_name, str):
            metadata["tool_name"] = tool_name

        blocks.append(
            PromptBlock(
                block_id=f"message:{index}",
                index=len(blocks),
                start=prev,
                end=end,
                role=role,
                kind=kind,
                message_start=index,
                message_end=index + 1,
                protected=tool_name in config.DFLASH_PROTECTED_TOOLS,
                metadata=metadata,
            )
        )
        prev = end

    if add_generation_prompt and prev < len(prompt_ids):
        blocks.append(
            PromptBlock(
                block_id="generation:0",
                index=len(blocks),
                start=prev,
                end=len(prompt_ids),
                role="assistant",
                kind="generation_prompt",
                message_start=len(messages),
                message_end=len(messages),
                protected=True,
                metadata={"generation_prompt": True},
            )
        )
        prev = len(prompt_ids)

    if prev != len(prompt_ids):
        raise ValueError("Prompt blocks did not cover the full prompt")
    return blocks


def _prompt_layout_from_messages(tokenizer, messages, add_generation_prompt=False, model_cfg=None, active=None):
    family = model_cfg.get("family") if isinstance(model_cfg, dict) else None
    if family == "qwen":
        specs = _qwen_prompt_block_specs(messages, add_generation_prompt=add_generation_prompt)
        block_texts = [spec["text"] for spec in specs]
        encoded_blocks = _tokenize_qwen_prompt_blocks(
            tokenizer,
            block_texts,
            cache=_prompt_block_cache_for(active),
        )
        prompt_ids = []
        prompt_blocks = []
        cursor = 0
        for index, (spec, block_ids) in enumerate(zip(specs, encoded_blocks)):
            start = cursor
            cursor += len(block_ids)
            prompt_ids.extend(block_ids)
            prompt_blocks.append(
                PromptBlock(
                    block_id=f"block:{index}",
                    index=index,
                    start=start,
                    end=cursor,
                    role=spec["role"],
                    kind=spec["kind"],
                    message_start=spec["message_start"],
                    message_end=spec["message_end"],
                    protected=bool(spec.get("protected")),
                    metadata=spec.get("metadata"),
                )
            )
        return prompt_ids, prompt_blocks

    prompt_ids = _prompt_ids_from_messages(
        tokenizer,
        messages,
        add_generation_prompt=add_generation_prompt,
        model_cfg=model_cfg,
        active=active,
    )
    return prompt_ids, _generic_prompt_blocks(
        messages,
        tokenizer,
        prompt_ids,
        add_generation_prompt=add_generation_prompt,
    )


def _prefix_cache_boundaries(blocks):
    return [
        block.end
        for block in blocks or []
        if block.message_end > block.message_start and block.kind != "generation_prompt" and block.end > 0
    ]


def _prompt_ids_from_messages(tokenizer, messages, add_generation_prompt=False, model_cfg=None, active=None):
    family = model_cfg.get("family") if isinstance(model_cfg, dict) else None
    qwen_error = None
    if family == "qwen":
        try:
            blocks = _qwen_prompt_blocks(messages, add_generation_prompt=add_generation_prompt)
            return _encode_qwen_prompt_blocks(
                tokenizer,
                blocks,
                cache=_prompt_block_cache_for(active),
            )
        except Exception as e:
            qwen_error = e

    try:
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=False,
        )
        if not isinstance(prompt_ids, list):
            prompt_ids = list(prompt_ids)
        return prompt_ids
    except Exception as e:
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
            return tokenizer.encode(prompt_text, add_special_tokens=False)
        except Exception:
            if qwen_error is not None:
                raise qwen_error
            raise e
