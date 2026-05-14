"""Qwen-family prompt block rendering and tokenization."""

import json
from collections import OrderedDict

from grimoire import config
from grimoire.prompt import _tool_name_from_message


def _qwen_render_content(content, is_system_content=False):
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        rendered = []
        for item in content:
            if not isinstance(item, dict):
                raise ValueError("Unexpected item type in content.")
            item_type = item.get("type")
            if "image" in item or "image_url" in item or item_type == "image":
                if is_system_content:
                    raise ValueError("System message cannot contain images.")
                rendered.append("<|vision_start|><|image_pad|><|vision_end|>")
                continue
            if "video" in item or item_type == "video":
                if is_system_content:
                    raise ValueError("System message cannot contain videos.")
                rendered.append("<|vision_start|><|video_pad|><|vision_end|>")
                continue
            if "text" in item:
                rendered.append(str(item.get("text", "")))
                continue
            raise ValueError("Unexpected item type in content.")
        return "".join(rendered)
    raise ValueError("Unexpected content type.")


def _qwen_last_query_index(messages):
    for index in range(len(messages) - 1, -1, -1):
        msg = messages[index]
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = _qwen_render_content(msg.get("content")).strip()
        if not (content.startswith("<tool_response>") and content.endswith("</tool_response>")):
            return index
    raise ValueError("No user query found in messages.")


def _qwen_prompt_block_specs(messages, add_generation_prompt=False):
    if not messages:
        raise ValueError("No messages provided.")

    specs = []
    tool_call_names = {}
    if isinstance(messages[0], dict) and messages[0].get("role") == "system":
        content = _qwen_render_content(messages[0].get("content"), is_system_content=True).strip()
        specs.append({
            "text": f"<|im_start|>system\n{content}<|im_end|>\n",
            "role": "system",
            "kind": "system",
            "message_start": 0,
            "message_end": 1,
            "protected": False,
            "metadata": {"message_index": 0},
        })

    last_query_index = _qwen_last_query_index(messages)
    index = 0
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, dict):
            raise ValueError("Unexpected message role.")
        role = message.get("role")
        if role == "system":
            if index != 0:
                raise ValueError("System message must be at the beginning.")
            index += 1
            continue
        if role == "user":
            content = _qwen_render_content(message.get("content")).strip()
            specs.append({
                "text": f"<|im_start|>user\n{content}<|im_end|>\n",
                "role": "user",
                "kind": "user",
                "message_start": index,
                "message_end": index + 1,
                "protected": False,
                "metadata": {"message_index": index},
            })
            index += 1
            continue
        if role == "assistant":
            content = _qwen_render_content(message.get("content")).strip()
            reasoning_content = ""
            raw_reasoning = message.get("reasoning_content")
            if isinstance(raw_reasoning, str):
                reasoning_content = raw_reasoning
            elif "</think>" in content:
                reasoning_content = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                content = content.split("</think>")[-1].lstrip("\n")
            reasoning_content = reasoning_content.strip()

            tool_calls = message.get("tool_calls") or []
            if isinstance(tool_calls, dict):
                tool_calls = []

            has_reasoning = bool(reasoning_content) and (index > last_query_index)
            has_content = bool(content)
            has_tcs = bool(tool_calls)

            sub_specs = []
            header_used = False

            if not has_reasoning and not has_tcs:
                block = f"<|im_start|>assistant\n{content}<|im_end|>\n"
                sub_specs.append({
                    "text": block,
                    "role": "assistant",
                    "kind": "assistant",
                    "message_start": index,
                    "message_end": index + 1,
                    "protected": False,
                    "metadata": {"message_index": index},
                })
            else:
                if has_reasoning:
                    text = f"<|im_start|>assistant\n<think>\n{reasoning_content}\n</think>\n\n"
                    if not has_content and not has_tcs:
                        text += "<|im_end|>\n"
                    header_used = True
                    sub_specs.append({
                        "text": text,
                        "role": "assistant",
                        "kind": "thinking",
                        "message_start": index,
                        "message_end": index + 1,
                        "protected": False,
                        "metadata": {"message_index": index, "reasoning": True},
                    })

                if has_content:
                    text = ""
                    if not header_used:
                        text = f"<|im_start|>assistant\n"
                        header_used = True
                    text += content
                    if not has_tcs:
                        text += "<|im_end|>\n"
                    sub_specs.append({
                        "text": text,
                        "role": "assistant",
                        "kind": "assistant_text",
                        "message_start": index,
                        "message_end": index + 1,
                        "protected": False,
                        "metadata": {"message_index": index},
                    })

                for i, tool_call in enumerate(tool_calls):
                    if not isinstance(tool_call, dict):
                        continue
                    fn = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else tool_call
                    if not isinstance(fn, dict):
                        continue
                    name = fn.get("name")
                    if not isinstance(name, str) or not name:
                        continue
                    tc_id = tool_call.get("id")
                    if isinstance(tc_id, str):
                        tool_call_names[tc_id] = name

                    text = ""
                    if not header_used:
                        text = f"<|im_start|>assistant\n"
                        header_used = True
                    elif has_content and i == 0:
                        text = "\n\n"
                    elif i > 0:
                        text = "\n"

                    text += f"<tool_call>\n<function={name}>\n"
                    arguments = fn.get("arguments")
                    if isinstance(arguments, dict):
                        for arg_name, arg_value in arguments.items():
                            text += f"<parameter={arg_name}>\n"
                            if isinstance(arg_value, str):
                                rendered_value = arg_value
                            else:
                                rendered_value = json.dumps(arg_value, ensure_ascii=False)
                            text += f"{rendered_value}\n</parameter>\n"
                    text += "</function>\n</tool_call>"

                    if i == len(tool_calls) - 1:
                        text += "<|im_end|>\n"

                    sub_specs.append({
                        "text": text,
                        "role": "assistant",
                        "kind": "tool_call",
                        "message_start": index,
                        "message_end": index + 1,
                        "protected": False,
                        "metadata": {
                            "message_index": index,
                            "tool_name": name,
                        },
                    })

            specs.extend(sub_specs)
            index += 1
            continue
        if role == "tool":
            parts = ["<|im_start|>user"]
            group_start = index
            tool_names = []
            protected = False
            while index < len(messages):
                tool_msg = messages[index]
                if not isinstance(tool_msg, dict) or tool_msg.get("role") != "tool":
                    break
                content = _qwen_render_content(tool_msg.get("content")).strip()
                parts.append(f"\n<tool_response>\n{content}\n</tool_response>")
                tool_name = _tool_name_from_message(tool_msg, tool_call_names)
                if isinstance(tool_name, str):
                    tool_names.append(tool_name)
                    protected = protected or tool_name in config.DFLASH_PROTECTED_TOOLS
                index += 1
            parts.append("<|im_end|>\n")
            specs.append({
                "text": "".join(parts),
                "role": "tool",
                "kind": "tool_group",
                "message_start": group_start,
                "message_end": index,
                "protected": protected,
                "metadata": {
                    "message_indexes": list(range(group_start, index)),
                    "tool_names": tool_names,
                },
            })
            continue
        raise ValueError("Unexpected message role.")

    if add_generation_prompt:
        specs.append({
            "text": "<|im_start|>assistant\n<think>\n",
            "role": "assistant",
            "kind": "generation_prompt",
            "message_start": len(messages),
            "message_end": len(messages),
            "protected": True,
            "metadata": {"generation_prompt": True},
        })
    return specs


def _qwen_prompt_blocks(messages, add_generation_prompt=False):
    return [spec["text"] for spec in _qwen_prompt_block_specs(messages, add_generation_prompt=add_generation_prompt)]


def _prompt_block_cache_for(active):
    if active is None or config.QWEN_PROMPT_BLOCK_CACHE_SIZE <= 0:
        return None
    cache = getattr(active, "_qwen_prompt_block_cache", None)
    if cache is None:
        cache = OrderedDict()
        setattr(active, "_qwen_prompt_block_cache", cache)
    return cache


def _tokenize_qwen_prompt_blocks(tokenizer, blocks, cache=None):
    encoded_blocks = []
    for block in blocks:
        block_ids = None
        if cache is not None:
            block_ids = cache.get(block)
            if block_ids is not None:
                cache.move_to_end(block)
        if block_ids is None:
            block_ids = tuple(tokenizer.encode(block, add_special_tokens=False))
            if cache is not None:
                cache[block] = block_ids
                cache.move_to_end(block)
                while len(cache) > config.QWEN_PROMPT_BLOCK_CACHE_SIZE:
                    cache.popitem(last=False)
        encoded_blocks.append(block_ids)
    return encoded_blocks


def _encode_qwen_prompt_blocks(tokenizer, blocks, cache=None):
    prompt_ids = []
    for block_ids in _tokenize_qwen_prompt_blocks(tokenizer, blocks, cache=cache):
        prompt_ids.extend(block_ids)
    return prompt_ids
