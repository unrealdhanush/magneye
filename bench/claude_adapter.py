"""Run Magneye's agents on Claude for comparison: drop-in replacements for meta.chat_json and meta.chat.

The agents speak OpenAI-style messages (that is what the Meta Model API takes); this adapter translates them to the
Anthropic Messages API and back. It keeps each Claude turn's original content blocks, so replayed assistant turns
carry their thinking blocks unchanged (Claude Opus 5.5 rejects edited history).
"""

from __future__ import annotations

import base64
import json
from typing import Any

import anthropic

MODEL = "claude-opus-5-5"
_client: anthropic.AsyncAnthropic | None = None

# Token usage of every call, in the same shape as magneye.meta.USAGE.
USAGE: list[dict[str, Any]] = []
# Original response content per assistant turn, keyed by the turn's first tool_call id (or its text).
_turns: dict[str, list[dict[str, Any]]] = {}

# Structured outputs accept a subset of JSON Schema; array length bounds are hints here, so drop them.
_UNSUPPORTED = {"minItems", "maxItems"}


def _schema(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _schema(v) for k, v in node.items() if k not in _UNSUPPORTED}
    if isinstance(node, list):
        return [_schema(v) for v in node]
    return node


def _image(url: str) -> dict[str, Any]:
    header, data = url.split(",", 1)
    media = header.removeprefix("data:").split(";")[0]
    return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}


def _content(parts: Any) -> list[dict[str, Any]]:
    if isinstance(parts, str):
        return [{"type": "text", "text": parts}] if parts else []
    out = []
    for part in parts:
        if part["type"] == "text":
            out.append({"type": "text", "text": part["text"]})
        elif part["type"] == "image_url":
            out.append(_image(part["image_url"]["url"]))
    return out


def _key(msg: dict[str, Any]) -> str:
    calls = msg.get("tool_calls") or []
    return calls[0]["id"] if calls else "text:" + (msg.get("content") or "")


def _translate(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """OpenAI-style history -> (system, Anthropic messages). Tool results and any attachments that follow them
    are merged into one user turn, tool results first."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    out: list[dict[str, Any]] = []
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "assistant":
            content = _turns.get(_key(m))
            if content is None:  # a turn this adapter didn't produce
                content = _content(m.get("content") or "") + [
                    {"type": "tool_use", "id": c["id"], "name": c["function"]["name"],
                     "input": json.loads(c["function"]["arguments"] or "{}")} for c in m.get("tool_calls") or []]
            out.append({"role": "assistant", "content": content})
            continue
        blocks = ([{"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}]
                  if m["role"] == "tool" else _content(m["content"]))
        if out and out[-1]["role"] == "user":
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": "user", "content": blocks})
    return system, out


async def _call(**params: Any) -> Any:
    """One streamed request with refusal fallbacks on; records usage and which model actually answered."""
    global _client
    _client = _client or anthropic.AsyncAnthropic()
    async with _client.beta.messages.stream(
        model=MODEL, betas=["server-side-fallback-2026-07-01"], fallbacks="default", **params,
    ) as stream:
        msg = await stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError(f"{MODEL} refused: {msg.stop_details}")
    USAGE.append({"model": msg.model, "prompt_tokens": msg.usage.input_tokens, "completion_tokens": msg.usage.output_tokens,
                  "cache_read": msg.usage.cache_read_input_tokens or 0})
    return msg


async def chat_json(*, model: str, system: str, user_text: str, image_jpeg: bytes | None, schema_name: str,
                    schema: dict[str, Any], reasoning_effort: str = "medium", max_tokens: int = 16000) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if image_jpeg is not None:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.b64encode(image_jpeg).decode()}})
    content.append({"type": "text", "text": user_text})
    params = {"messages": [{"role": "user", "content": content}], "max_tokens": max(max_tokens, 32000)}
    try:
        msg = await _call(system=system, **params,
                          output_config={"effort": reasoning_effort, "format": {"type": "json_schema", "schema": _schema(schema)}})
    except anthropic.BadRequestError as e:
        if "grammar is too large" not in str(e):
            raise
        # Large schemas exceed the structured-output grammar limit. Fall back to the schema in the prompt; the Meta side
        # sends its schema with strict=false too, so neither model is grammar-constrained on these calls.
        USAGE.append({"model": MODEL, "note": f"{schema_name}: schema in prompt (grammar too large)"})
        msg = await _call(system=f"{system}\n\nRespond with only a JSON object (no prose, no code fences) matching this JSON Schema:\n"
                                 f"{json.dumps(_schema(schema))}", **params, output_config={"effort": reasoning_effort})
    text = next(b.text for b in msg.content if b.type == "text")
    return json.loads(text[text.index("{"):text.rindex("}") + 1])


async def chat(*, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
               reasoning_effort: str = "medium", max_tokens: int = 8000) -> dict[str, Any]:
    system, history = _translate(messages)
    specs = [{"name": t["function"]["name"], "description": t["function"]["description"],
              "input_schema": t["function"]["parameters"]} for t in tools]
    msg = await _call(system=system, messages=history, tools=specs, max_tokens=max(max_tokens, 16000),
                      output_config={"effort": reasoning_effort})
    text = "".join(b.text for b in msg.content if b.type == "text")
    calls = [{"id": b.id, "type": "function", "function": {"name": b.name, "arguments": json.dumps(b.input)}}
             for b in msg.content if b.type == "tool_use"]
    reply = {"content": text, "tool_calls": calls}
    _turns[_key({"tool_calls": calls, "content": text})] = [b.model_dump(exclude_none=True) for b in msg.content]
    return reply
