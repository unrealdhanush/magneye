"""A small tool-calling loop shared by Magneye's agents, emitting a trace the UI streams live."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from . import meta

Emit = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class ToolResult:
    """What a tool hands back to the model, plus what the trace should show."""

    content: str
    summary: str
    ok: bool = True
    # Extra user-message content (e.g. an image) appended after the tool results, since tool
    # messages carry text only.
    attach: list[dict[str, Any]] = field(default_factory=list)
    # Ends the loop after this turn (e.g. the agent asked the user a question or finished).
    stop: bool = False


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[[dict[str, Any]], Awaitable[ToolResult]]

    def spec(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}


def params(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props) if required is None else required}


async def run_loop(
    *,
    agent: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[Tool],
    emit: Emit,
    max_turns: int = 8,
    reasoning_effort: str = "medium",
) -> str:
    """Run model turns until it answers without tools, a tool stops the loop, or turns run out.

    `messages` is extended in place so a caller can persist and resume the conversation.
    Returns the model's final text (may be empty when a tool stopped the loop).
    """
    by_name = {t.name: t for t in tools}
    for _ in range(max_turns):
        msg = await meta.chat(model=model, messages=messages, tools=[t.spec() for t in tools],
                              reasoning_effort=reasoning_effort)
        calls = msg.get("tool_calls") or []
        messages.append({"role": "assistant", "content": msg.get("content") or "", **({"tool_calls": calls} if calls else {})})
        if msg.get("content") and calls:
            await emit({"type": "thought", "agent": agent, "text": msg["content"]})
        if not calls:
            return msg.get("content") or ""

        attachments: list[dict[str, Any]] = []
        stop = False
        for call in calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = None
            tool = by_name.get(name)
            if tool is None or not isinstance(args, dict):
                result = ToolResult(f"Error: unknown tool or invalid JSON arguments for {name!r}.", f"{name}: bad call", ok=False)
            else:
                await emit({"type": "step", "agent": agent, "tool": name, "status": "running"})
                try:
                    result = await tool.run(args)
                except Exception as e:  # a failing tool is information for the model, not a crash
                    result = ToolResult(f"Error: {e}", f"{name} failed: {e}", ok=False)
            await emit({"type": "step", "agent": agent, "tool": name, "status": "done" if result.ok else "error",
                        "text": result.summary})
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result.content})
            attachments += result.attach
            stop = stop or result.stop
        if attachments:
            messages.append({"role": "user", "content": attachments})
        if stop:
            return ""
    await emit({"type": "step", "agent": agent, "tool": "limit", "status": "error", "text": f"Stopped after {max_turns} turns"})
    return ""
