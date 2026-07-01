r"""System prompt handling and Codex instruction derivation.

This module handles the conversion of OpenAI message formats to Codex input format,
including system prompt processing, tool call normalization, and instruction derivation.

The prompt system supports:
- OpenAI to Codex message format conversion
- System prompt filtering and instruction extraction
- Tool call normalization and bridge prompt generation
- Legacy toolchain prompt detection and removal
- Function call output conversion to Codex schema

Message Processing Pipeline
---------------------------
1. **Role-based Processing**: Handle system, user, assistant, and tool messages
2. **Content Extraction**: Convert various content formats to text
3. **Tool Normalization**: Convert OpenAI tool calls to Codex format
4. **System Prompt Handling**: Extract and combine system instructions
5. **Bridge Prompt Addition**: Add tool bridge for function calling

Supported Message Types
-----------------------
- **System Messages**: Converted to Codex instructions
- **User Messages**: Direct conversion to Codex messages
- **Assistant Messages**: Preserved with content and tool calls
- **Tool Messages**: Converted to function_call_output format
- **Function Calls**: Normalized to Codex function_call schema

Tool Call Handling
------------------
The module handles multiple tool call formats:
- OpenAI format: `{"tool_calls": [{"function": {"name": "...", "arguments": "..."}}]}`
- Legacy format: `{"function_call": {"name": "...", "arguments": "..."}}`
- Function output: `{"function_call_output": "..."}`

Examples
--------
Message conversion:

>>> from litellm_codex_oauth_provider.prompts import _to_codex_input
>>> openai_message = {"role": "user", "content": "Hello"}
>>> codex_input = _to_codex_input(openai_message)
>>> print(codex_input)
{'type': 'message', 'content': 'Hello', 'role': 'user'}

Tool call conversion:

>>> tool_message = {"role": "tool", "tool_call_id": "call_123", "content": "Tool result"}
>>> codex_input = _to_codex_input(tool_message)
>>> print(codex_input)
{'type': 'function_call_output', 'output': {'tool_call_id': 'call_123', 'content': 'Tool result'}, 'role': 'assistant'}

Instruction derivation:

>>> from litellm_codex_oauth_provider.prompts import derive_instructions
>>> messages = [
...     {"role": "system", "content": "You are a helpful assistant."},
...     {"role": "user", "content": "Hello"},
... ]
>>> instructions, input_messages = derive_instructions(messages, normalized_model="gpt-5.1-codex")

Tool bridge message:

>>> from litellm_codex_oauth_provider.prompts import build_tool_bridge_message
>>> bridge = build_tool_bridge_message()
>>> print(bridge["content"][0]["text"][:50])
'# Codex Tool Bridge\\n\\nYou are an open-source AI coding assistant...'

Notes
-----
- System prompts are filtered for legacy toolchain markers
- Tool bridge prompts are added when tools are present
- Function calls are normalized to Codex schema
- Content is coerced to text format for consistency
- The module provides both individual conversion and batch derivation functions

See Also
--------
- `provider`: Main provider using these prompt functions
- `remote_resources`: Instruction fetching and caching
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, Final

from . import constants
from .remote_resources import fetch_codex_instructions

DEFAULT_INSTRUCTIONS: Final[str] = constants.DEFAULT_INSTRUCTIONS
LEGACY_TOOLCHAIN_MARKERS: Final[tuple[str, ...]] = (
    "toolchain system prompt",
    "toolchain::system",
    "legacy toolchain",
)
TOOL_BRIDGE_PROMPT: Final[str] = """# Codex Tool Bridge

You are an open-source AI coding assistant with tool support, running behind a developer CLI. \
When tools are provided, prefer invoking them via standard OpenAI tool calls, using the provided \
tool schema exactly. Do not fabricate results—issue tool calls whenever they are needed to satisfy \
the request."""


def _coerce_text(content: Any) -> str:
    """Convert OpenAI content payloads to plain text for inspection.

    Parameters
    ----------
    content : Any
        Content payload which may be a string, mapping, iterable, or None.

    Returns
    -------
    str
        Concatenated text representation of the content.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return _coerce_text(content.get("text") or content.get("content"))
    if isinstance(content, Iterable):
        parts = [_coerce_text(part) for part in content]
        return "\n".join(part for part in parts if part)
    return str(content)


def _is_toolchain_system_prompt(content: str) -> bool:
    """Identify legacy toolchain system prompts that should be filtered in Codex mode.

    Parameters
    ----------
    content : str
        System prompt content to inspect.

    Returns
    -------
    bool
        ``True`` when the prompt matches known legacy markers.
    """
    lowered = content.lower()
    return any(marker in lowered for marker in LEGACY_TOOLCHAIN_MARKERS)


def _strip_message_metadata(message: dict[str, Any]) -> dict[str, Any]:
    """Remove identifiers that are not part of the Codex schema.

    Parameters
    ----------
    message : dict[str, Any]
        Message payload possibly containing metadata.

    Returns
    -------
    dict[str, Any]
        Cleaned message without Codex-incompatible metadata fields.
    """
    return {key: value for key, value in message.items() if key not in {"id", "item_reference"}}


def _drop_stray_function_output(message: dict[str, Any]) -> dict[str, Any]:
    """Remove orphaned function_call_output payloads.

    Parameters
    ----------
    message : dict[str, Any]
        Assistant message that may contain stray ``function_call_output`` fields.

    Returns
    -------
    dict[str, Any]
        Message with stray outputs removed when no matching function call exists.
    """
    if (
        message.get("role") == "assistant"
        and "function_call_output" in message
        and "function_call" not in message
    ):
        cleaned = dict(message)
        cleaned.pop("function_call_output", None)
        return cleaned
    return message


def _clean_message_payload(message: dict[str, Any]) -> dict[str, Any]:
    """Normalize message payload by removing Codex-incompatible metadata.

    Parameters
    ----------
    message : dict[str, Any]
        Message payload to clean.

    Returns
    -------
    dict[str, Any]
        Cleaned message ready for Codex conversion.
    """
    stripped = _strip_message_metadata(message)
    return _drop_stray_function_output(stripped)


def _stringify_arguments(arguments: Any) -> str:
    """Coerce a tool-call arguments payload to a JSON string."""
    if arguments is None:
        return ""
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments)


def _function_call_items(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Emit one Responses-API ``function_call`` item per chat-completions tool call.

    The Responses API requires ``call_id``, ``name`` and ``arguments`` at the top
    level of each ``function_call`` item -- not nested under a ``function_call``
    or ``function`` key. Chat-completions assistant messages carry these as
    ``tool_calls: [{id, type:"function", function:{name, arguments}}, ...]`` (and
    historically as a single legacy ``function_call: {name, arguments}`` field).
    """
    items: list[dict[str, Any]] = []
    for tc in message.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        # New shape: {id, type:"function", function:{name, arguments}}.
        # Legacy/flat shape: {id, name, arguments}.
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        items.append(
            {
                "type": "function_call",
                "call_id": tc.get("id") or tc.get("call_id"),
                "name": fn.get("name"),
                "arguments": _stringify_arguments(fn.get("arguments")),
            }
        )
    legacy = message.get("function_call")
    if isinstance(legacy, dict):
        items.append(
            {
                "type": "function_call",
                "call_id": message.get("id") or message.get("call_id"),
                "name": legacy.get("name"),
                "arguments": _stringify_arguments(legacy.get("arguments")),
            }
        )
    return items


def _function_call_output_item(message: dict[str, Any]) -> dict[str, Any] | None:
    """Emit a ``function_call_output`` item for a tool-role message.

    The Responses API expects ``call_id`` (top level) and ``output`` as a string.
    ``role`` is *not* a valid field on this item type.
    """
    if message.get("role") != "tool":
        return None
    output = message.get("content")
    if isinstance(output, (dict, list)):
        # Structured tool outputs go to the wire as JSON strings.
        output = json.dumps(output)
    elif not isinstance(output, str):
        output = _coerce_text(output)
    return {
        "type": "function_call_output",
        "call_id": message.get("tool_call_id") or message.get("call_id"),
        "output": output,
    }


def _to_responses_content(content: Any, role: str) -> Any:
    """Convert chat-completions message content to Responses-API content.

    Chat-completions part types (``text``, ``image_url``) are rejected by the
    Responses API, which requires ``input_text``/``input_image``/``input_file``
    for input roles (user/system/developer) and ``output_text`` for assistant
    text. A plain string is accepted as-is, so it passes through unchanged; a
    list of parts is remapped part-by-part.
    """
    if not isinstance(content, list):
        return content
    text_type = "output_text" if role == "assistant" else "input_text"
    parts: list[Any] = []
    for part in content:
        if not isinstance(part, dict):
            parts.append({"type": text_type, "text": _coerce_text(part)})
            continue
        ptype = part.get("type")
        if ptype in ("text", "input_text", "output_text"):
            parts.append({"type": text_type, "text": part.get("text", "")})
        elif ptype in ("image_url", "input_image"):
            image_url = part.get("image_url", part.get("url"))
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            parts.append({"type": "input_image", "image_url": image_url})
        else:
            # Already Responses-shaped (e.g. input_file) or unknown: pass through.
            parts.append(part)
    return parts


def _to_codex_input_items(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert one chat-completions message to one-or-more Responses-API input items.

    Tool calls and tool results map to dedicated item types; assistant text and
    tool calls in the same message both get emitted (the message item first, then
    the function_call items, matching the order the model produced them).
    """
    fco = _function_call_output_item(message)
    if fco is not None:
        return [fco]

    items: list[dict[str, Any]] = []
    content = message.get("content")
    role = message.get("role", "user")
    has_text = content not in (None, "", [])
    if has_text:
        items.append(
            {
                "type": "message",
                "role": role,
                "content": _to_responses_content(content, role),
            }
        )
    items.extend(_function_call_items(message))
    if not items:
        # Empty assistant message (e.g. content=None with no tool calls): still
        # emit a placeholder so the conversation shape is preserved.
        items.append(
            {
                "type": "message",
                "role": message.get("role", "user"),
                "content": "",
            }
        )
    return items


def derive_instructions(
    messages: list[dict[str, Any]],
    *,
    normalized_model: str,
    instructions_text: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Extract instructions and convert messages into Codex `input` format.

    Parameters
    ----------
    messages : list of dict
        Chat messages to adapt to the Codex schema.
    normalized_model : str
        Normalized model identifier (reserved for future gating).
    instructions_text : str, optional
        Pre-fetched Codex instructions to prepend.

    Returns
    -------
    tuple[str, list[dict[str, Any]]]
        Combined instruction string and Codex-ready message payloads.
    """
    _ = normalized_model  # reserved for future gating logic
    system_parts: list[str] = []
    input_payload: list[dict[str, Any]] = []

    for message in messages:
        if message.get("role") == "system":
            content = _coerce_text(message.get("content"))
            if not content:
                continue
            if _is_toolchain_system_prompt(content):
                continue
            system_parts.append(content)
            continue

        cleaned = _clean_message_payload(message)
        input_payload.extend(_to_codex_input_items(cleaned))

    base_instructions = instructions_text or DEFAULT_INSTRUCTIONS
    instructions_parts: list[str] = [base_instructions, *system_parts]
    instructions = "\n\n".join(part for part in instructions_parts if part) or DEFAULT_INSTRUCTIONS
    return instructions, input_payload


def build_tool_bridge_message() -> dict[str, Any]:
    """Return the Codex/OpenCode bridge developer message for tool-enabled requests."""
    return {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": TOOL_BRIDGE_PROMPT}],
    }


get_codex_instructions = fetch_codex_instructions
