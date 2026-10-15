"""Bridge chat-completions tool names that exceed the Codex name limit.

The Codex ``/responses`` backend caps a function name at 128 characters and
enforces it on *every* occurrence: the ``tools`` definitions and each
``function_call`` item replayed in ``input``. Chat-completions clients routinely
exceed it -- MCP tool names are namespaced (``mcp__<server>__<tool>``) and run
well past 128 -- and the backend then rejects the whole request with::

    HTTP 400 Invalid 'input[17].name': string too long. Expected a string with
    maximum length 128, but got a string with length 144 instead.

Over-long names are therefore shortened on the way out and restored on the way
back, so a caller only ever sees the names it sent.

The shortening is a pure function of the name alone -- a truncated prefix plus a
digest of the full name. That matters for two reasons: the same tool shortens
identically in the definitions and in conversation history replayed by a later
request (which may no longer offer that tool at all), and the digest keeps two
tools that share a 119-character prefix distinct.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, Final

# Backend limit on `tools[].name` and `input[].name`.
MAX_TOOL_NAME_LENGTH: Final[int] = 128
# Hex digest characters appended (plus a "_" separator) to keep shortened names
# unique. Short enough to leave a recognizable prefix, long enough that a
# collision within one request is not a practical concern.
_DIGEST_LENGTH: Final[int] = 8


def shorten_tool_name(name: Any) -> Any:
    """Return ``name`` shortened to the backend limit, or unchanged if it fits.

    Non-string input (a malformed tool call) is passed through untouched so the
    backend reports the real problem rather than this helper masking it.
    """
    if not isinstance(name, str) or len(name) <= MAX_TOOL_NAME_LENGTH:
        return name
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:_DIGEST_LENGTH]
    prefix_length = MAX_TOOL_NAME_LENGTH - _DIGEST_LENGTH - 1
    return f"{name[:prefix_length]}_{digest}"


def _tool_name(tool: Any) -> str | None:
    """Pull the function name out of either tool shape, or ``None``.

    Mirrors the precedence normalization uses (a top-level ``name`` wins over
    ``function.name``), so the map is keyed on the name actually sent.
    """
    if not isinstance(tool, Mapping):
        return None
    name = tool.get("name")
    if not isinstance(name, str):
        function_payload = tool.get("function")
        name = function_payload.get("name") if isinstance(function_payload, Mapping) else None
    return name if isinstance(name, str) else None


def build_tool_name_map(tools: Any) -> dict[str, str]:
    """Map shortened name -> original, for the tools that needed shortening.

    Accepts either chat-completions (``{"function": {"name": ...}}``) or flat
    (``{"name": ...}``) tool definitions. Tools whose names already fit are
    omitted, so the map is empty for the common case.
    """
    mapping: dict[str, str] = {}
    for tool in tools or []:
        name = _tool_name(tool)
        if name is None:
            continue
        shortened = shorten_tool_name(name)
        if shortened != name:
            mapping[shortened] = name
    return mapping


def restore_tool_name(name: Any, mapping: Mapping[str, str] | None) -> Any:
    """Undo :func:`shorten_tool_name` for a name the backend called back with."""
    if not mapping or not isinstance(name, str):
        return name
    return mapping.get(name, name)
