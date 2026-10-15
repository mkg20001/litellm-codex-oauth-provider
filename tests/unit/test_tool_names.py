"""Given tool names longer than the backend allows, when a request is built,

then names are shortened consistently across definitions and replayed history,
and restored on the response side.
"""

from __future__ import annotations

from litellm_codex_oauth_provider.prompts import _function_call_items
from litellm_codex_oauth_provider.provider import CodexAuthProvider, _normalize_tools
from litellm_codex_oauth_provider.streaming_utils import ToolCallTracker
from litellm_codex_oauth_provider.tool_names import (
    MAX_TOOL_NAME_LENGTH,
    build_tool_name_map,
    restore_tool_name,
    shorten_tool_name,
)

# MCP-style namespaced name of 144 characters -- the shape the backend rejected
# with "Invalid 'input[17].name': string too long".
LONG_NAME = "mcp__" + "a" * 139


def test_shorten_leaves_names_within_the_limit_untouched() -> None:
    """Given a name at or below the limit, when shortened, then it is unchanged."""
    assert shorten_tool_name("bash") == "bash"
    assert shorten_tool_name("x" * MAX_TOOL_NAME_LENGTH) == "x" * MAX_TOOL_NAME_LENGTH


def test_shorten_fits_the_limit_and_is_deterministic() -> None:
    """Given an over-long name, when shortened, then it fits and repeats identically."""
    assert len(LONG_NAME) == 144
    shortened = shorten_tool_name(LONG_NAME)
    assert len(shortened) == MAX_TOOL_NAME_LENGTH
    assert shortened == shorten_tool_name(LONG_NAME)
    assert shortened.startswith(LONG_NAME[:64])


def test_shorten_distinguishes_names_sharing_a_prefix() -> None:
    """Given two names differing only past the truncation point, then outputs differ.

    The appended digest covers the full name, so a shared prefix can't collide.
    """
    first = shorten_tool_name("p" * 130 + "_one")
    second = shorten_tool_name("p" * 130 + "_two")
    assert first != second


def test_normalize_tools_shortens_the_definition_name() -> None:
    """Given an over-long tool definition, when normalized, then its name fits."""
    tools = _normalize_tools(
        [{"type": "function", "function": {"name": LONG_NAME, "parameters": {"type": "object"}}}]
    )
    assert tools is not None
    assert tools[0]["name"] == shorten_tool_name(LONG_NAME)
    assert len(tools[0]["name"]) == MAX_TOOL_NAME_LENGTH


def test_replayed_tool_call_shortens_to_the_definition_name() -> None:
    """Given history referencing an over-long tool, then the item matches the definition.

    A later request replays the assistant's tool call; its ``name`` has to shorten
    exactly the way the tool definition did or the backend rejects the input item.
    """
    items = _function_call_items(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": LONG_NAME, "arguments": '{"a": 1}'},
                }
            ],
        }
    )
    tools = _normalize_tools([{"type": "function", "function": {"name": LONG_NAME}}])
    assert tools is not None
    assert items[0]["name"] == tools[0]["name"]
    assert len(items[0]["name"]) == MAX_TOOL_NAME_LENGTH


def test_name_map_round_trips_only_shortened_names() -> None:
    """Given mixed tool names, when mapped, then only shortened ones are tracked."""
    mapping = build_tool_name_map(
        [
            {"type": "function", "function": {"name": LONG_NAME}},
            {"type": "function", "function": {"name": "bash"}},
        ]
    )
    assert mapping == {shorten_tool_name(LONG_NAME): LONG_NAME}
    assert restore_tool_name(shorten_tool_name(LONG_NAME), mapping) == LONG_NAME
    # Names the caller sent as-is, and unknown names, pass through untouched.
    assert restore_tool_name("bash", mapping) == "bash"
    assert restore_tool_name("bash", {}) == "bash"


def test_streaming_tool_call_reports_the_callers_name() -> None:
    """Given a shortened name in an SSE event, when streamed, then the original is emitted."""
    provider = CodexAuthProvider()
    tracker = ToolCallTracker()
    mapping = build_tool_name_map([{"type": "function", "function": {"name": LONG_NAME}}])
    event = {
        "type": "function_call_started",
        "item_id": "fc_1",
        "call_id": "call_1",
        "name": shorten_tool_name(LONG_NAME),
    }

    chunk = provider._handle_tool_call_started(event, tracker, mapping)  # noqa: SLF001

    assert chunk is not None
    assert chunk["tool_use"]["function"]["name"] == LONG_NAME
    assert tracker.get_active_calls()["fc_1"]["name"] == LONG_NAME
