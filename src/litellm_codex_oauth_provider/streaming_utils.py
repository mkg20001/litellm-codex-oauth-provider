"""Streaming utilities for building LiteLLM-compatible streaming chunks.

This module provides functions for converting SSE events into GenericStreamingChunk
objects that are compatible with LiteLLM's streaming interface. It handles text
deltas, tool call arguments, and final chunks with proper formatting.

Key Functions:
- build_text_chunk: Create streaming chunk for text delta
- build_tool_call_chunk: Create streaming chunk for tool arguments
- build_final_chunk: Create final streaming chunk with usage
"""

from __future__ import annotations

from typing import Any, TypedDict

from litellm.types.utils import ChatCompletionUsageBlock, GenericStreamingChunk


def _usage_block(usage: dict[str, int] | None) -> ChatCompletionUsageBlock | None:
    """Build a GenericStreamingChunk usage block (a plain dict).

    GenericStreamingChunk.usage must be a ChatCompletionUsageBlock (a TypedDict),
    not a Usage model -- litellm's stream handler does ``Usage(**chunk.usage)`` and
    crashes if it gets a Usage object.
    """
    if not usage:
        return None
    return ChatCompletionUsageBlock(
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        total_tokens=int(usage.get("total_tokens", 0)),
    )


# Tool call chunk structures as defined in comments.md
class ChatCompletionToolCallFunctionChunk(TypedDict):
    """Function chunk in tool call."""

    name: str | None  # Function name, can be None for streaming deltas
    arguments: str  # JSON string of arguments


class ChatCompletionToolCallChunk(TypedDict):
    """Complete tool call chunk."""

    id: str | None  # Unique identifier for the tool call
    type: str  # Always "function" for function calls
    index: int  # Index in the response
    function: ChatCompletionToolCallFunctionChunk


def build_text_chunk(
    text_delta: str, index: int = 0, is_finished: bool = False, finish_reason: str | None = None
) -> GenericStreamingChunk:
    """Build streaming chunk for text delta.

    Parameters
    ----------
    text_delta : str
        Incremental text content from SSE event
    index : int, default 0
        Choice index in the response
    is_finished : bool, default False
        Whether this is the final chunk
    finish_reason : str | None
        Final reason for completion (stop, tool_calls, etc.)

    Returns
    -------
    GenericStreamingChunk
        Formatted streaming chunk for text content
    """
    return GenericStreamingChunk(
        text=text_delta,
        tool_use=None,
        is_finished=is_finished,
        finish_reason=finish_reason,
        index=index,
        usage=None,
    )


def build_reasoning_chunk(reasoning_delta: str, index: int = 0) -> GenericStreamingChunk:
    """Build streaming chunk for reasoning delta."""
    return GenericStreamingChunk(
        text="",
        tool_use=None,
        is_finished=False,
        finish_reason=None,
        index=index,
        usage=None,
        reasoning_content=reasoning_delta,
    )


def build_tool_call_chunk(
    call_id: str | None,
    name: str | None,
    arguments: str,
    is_final: bool = False,
    index: int = 0,
    tool_index: int | None = None,
) -> GenericStreamingChunk:
    """Build streaming chunk for tool call arguments.

    Chat-completions streaming clients accumulate ``function.name`` and
    ``function.arguments`` by concatenation, keyed by the tool call's position
    (``tool_use.index``). So only a call's first chunk may carry ``id`` and
    ``name``; later argument deltas pass ``None`` for both, or the client glues
    the name onto itself once per delta.

    Parameters
    ----------
    call_id : str | None
        Unique identifier for the tool call (first chunk only)
    name : str | None
        Name of the function being called (first chunk only)
    arguments : str
        JSON string of function arguments (can be incremental)
    is_final : bool, default False
        Whether this is the final chunk for this tool call
    index : int, default 0
        Choice index in the response
    tool_index : int | None
        Position of this tool call within the response; distinct per parallel
        call so clients don't merge them. Defaults to ``index``.

    Returns
    -------
    GenericStreamingChunk
        Formatted streaming chunk for tool call
    """
    tool_call = ChatCompletionToolCallChunk(
        id=call_id,
        type="function",
        index=index if tool_index is None else tool_index,
        function=ChatCompletionToolCallFunctionChunk(name=name, arguments=arguments),
    )

    return GenericStreamingChunk(
        text="",
        tool_use=tool_call,
        is_finished=is_final,
        finish_reason="tool_calls" if is_final else None,
        index=index,
        usage=None,
    )


def build_tool_arguments_delta(arguments_delta: str, index: int = 0) -> GenericStreamingChunk:
    """Build streaming chunk for tool arguments delta.

    This is for incremental streaming of tool arguments when they arrive in chunks.

    Parameters
    ----------
    arguments_delta : str
        Incremental JSON arguments (partial string that gets concatenated)
    index : int, default 0
        Choice index in the response

    Returns
    -------
    GenericStreamingChunk
        Formatted streaming chunk for incremental tool arguments
    """
    tool_call = ChatCompletionToolCallChunk(
        id=None,  # ID can be None for incremental deltas
        type="function",
        index=index,
        function=ChatCompletionToolCallFunctionChunk(
            name=None,  # Name can be None for incremental deltas
            arguments=arguments_delta,
        ),
    )

    return GenericStreamingChunk(
        text="", tool_use=tool_call, is_finished=False, finish_reason=None, index=index, usage=None
    )


def build_final_chunk(
    usage: dict[str, int] | None, finish_reason: str, index: int = 0
) -> GenericStreamingChunk:
    """Build final streaming chunk with usage and finish reason.

    Parameters
    ----------
    usage : dict[str, int] | None
        Usage statistics (prompt_tokens, completion_tokens, total_tokens)
    finish_reason : str
        Reason for completion (stop, tool_calls, length, etc.)
    index : int, default 0
        Choice index in the response

    Returns
    -------
    GenericStreamingChunk
        Final streaming chunk with usage and completion status
    """
    return GenericStreamingChunk(
        text="",
        tool_use=None,
        is_finished=True,
        finish_reason=finish_reason,
        index=index,
        usage=_usage_block(usage),
    )


def build_completion_text_chunk(
    text: str, usage: dict[str, int] | None, finish_reason: str, index: int = 0
) -> GenericStreamingChunk:
    """Build streaming chunk for completed text response.

    This is used when converting non-streaming responses to streaming format.

    Parameters
    ----------
    text : str
        Final text content
    usage : dict[str, int] | None
        Usage statistics
    finish_reason : str
        Reason for completion
    index : int, default 0
        Choice index in the response

    Returns
    -------
    GenericStreamingChunk
        Formatted completion chunk
    """
    return GenericStreamingChunk(
        text=text,
        tool_use=None,
        is_finished=True,
        finish_reason=finish_reason,
        index=index,
        usage=_usage_block(usage),
    )


class ToolCallTracker:
    """Helper class to track tool call state during streaming.

    This class manages the state of tool calls as they stream incrementally,
    allowing proper accumulation of arguments and final tool call chunk creation.
    """

    def __init__(self) -> None:
        self._active_calls: dict[str, dict[str, Any]] = {}
        # Calls started so far; each call's position (its chat-completions
        # ``tool_calls[].index``), so parallel calls stream into separate slots.
        self._started = 0

    def start_tool_call(self, key: str, name: str, call_id: str | None = None) -> None:
        """Start tracking a new tool call.

        Parameters
        ----------
        key : str
            Tracker lookup key. For Responses-API streaming this is the
            ``item_id`` (the only id carried on argument-delta events).
        name : str
            Name of the function being called.
        call_id : str | None
            Tool-call id surfaced to chat-completions clients (``tool_call.id``).
            Defaults to ``key`` when callers don't have a separate value.
        """
        self._active_calls[key] = {
            "name": name,
            "arguments": "",
            "call_id": call_id or key,
            "index": self._started,
        }
        self._started += 1

    def add_arguments_delta(self, call_id: str, arguments_delta: str) -> None:
        """Add incremental arguments to a tool call.

        Parameters
        ----------
        call_id : str
            Unique identifier for the tool call
        arguments_delta : str
            Incremental JSON arguments to append
        """
        if call_id in self._active_calls:
            self._active_calls[call_id]["arguments"] += arguments_delta

    def finalize_tool_call(self, call_id: str) -> ChatCompletionToolCallChunk | None:
        """Get final tool call chunk and remove from tracking.

        Parameters
        ----------
        call_id : str
            Unique identifier for the tool call

        Returns
        -------
        ChatCompletionToolCallChunk | None
            Complete tool call chunk or None if not found
        """
        if call_id not in self._active_calls:
            return None

        call_data = self._active_calls.pop(call_id)

        return ChatCompletionToolCallChunk(
            id=call_data.get("call_id", call_id),
            type="function",
            index=call_data["index"],
            function=ChatCompletionToolCallFunctionChunk(
                name=call_data["name"], arguments=call_data["arguments"]
            ),
        )

    @property
    def started_count(self) -> int:
        """Number of tool calls started so far (including finalized ones)."""
        return self._started

    def get_active_calls(self) -> dict[str, dict[str, Any]]:
        """Get currently active tool calls.

        Returns
        -------
        dict[str, dict[str, Any]]
            Active tool calls with their current state
        """
        return self._active_calls.copy()

    def clear(self) -> None:
        """Clear all active tool calls."""
        self._active_calls.clear()
        self._started = 0
