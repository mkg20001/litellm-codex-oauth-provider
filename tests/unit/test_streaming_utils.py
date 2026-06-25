"""Given streaming chunk builders, when emitting usage, then it is a plain dict.

litellm's stream handler does ``Usage(**chunk["usage"])``; if the provider puts a
Usage *model* into GenericStreamingChunk.usage it crashes mid-stream with
"argument after ** must be a mapping, not Usage". These tests pin the usage block
to a ChatCompletionUsageBlock (a TypedDict / plain dict).
"""

from __future__ import annotations

from litellm_codex_oauth_provider.streaming_utils import (
    build_completion_text_chunk,
    build_final_chunk,
)

USAGE = {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}


def test_build_final_chunk_usage_is_mapping() -> None:
    """Given usage stats, when building the final chunk, then usage is a splattable dict."""
    chunk = build_final_chunk(USAGE, "stop")
    usage = chunk["usage"]
    assert isinstance(usage, dict)
    # Must be safe to splat the way litellm does.
    assert {**usage} == {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}
    assert chunk["is_finished"] is True


def test_build_completion_text_chunk_usage_is_mapping() -> None:
    """Given a completed text chunk, when built, then usage is a splattable dict."""
    chunk = build_completion_text_chunk("hello", USAGE, "stop")
    assert isinstance(chunk["usage"], dict)
    assert {**chunk["usage"]}["total_tokens"] == 8
    assert chunk["text"] == "hello"


def test_final_chunk_without_usage_is_none() -> None:
    """Given no usage, when building the final chunk, then usage is None (not an empty model)."""
    assert build_final_chunk(None, "stop")["usage"] is None
