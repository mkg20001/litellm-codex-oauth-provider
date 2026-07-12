"""Tests for Responses-API -> chat-completions usage normalization."""

from __future__ import annotations

from litellm_codex_oauth_provider.sse_utils import normalize_usage


def test_normalize_usage_maps_responses_api_keys() -> None:
    """Given Codex input/output token counters, when normalizing, then litellm keys are set.

    The backend reports input_tokens/output_tokens; without the mapping the
    prompt/completion splits become 0 and spend is tracked as $0.
    """
    usage = normalize_usage(
        {
            "input_tokens": 3255,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 17,
            "output_tokens_details": {"reasoning_tokens": 10},
            "total_tokens": 3272,
        }
    )

    assert usage == {"prompt_tokens": 3255, "completion_tokens": 17, "total_tokens": 3272}


def test_normalize_usage_keeps_chat_completions_keys() -> None:
    """Given already-normalized usage, when normalizing, then values pass through."""
    usage = normalize_usage({"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12})

    assert usage == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}


def test_normalize_usage_computes_missing_total() -> None:
    """Given no total_tokens, when normalizing, then the total is derived."""
    usage = normalize_usage({"input_tokens": 5, "output_tokens": 7})

    assert usage["total_tokens"] == 12


def test_normalize_usage_empty() -> None:
    """Given no usage, when normalizing, then an empty dict is returned."""
    assert normalize_usage(None) == {}
    assert normalize_usage({}) == {}
