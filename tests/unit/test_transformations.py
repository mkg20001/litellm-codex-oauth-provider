"""Given assorted Codex inputs, when transformation helpers run, then model mapping,

reasoning config, and prompt normalization behave as expected.
"""

from __future__ import annotations

from litellm_codex_oauth_provider.model_map import normalize_model
from litellm_codex_oauth_provider.prompts import _to_codex_input_items, derive_instructions
from litellm_codex_oauth_provider.reasoning import apply_reasoning_config


def test_normalize_model_handles_alias_and_suffix() -> None:
    """Given a prefixed legacy model, when normalized, then a codex base name is returned.

    Confirms alias resolution drops provider prefixes and effort suffixes to the canonical model.
    """
    normalized = normalize_model("codex/gpt-5.1-codex-high")
    assert normalized == "gpt-5.1-codex-high"


def test_reasoning_config_clamps_codex_mini() -> None:
    """Given a codex-mini xhigh request, when applied, then effort is clamped to high.

    Verifies family-specific constraints prevent unsupported effort levels for mini models.
    """
    config = apply_reasoning_config(
        original_model="gpt-5.1-codex-mini-xhigh",
        normalized_model="gpt-5.1-codex-mini",
        reasoning_effort=None,
        verbosity=None,
    )
    assert config["reasoning"]["effort"] == "high"
    assert config["text"]["verbosity"] == "medium"


def test_reasoning_config_rewrites_minimal_for_codex() -> None:
    """Given a minimal effort codex request, when applied, then effort becomes low.

    Ensures the clamping rules upgrade too-low efforts to the supported floor for codex.
    """
    config = apply_reasoning_config(
        original_model="gpt-5.1-codex-minimal",
        normalized_model="gpt-5.1-codex",
        reasoning_effort=None,
        verbosity="low",
    )
    assert config["reasoning"]["effort"] == "low"
    assert config["text"]["verbosity"] == "low"


def test_derive_instructions_filters_legacy_toolchain_prompts() -> None:
    """Given legacy toolchain prompts, when deriving instructions, then Codex instructions are kept and legacy prompt is removed.

    Validates system prompt filtering strips legacy toolchain markers while preserving provided instructions and user content.
    """
    instructions, filtered_messages = derive_instructions(
        [
            {"role": "system", "content": "toolchain system prompt content"},
            {"role": "user", "content": "Ping"},
        ],
        normalized_model="gpt-5.1-codex",
        instructions_text="codex instructions",
    )

    assert instructions == "codex instructions"
    assert "toolchain system prompt content" not in instructions
    assert filtered_messages == [{"type": "message", "content": "Ping", "role": "user"}]


def test_to_codex_input_user_message() -> None:
    """A plain user message maps to a single Responses-API ``message`` item."""
    msg = {"role": "user", "content": "Hello", "id": "abc123"}
    items = _to_codex_input_items(msg)
    assert items == [{"type": "message", "content": "Hello", "role": "user"}]


def test_to_codex_input_tool_call() -> None:
    """An assistant ``tool_calls`` message emits a ``function_call`` item with
    ``call_id`` / ``name`` / ``arguments`` at the top level (no nesting)."""
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_xyz",
                "type": "function",
                "function": {"name": "foo", "arguments": {"x": 1}},
            }
        ],
    }
    items = _to_codex_input_items(msg)

    assert items == [
        {
            "type": "function_call",
            "call_id": "call_xyz",
            "name": "foo",
            "arguments": '{"x": 1}',
        }
    ]


def test_to_codex_input_assistant_text_and_tool_call() -> None:
    """Assistant text + tool_calls emit two items, message first then function_call."""
    msg = {
        "role": "assistant",
        "content": "thinking...",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        ],
    }
    items = _to_codex_input_items(msg)
    assert items[0] == {"type": "message", "role": "assistant", "content": "thinking..."}
    assert items[1] == {
        "type": "function_call",
        "call_id": "c1",
        "name": "f",
        "arguments": "{}",
    }


def test_to_codex_input_tool_role_output() -> None:
    """A ``role: tool`` message emits a ``function_call_output`` item with
    ``call_id`` at the top level and ``output`` as a JSON string."""
    msg = {"role": "tool", "tool_call_id": "call-1", "content": {"foo": "bar"}}

    items = _to_codex_input_items(msg)

    assert items == [
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": '{"foo": "bar"}',
        }
    ]


def test_to_codex_input_legacy_function_call() -> None:
    """The deprecated single ``function_call`` field still maps to a function_call item."""
    msg = {
        "role": "assistant",
        "content": None,
        "function_call": {"name": "lookup", "arguments": '{"q":"x"}'},
        "id": "leg-1",
    }
    items = _to_codex_input_items(msg)
    assert items == [
        {
            "type": "function_call",
            "call_id": "leg-1",
            "name": "lookup",
            "arguments": '{"q":"x"}',
        }
    ]


def test_to_codex_input_multiple_tool_calls() -> None:
    """Multiple tool_calls in one assistant message produce one function_call item each."""
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "b", "arguments": '{"k":1}'}},
        ],
    }
    items = _to_codex_input_items(msg)
    assert [it["call_id"] for it in items] == ["c1", "c2"]
    assert all(it["type"] == "function_call" for it in items)
    assert items[1]["arguments"] == '{"k":1}'


def test_to_codex_input_tool_output_string() -> None:
    """A plain-string tool result is forwarded verbatim as ``output``."""
    msg = {"role": "tool", "tool_call_id": "c1", "content": "hello world"}
    items = _to_codex_input_items(msg)
    assert items == [
        {"type": "function_call_output", "call_id": "c1", "output": "hello world"}
    ]


def test_derive_instructions_round_trip_multi_turn_tool_use() -> None:
    """End-to-end: a chat-completions multi-turn with a prior tool_call lowers to
    exactly the item sequence the Responses API requires (this is the shape that
    used to provoke 'Missing required parameter: input[2].call_id')."""
    messages = [
        {"role": "user", "content": "list the cwd"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "arguments": '{"command":["ls","."]}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "a.txt\nb.txt"},
        {"role": "user", "content": "now read b.txt"},
    ]
    _instructions, items = derive_instructions(
        messages, normalized_model="gpt-5.5", instructions_text="x"
    )
    assert items == [
        {"type": "message", "role": "user", "content": "list the cwd"},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "shell",
            "arguments": '{"command":["ls","."]}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "a.txt\nb.txt",
        },
        {"type": "message", "role": "user", "content": "now read b.txt"},
    ]


def test_to_codex_input_empty_assistant_no_calls() -> None:
    """A wholly-empty assistant message (no content, no tool_calls) still emits a
    placeholder so the conversation shape isn't silently lost."""
    items = _to_codex_input_items({"role": "assistant", "content": None})
    assert items == [{"type": "message", "role": "assistant", "content": ""}]
