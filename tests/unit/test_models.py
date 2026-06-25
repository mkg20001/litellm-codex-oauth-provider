"""Given the Codex models endpoint, when discovery runs, then slugs/instructions resolve.

This suite covers the live model-discovery layer: fetching the account's models,
caching the result, filtering for listable/API-usable slugs, sourcing per-model
instructions, and degrading gracefully (to the cache) when the backend is
unreachable. Tests follow a Given/When/Then flow and never hit the network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import pytest

from litellm_codex_oauth_provider import models as models_mod
from litellm_codex_oauth_provider.auth import AuthContext

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


SAMPLE_MODELS = [
    {"slug": "gpt-5.5", "visibility": "list", "supported_in_api": True, "base_instructions": "Be GPT-5.5."},
    {"slug": "gpt-5.4-mini", "visibility": "list", "supported_in_api": True, "base_instructions": "Be mini."},
    {"slug": "gpt-5.3-codex-spark", "visibility": "list", "supported_in_api": False},
    {"slug": "codex-auto-review", "visibility": "hide", "supported_in_api": True},
]


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    """Reset the module cache so each test starts from a cold, deterministic state."""
    models_mod._cache.update({"models": None, "fetched_at": 0.0})


def _patch_backend(mocker: MockerFixture, payload: dict[str, Any] | Exception) -> Any:
    """Patch auth + httpx so fetch_models sees ``payload`` (or raises ``payload``)."""
    mocker.patch.object(
        models_mod,
        "get_auth_context",
        return_value=AuthContext(access_token="tok", account_id="acct"),
    )

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return payload  # type: ignore[return-value]

    get_mock = mocker.MagicMock()
    if isinstance(payload, Exception):
        get_mock.side_effect = payload
    else:
        get_mock.return_value = _Resp()

    client_cm = mocker.MagicMock()
    client_cm.__enter__.return_value = mocker.MagicMock(get=get_mock)
    client_cm.__exit__.return_value = False
    mocker.patch.object(models_mod.httpx, "Client", return_value=client_cm)
    return get_mock


def test_fetch_models_returns_backend_list(mocker: MockerFixture) -> None:
    """Given a healthy backend, when fetch_models runs, then it returns the model list.

    Confirms the endpoint payload is parsed into the raw model descriptors callers expect.
    """
    _patch_backend(mocker, {"models": SAMPLE_MODELS})
    result = models_mod.fetch_models(force=True)
    assert [m["slug"] for m in result] == [m["slug"] for m in SAMPLE_MODELS]


def test_fetch_models_sends_client_version(mocker: MockerFixture) -> None:
    """Given the endpoint requires client_version, when fetching, then it is sent as a query param.

    The /codex/models endpoint 400s without client_version, so guard that it is always supplied.
    """
    get_mock = _patch_backend(mocker, {"models": SAMPLE_MODELS})
    models_mod.fetch_models(force=True)
    _, kwargs = get_mock.call_args
    assert "client_version" in kwargs["params"]


def test_fetch_models_is_cached(mocker: MockerFixture) -> None:
    """Given a prior fetch, when fetch_models runs again, then the backend is not re-queried.

    Verifies the in-process TTL cache avoids a network round-trip on every request.
    """
    get_mock = _patch_backend(mocker, {"models": SAMPLE_MODELS})
    models_mod.fetch_models(force=True)
    models_mod.fetch_models()  # cached
    assert get_mock.call_count == 1


def test_fetch_models_falls_back_to_cache_on_error(mocker: MockerFixture) -> None:
    """Given a populated cache, when the backend later fails, then the cached list is returned.

    Ensures transient backend/auth failures don't blank out the model list mid-operation.
    """
    _patch_backend(mocker, {"models": SAMPLE_MODELS})
    models_mod.fetch_models(force=True)
    _patch_backend(mocker, httpx.ConnectError("boom"))
    assert [m["slug"] for m in models_mod.fetch_models(force=True)] == [m["slug"] for m in SAMPLE_MODELS]


def test_fetch_models_empty_on_error_without_cache(mocker: MockerFixture) -> None:
    """Given no cache, when the backend fails, then an empty list is returned (don't block).

    Callers treat an empty list as 'let the backend validate', so discovery must not raise.
    """
    _patch_backend(mocker, httpx.ConnectError("boom"))
    assert models_mod.fetch_models(force=True) == []


def test_available_vs_listable_slugs(mocker: MockerFixture) -> None:
    """Given mixed visibility/api flags, when listing slugs, then filters apply correctly.

    available_model_slugs includes everything usable; listable filters to visible + API-usable.
    """
    _patch_backend(mocker, {"models": SAMPLE_MODELS})
    assert models_mod.available_model_slugs() == [
        "gpt-5.5",
        "gpt-5.4-mini",
        "gpt-5.3-codex-spark",
        "codex-auto-review",
    ]
    assert models_mod.available_model_slugs(api_only=True) == [
        "gpt-5.5",
        "gpt-5.4-mini",
        "codex-auto-review",
    ]
    assert models_mod.listable_model_slugs() == ["gpt-5.5", "gpt-5.4-mini"]


def test_model_instructions_lookup(mocker: MockerFixture) -> None:
    """Given a known slug, when requesting instructions, then its base_instructions are returned.

    Unknown slugs (or models without instructions) return None so callers can fall back.
    """
    _patch_backend(mocker, {"models": SAMPLE_MODELS})
    assert models_mod.model_instructions("gpt-5.5") == "Be GPT-5.5."
    assert models_mod.model_instructions("gpt-5.3-codex-spark") is None
    assert models_mod.model_instructions("does-not-exist") is None


def test_client_version_env_override(mocker: MockerFixture) -> None:
    """Given CODEX_CLIENT_VERSION, when fetching, then the override is used over the default."""
    mocker.patch.dict("os.environ", {"CODEX_CLIENT_VERSION": "9.9.9"})
    get_mock = _patch_backend(mocker, {"models": SAMPLE_MODELS})
    models_mod.fetch_models(force=True)
    _, kwargs = get_mock.call_args
    assert kwargs["params"]["client_version"] == "9.9.9"
