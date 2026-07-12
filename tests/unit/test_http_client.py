"""Tests for the Codex HTTP client."""

from __future__ import annotations

from typing import TYPE_CHECKING

from litellm_codex_oauth_provider import constants
from litellm_codex_oauth_provider.http_client import CodexAPIClient

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _client() -> CodexAPIClient:
    return CodexAPIClient(token_provider=lambda: "tok", account_id_provider=lambda: "acct")


def test_build_headers_includes_client_version() -> None:
    """Given no override, when building headers, then the default version is sent.

    The /codex/responses endpoint hides models newer than the reported client
    version (404 "Model not found"), so every request must carry it.
    """
    headers = _client()._build_headers()

    assert headers[constants.VERSION_HEADER] == constants.CODEX_CLIENT_VERSION


def test_build_headers_client_version_env_override(mocker: MockerFixture) -> None:
    """Given CODEX_CLIENT_VERSION, when building headers, then the override is sent."""
    mocker.patch.dict("os.environ", {"CODEX_CLIENT_VERSION": "9.9.9"})

    headers = _client()._build_headers()

    assert headers[constants.VERSION_HEADER] == "9.9.9"
