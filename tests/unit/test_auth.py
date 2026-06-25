"""Given Codex auth fixtures, when auth helpers run, then tokens and errors behave predictably.

This suite covers the authentication pipeline end-to-end: loading auth.json, decoding JWT
claims, surfacing expiry, and refreshing tokens. Each test follows a Given/When/Then flow
to assert the provider surfaces clear failures for missing, malformed, or expired data.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from litellm_codex_oauth_provider.auth import (
    _decode_account_id,
    _extract_bearer_token,
    _get_auth_path,
    _load_auth_data,
    _refresh_token,
    get_auth_context,
    get_bearer_token,
)
from litellm_codex_oauth_provider.exceptions import (
    CodexAuthFileNotFoundError,
    CodexAuthRefreshError,
    CodexAuthTokenError,
    CodexAuthTokenExpiredError,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

JWT_SEGMENT_COUNT = 2


# =============================================================================
# TESTS
# =============================================================================
def test_get_auth_path_existing_file(mock_auth_file: Path) -> None:
    """Given an existing auth file, when _get_auth_path runs, then the patched path is returned.

    Verifies that path resolution honors the fixture override so subsequent auth helpers read
    the intended temporary file rather than the real user configuration.
    """
    auth_path = _get_auth_path()
    assert auth_path == mock_auth_file


def test_get_auth_path_missing_file(mocker: MockerFixture) -> None:
    """Given no auth file, when _get_auth_path runs, then a file-not-found error is raised.

    Ensures missing auth configuration surfaces a clear CodexAuthFileNotFoundError instead of
    silently continuing with invalid state.
    """
    mocker.patch(
        "litellm_codex_oauth_provider.constants.DEFAULT_CODEX_AUTH_FILE",
        Path("/nonexistent/auth.json"),
    )
    with pytest.raises(CodexAuthFileNotFoundError):
        _get_auth_path()


def test_load_auth_data_valid(mock_auth_file: Path, mock_auth_data: dict) -> None:
    """Given valid auth JSON, when _load_auth_data runs, then parsed content is returned.

    Confirms JSON loading works end-to-end using the patched auth file and produces the
    in-memory structure expected by downstream token helpers.
    """
    _ = mock_auth_file  # ensure fixture patch is applied
    auth_data = _load_auth_data()
    assert auth_data == mock_auth_data


def test_load_auth_data_invalid_json(mock_auth_file: Path) -> None:
    """Given malformed auth JSON, when loaded, then CodexAuthTokenError is raised.

    Guards against corrupted auth files by asserting JSON parsing failures surface as the
    token-specific error rather than generic exceptions.
    """
    # Write invalid JSON to the file
    with mock_auth_file.open("w") as f:
        f.write("invalid json")

    with pytest.raises(CodexAuthTokenError):
        _load_auth_data()


def test_extract_bearer_token_valid(mock_auth_file: Path) -> None:
    """Given valid auth data, when _extract_bearer_token runs, then the access token is returned.

    Confirms the happy path extracts a JWT-like token without altering its structure.
    """
    _ = mock_auth_file
    token = _extract_bearer_token()
    assert token.count(".") == JWT_SEGMENT_COUNT  # JWT-like structure


def test_extract_bearer_token_missing_token(mock_auth_file: Path) -> None:
    """Given missing access token, when _extract_bearer_token runs, then CodexAuthTokenError is raised.

    Validates that incomplete auth payloads are rejected early to prevent downstream failures.
    """
    # Write auth data without access token
    invalid_data = {"chatgpt": {"refresh_token": "refresh_token"}}
    with mock_auth_file.open("w") as f:
        json.dump(invalid_data, f)

    with pytest.raises(CodexAuthTokenError):
        _extract_bearer_token()


def test_extract_bearer_token_expired_token(mock_auth_file: Path) -> None:
    """Given an expired token, when _extract_bearer_token runs, then CodexAuthTokenExpiredError is raised.

    Ensures timestamp checks block stale tokens so the provider can trigger refresh flows instead.
    """
    # Write auth data with expired token
    expired_data = {
        "chatgpt": {
            "access_token": "expired_token",
            "expires_at": time.time() - 1000,  # Expired 1000 seconds ago
        }
    }
    with mock_auth_file.open("w") as f:
        json.dump(expired_data, f)

    with pytest.raises(CodexAuthTokenExpiredError):
        _extract_bearer_token()


def test_get_bearer_token(mock_auth_file: Path) -> None:
    """Given valid auth data, when get_bearer_token runs, then the bearer token is returned.

    Confirms the public helper forwards to internal extraction and preserves JWT structure.
    """
    _ = mock_auth_file
    token = get_bearer_token()
    assert token.count(".") == JWT_SEGMENT_COUNT


def test_get_auth_context(mock_auth_file: Path) -> None:
    """Given valid auth data, when get_auth_context runs, then token and account id are returned.

    Validates combined token retrieval and JWT claim decoding to keep account metadata in sync.
    """
    _ = mock_auth_file
    context = get_auth_context()
    assert context.access_token.count(".") == JWT_SEGMENT_COUNT
    assert context.account_id == "mock-account"
    assert _decode_account_id(context.access_token) == "mock-account"


def test_refresh_token_missing_refresh_token(mock_auth_file: Path) -> None:
    """Given missing refresh token, when _refresh_token runs, then CodexAuthRefreshError is raised.

    Ensures refresh failures from incomplete auth files surface a clear, actionable error.
    """
    # Write auth data without refresh token
    invalid_data = {"chatgpt": {"access_token": "access_token"}}
    with mock_auth_file.open("w") as f:
        json.dump(invalid_data, f)

    with pytest.raises(CodexAuthRefreshError):
        _refresh_token()


def _jwt(exp: float, account_id: str = "mock-account") -> str:
    """Build an unsigned JWT carrying an exp and the ChatGPT account claim."""
    import base64

    def _enc(part: dict) -> str:
        raw = json.dumps(part, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    payload = {"exp": exp, "https://api.openai.com/auth": {"chatgpt_account_id": account_id}}
    return f"{_enc({'alg': 'none'})}.{_enc(payload)}.sig"


def test_refresh_token_success(mock_auth_file: Path, mocker: MockerFixture) -> None:
    """Given a refresh token, when _refresh_token runs, then it rotates and persists tokens.

    Verifies the OAuth exchange updates access/refresh/id tokens in auth.json in place so the
    proxy keeps working without a manual re-login.
    """
    new_access = _jwt(time.time() + 3600)
    with mock_auth_file.open("w") as f:
        json.dump({"tokens": {"access_token": "old", "refresh_token": "old-refresh"}}, f)

    resp = mocker.Mock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "access_token": new_access,
        "refresh_token": "new-refresh",
        "id_token": "new-id",
        "expires_in": 3600,
    }
    post = mocker.patch("litellm_codex_oauth_provider.auth.httpx.post", return_value=resp)

    returned = _refresh_token()

    assert returned == new_access
    # The OAuth exchange used the existing refresh token.
    assert post.call_args.kwargs["json"]["refresh_token"] == "old-refresh"
    # Rotated credentials were written back to disk.
    persisted = json.loads(mock_auth_file.read_text())["tokens"]
    assert persisted["access_token"] == new_access
    assert persisted["refresh_token"] == "new-refresh"
    assert persisted["id_token"] == "new-id"


def test_get_auth_context_refreshes_expired_jwt(
    mock_auth_file: Path, mocker: MockerFixture
) -> None:
    """Given an expired access token, when get_auth_context runs, then it refreshes transparently.

    Ensures callers never see an expired token: the proxy refreshes proactively from the
    refresh_token instead of failing the request.
    """
    fresh = _jwt(time.time() + 3600)
    with mock_auth_file.open("w") as f:
        json.dump(
            {"tokens": {"access_token": _jwt(time.time() - 10), "refresh_token": "r"}}, f
        )

    resp = mocker.Mock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"access_token": fresh, "expires_in": 3600}
    mocker.patch("litellm_codex_oauth_provider.auth.httpx.post", return_value=resp)

    context = get_auth_context()
    assert context.access_token == fresh
    assert context.account_id == "mock-account"
