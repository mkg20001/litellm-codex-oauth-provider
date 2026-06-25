"""Simplified authentication module for the LiteLLM Codex OAuth Provider.

This module provides streamlined OAuth token handling from Codex CLI's auth.json file.
It focuses on core functionality while maintaining essential security and reliability.

Key Features:
- Simple token extraction from auth.json
- JWT account ID decoding
- Basic validation and error handling
- Clean, maintainable code structure

The simplified version removes complex caching layers, multiple format handling,
and extensive validation while preserving core authentication functionality.
"""

from __future__ import annotations

import base64
import datetime
import json
import os
import tempfile
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from . import constants
from .exceptions import (
    CodexAuthFileNotFoundError,
    CodexAuthRefreshError,
    CodexAuthTokenError,
    CodexAuthTokenExpiredError,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class AuthContext:
    """Simplified authentication context from auth.json.

    Attributes
    ----------
    access_token : str
        OAuth bearer token for API authentication
    account_id : str
        ChatGPT account ID extracted from JWT token claims
    """

    access_token: str
    account_id: str


def _get_auth_path() -> Path:
    """Get the path to Codex auth.json file.

    Returns
    -------
    Path
        Path to the Codex auth.json file.

    Raises
    ------
    CodexAuthFileNotFoundError
        If the auth file is not found.
    """
    auth_file = constants.DEFAULT_CODEX_AUTH_FILE
    if not auth_file.exists():
        raise CodexAuthFileNotFoundError(
            f"Codex auth file not found at {auth_file}. Please run 'codex login' first."
        )
    return auth_file


def _load_auth_data() -> dict[str, Any]:
    """Load and parse auth.json from Codex CLI.

    Returns
    -------
    dict[str, Any]
        Parsed auth data from the JSON file.

    Raises
    ------
    CodexAuthTokenError
        If there's an error reading or parsing the auth file.
    """
    auth_path = _get_auth_path()

    try:
        with auth_path.open() as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise CodexAuthTokenError(f"Failed to parse Codex auth data: {e}") from e
    except Exception as e:
        raise CodexAuthTokenError(f"Failed to read Codex auth data: {e}") from e


def _token_container(auth_data: dict[str, Any]) -> dict[str, Any]:
    """Return the sub-dict of auth.json that holds the OAuth tokens.

    Codex CLI writes ``{"tokens": {...}}``; older/alternate formats use
    ``{"chatgpt": {...}}`` or a flat ``{"access_token": ...}``. The returned
    dict is a live reference into ``auth_data``, so callers can mutate it in
    place and persist the whole structure back with :func:`_write_auth_data`.

    Parameters
    ----------
    auth_data : dict[str, Any]
        Parsed contents of auth.json.

    Returns
    -------
    dict[str, Any]
        The dict containing ``access_token`` / ``refresh_token``.

    Raises
    ------
    CodexAuthTokenError
        If no recognised token container is present.
    """
    if "tokens" in auth_data:
        return auth_data["tokens"]
    if "chatgpt" in auth_data:
        return auth_data["chatgpt"]
    if "access_token" in auth_data:
        return auth_data
    raise CodexAuthTokenError(
        "Unsupported Codex auth.json structure. Expected one of: 'tokens', 'chatgpt', or 'access_token' keys."
    )


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    """Decode (without verifying) the payload of a JWT access token.

    Parameters
    ----------
    token : str
        JWT access token.

    Returns
    -------
    dict[str, Any]
        The decoded claims, or an empty dict if the token is not a JWT.
    """
    try:
        _, payload_b64, _ = token.split(".")
        padding = "=" * (-len(payload_b64) % 4)
        return json.loads(base64.urlsafe_b64decode(payload_b64 + padding))
    except Exception:  # noqa: BLE001 - opaque/non-JWT tokens are handled by callers
        return {}


def _token_expiry(token: str) -> float | None:
    """Return the JWT ``exp`` (epoch seconds) of an access token, if present."""
    exp = _decode_jwt_payload(token).get("exp")
    try:
        return float(exp) if exp is not None else None
    except (TypeError, ValueError):
        return None


def _token_needs_refresh(token: str, leeway: float | None = None) -> bool:
    """Whether ``token`` is expired or within ``leeway`` seconds of expiring.

    A token with no decodable ``exp`` claim is treated as not-needing-refresh:
    we cannot prove it is stale, and refreshing eagerly would defeat caching.
    """
    if leeway is None:
        leeway = constants.TOKEN_REFRESH_LEEWAY_SECONDS
    expiry = _token_expiry(token)
    if expiry is None:
        return False
    return time.time() >= (expiry - leeway)


def _write_auth_data(auth_data: dict[str, Any]) -> None:
    """Atomically persist ``auth_data`` back to auth.json, preserving its mode.

    Written via a temp file in the same directory + ``os.replace`` so a crashed
    or concurrent write can never leave a truncated auth.json behind.

    Raises
    ------
    CodexAuthRefreshError
        If the refreshed tokens cannot be written back.
    """
    auth_path = _get_auth_path()
    try:
        mode = auth_path.stat().st_mode & 0o777
    except OSError:
        mode = 0o600

    directory = str(auth_path.parent)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".auth-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(auth_data, f, indent=2)
            f.write("\n")
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, auth_path)
    except Exception as exc:  # noqa: BLE001 - surface as a refresh failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise CodexAuthRefreshError(
            f"Failed to persist refreshed Codex tokens to {auth_path}: {exc}"
        ) from exc


def _extract_bearer_token() -> str:
    """Extract the OAuth bearer token from Codex auth data.

    Simplified version that handles the most common auth.json structure.

    Returns
    -------
    str
        The access token.

    Raises
    ------
    CodexAuthTokenError
        If no access token is found.
    CodexAuthTokenExpiredError
        If the token has expired.
    """
    auth_data = _load_auth_data()

    token_data = _token_container(auth_data)

    access_token = token_data.get("access_token")
    if not access_token:
        raise CodexAuthTokenError("No access_token found in Codex auth data")

    # Check expiry if available
    expires_at = token_data.get("expires_at")
    if expires_at and expires_at < time.time():
        raise CodexAuthTokenExpiredError(
            "Codex OAuth token has expired. Please run 'codex login' to refresh your authentication and get a new token."
        )

    return access_token


def _decode_account_id(access_token: str) -> str:
    """Decode the ChatGPT account ID from the JWT access token.

    Parameters
    ----------
    access_token : str
        JWT access token from auth.json

    Returns
    -------
    str
        ChatGPT account ID

    Raises
    ------
    CodexAuthTokenError
        If account ID cannot be decoded
    """
    try:
        # Decode JWT payload (base64 URL-safe)
        _, payload_b64, _ = access_token.split(".")
        padding = "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + padding))

        # Extract account ID from claims
        account_claim = payload.get(constants.JWT_ACCOUNT_CLAIM, {})
        account_id = account_claim.get("chatgpt_account_id")

        if not account_id:
            raise CodexAuthTokenError("No chatgpt_account_id found in token claims")

        return str(account_id)

    except Exception as exc:
        raise CodexAuthTokenError("Failed to decode ChatGPT account ID from token") from exc


def get_auth_context() -> AuthContext:
    """Get authentication context from Codex auth.json.

    This function provides a simplified authentication flow:
    1. Load and parse auth.json
    2. Extract bearer token
    3. Decode account ID from JWT
    4. Return as AuthContext object

    Returns
    -------
    AuthContext
        Object containing bearer token and account ID.

    Raises
    ------
    CodexAuthFileNotFoundError
        If the Codex CLI auth.json file is not found.
        Please run 'codex login' to authenticate first.
    CodexAuthTokenError
        If there's an issue with token format or decoding.
    CodexAuthTokenExpiredError
        If the access token has expired. Please run 'codex login' to refresh.

    Examples
    --------
    >>> from litellm_codex_oauth_provider.auth import get_auth_context
    >>> context = get_auth_context()
    >>> print(f"Token: {context.access_token[:20]}...")
    >>> print(f"Account ID: {context.account_id}")
    """
    # Extract the current bearer token. A token flagged expired via an explicit
    # `expires_at` field is refreshed rather than fatal.
    try:
        token = _extract_bearer_token()
        stale = _token_needs_refresh(token)
    except CodexAuthTokenExpiredError:
        token = None
        stale = True

    # Refresh proactively when the access token is expired or about to expire.
    if stale:
        token = _refresh_token()

    # Decode account ID from JWT
    account_id = _decode_account_id(token)

    return AuthContext(access_token=token, account_id=account_id)


# Legacy function for backward compatibility
def _decode_account_id_old(access_token: str) -> str:
    """Legacy function name for backward compatibility."""
    return _decode_account_id(access_token)


def _refresh_token() -> str:
    """Refresh the access token using the refresh token.

    This function loads the auth data and attempts to refresh the access token.
    Raises CodexAuthRefreshError if no refresh token is available or refresh fails.

    Returns
    -------
    str
        The new access token

    Raises
    ------
    CodexAuthRefreshError
        If no refresh token is available or refresh fails
    """
    auth_data = _load_auth_data()
    token_data = _token_container(auth_data)
    refresh_token = token_data.get("refresh_token")

    if not refresh_token:
        raise CodexAuthRefreshError(
            "No refresh token available in auth data. "
            "Please ensure your auth.json file includes a 'refresh_token' field."
        )

    # Exchange the refresh token for a fresh access token using the same OAuth
    # endpoint and public client id the Codex CLI uses for `codex login`.
    request = {
        "client_id": constants.CODEX_OAUTH_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "scope": constants.OAUTH_REFRESH_SCOPE,
    }

    try:
        response = httpx.post(
            constants.OAUTH_TOKEN_URL,
            json=request,
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        )
        response.raise_for_status()
        refreshed = response.json()
    except httpx.HTTPStatusError as exc:
        raise CodexAuthRefreshError(
            f"Codex OAuth token refresh failed ({exc.response.status_code}). "
            "The refresh token may be revoked or expired; run 'codex login' again."
        ) from exc
    except httpx.HTTPError as exc:
        raise CodexAuthRefreshError(f"Codex OAuth token refresh request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CodexAuthRefreshError(f"Could not parse Codex OAuth refresh response: {exc}") from exc

    new_access_token = refreshed.get("access_token")
    if not new_access_token:
        raise CodexAuthRefreshError("Codex OAuth refresh response did not include an access_token.")

    # Persist the rotated credentials back, preserving any fields we don't touch.
    token_data["access_token"] = new_access_token
    if refreshed.get("refresh_token"):
        token_data["refresh_token"] = refreshed["refresh_token"]
    if refreshed.get("id_token"):
        token_data["id_token"] = refreshed["id_token"]
    expires_in = refreshed.get("expires_in")
    if expires_in is not None:
        try:
            token_data["expires_at"] = time.time() + float(expires_in)
        except (TypeError, ValueError):
            pass
    # Mirror the Codex CLI, which stamps the refresh time at the top level.
    auth_data["last_refresh"] = (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

    _write_auth_data(auth_data)

    return new_access_token


def get_bearer_token() -> str:
    """Get bearer token from auth context.

    This is a convenience function that extracts just the bearer token
    from the authentication context.

    Returns
    -------
    str
        The bearer token for API authentication

    Raises
    ------
    Exception
        Any exception raised by get_auth_context()
    """
    context = get_auth_context()
    return context.access_token
