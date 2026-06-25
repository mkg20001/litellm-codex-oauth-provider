"""Live Codex model discovery.

Instead of hard-coding a model list, query the ChatGPT Codex backend for the
models the authenticated account can actually use. The same endpoint also
returns each model's canonical ``base_instructions``, so we source instructions
from it too rather than scraping GitHub.

The result is cached in-process with a short TTL; on any network/auth failure we
fall back to the last good cache (or an empty list, which callers treat as
"don't block locally -- let the backend decide").
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import httpx

from . import constants
from .auth import get_auth_context

_CACHE_TTL_SECONDS = 15 * 60

_lock = threading.Lock()
_cache: dict[str, Any] = {"models": None, "fetched_at": 0.0}


def _client_version() -> str:
    """Codex CLI version reported to the models endpoint (it requires one)."""
    return os.getenv("CODEX_CLIENT_VERSION", constants.CODEX_CLIENT_VERSION)


def fetch_models(*, force: bool = False) -> list[dict[str, Any]]:
    """Return the raw model objects from the Codex backend, cached with a TTL.

    Parameters
    ----------
    force : bool
        Bypass the cache and re-fetch.

    Returns
    -------
    list[dict[str, Any]]
        Model descriptors (``slug``, ``base_instructions``, ``visibility``,
        ``supported_in_api``, ...). Empty list if discovery fails and nothing is
        cached.
    """
    now = time.time()
    with _lock:
        cached = _cache["models"]
        if not force and cached is not None and (now - _cache["fetched_at"]) < _CACHE_TTL_SECONDS:
            return cached

    try:
        ctx = get_auth_context()
        headers = {
            "Authorization": f"Bearer {ctx.access_token}",
            constants.CHATGPT_ACCOUNT_HEADER: ctx.account_id,
            constants.OPENAI_BETA_HEADER: constants.OPENAI_BETA_VALUE,
            constants.OPENAI_ORIGINATOR_HEADER: constants.OPENAI_ORIGINATOR_VALUE,
        }
        url = f"{constants.CODEX_API_BASE_URL.rstrip('/')}/codex/models"
        with httpx.Client(timeout=20.0) as client:
            resp = client.get(url, params={"client_version": _client_version()}, headers=headers)
            resp.raise_for_status()
            models = resp.json().get("models", []) or []
    except Exception:  # noqa: BLE001 - discovery is best-effort; fall back to cache
        with _lock:
            return _cache["models"] or []

    with _lock:
        _cache["models"] = models
        _cache["fetched_at"] = now
    return models


def available_model_slugs(*, api_only: bool = False) -> list[str]:
    """Slugs of all discoverable models (optionally only API-usable ones)."""
    out = []
    for m in fetch_models():
        slug = m.get("slug")
        if not slug:
            continue
        if api_only and not m.get("supported_in_api", True):
            continue
        out.append(slug)
    return out


def listable_model_slugs() -> list[str]:
    """Slugs suitable for exposing as proxy models (visible + API-usable)."""
    return [
        m["slug"]
        for m in fetch_models()
        if m.get("slug")
        and m.get("supported_in_api", True)
        and m.get("visibility", "list") == "list"
    ]


def model_instructions(slug: str) -> str | None:
    """Canonical ``base_instructions`` for ``slug``, if the backend provides them."""
    for m in fetch_models():
        if m.get("slug") == slug:
            instructions = m.get("base_instructions")
            if isinstance(instructions, str) and instructions.strip():
                return instructions
    return None
