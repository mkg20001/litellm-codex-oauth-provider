"""Client-side inlining of remote image URLs for vision requests.

The Codex ``/responses`` backend is built for the Codex CLI, which only ever
attaches images as base64 ``data:`` URLs (local files). Its server-side URL
fetcher is unreliable for arbitrary public URLs -- e.g. a plain Wikimedia
thumbnail fails with::

    {"message": "Error while downloading file. Upstream status code: 400.",
     "param": "url", "code": "invalid_value"}

So instead of forwarding ``http(s)`` image URLs and hoping the backend can
fetch them, download them here and rewrite the ``input_image`` part to a
``data:`` URL, which the backend accepts (verified end-to-end).

Failures are non-fatal: an image that can't be downloaded keeps its original
URL so the backend produces its own (accurate) error message.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Codex CLI's own attachment ceiling is generous; cap ours to keep a single
# hostile URL from ballooning the request payload.
MAX_IMAGE_BYTES = 20 * 1024 * 1024
FETCH_TIMEOUT_SECONDS = 20.0

# Fallback when the server sends no usable Content-Type; the backend only
# needs *a* plausible image MIME to parse the data URL.
DEFAULT_IMAGE_MIME = "image/jpeg"

# CDNs commonly reject library-default user agents (Wikimedia 403s
# "python-httpx/x.y" outright); a descriptive UA with a contact URL satisfies
# their bot policies.
USER_AGENT = (
    "Mozilla/5.0 (compatible; litellm-codex-oauth-provider/0.3; "
    "+https://github.com/mkg20001/litellm-codex-oauth-provider)"
)


def _is_remote_url(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def _iter_remote_image_parts(input_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect ``input_image`` parts whose ``image_url`` is a remote URL."""
    found: list[dict[str, Any]] = []
    for item in input_items:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "input_image"
                and _is_remote_url(part.get("image_url"))
            ):
                found.append(part)
    return found


def _to_data_url(content_type: str | None, body: bytes) -> str:
    mime = (content_type or "").split(";")[0].strip().lower()
    if not mime.startswith("image/"):
        mime = DEFAULT_IMAGE_MIME
    return f"data:{mime};base64,{base64.b64encode(body).decode('ascii')}"


async def inline_remote_images(input_items: list[dict[str, Any]]) -> None:
    """Rewrite remote image URLs in ``input_items`` to base64 ``data:`` URLs.

    Mutates the ``input_image`` parts in place (the payload is
    request-local, built by ``derive_instructions``). Download failures are
    logged and the part is left untouched.
    """
    parts = _iter_remote_image_parts(input_items)
    if not parts:
        return
    async with httpx.AsyncClient(
        timeout=FETCH_TIMEOUT_SECONDS,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        for part in parts:
            url = part["image_url"]
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                if len(resp.content) > MAX_IMAGE_BYTES:
                    logger.warning(
                        "image at %s exceeds %d bytes; forwarding URL as-is",
                        url,
                        MAX_IMAGE_BYTES,
                    )
                    continue
                part["image_url"] = _to_data_url(
                    resp.headers.get("content-type"), resp.content
                )
            except Exception as exc:  # noqa: BLE001 - non-fatal, backend re-reports
                logger.warning("failed to inline image %s: %s", url, exc)
