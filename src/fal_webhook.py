"""Utilities for validating fal.ai webhook signatures."""

from __future__ import annotations

import base64
import hashlib
import os
import time
from typing import Callable, Iterable, Mapping, Sequence

import requests
from nacl.exceptions import BadSignatureError
from nacl.signing import VerifyKey


__all__ = [
    "FalWebhookVerificationError",
    "fetch_jwks",
    "verify_fal_webhook",
]


JWKS_URL = os.getenv(
    "FAL_JWKS_URL",
    "https://rest.alpha.fal.ai/.well-known/jwks.json",
)

_MAX_CACHE_SECONDS = 24 * 60 * 60
try:
    _configured_cache = int(os.getenv("FAL_JWKS_CACHE_SECONDS", "3600"))
except ValueError:  # pragma: no cover - invalid env configuration
    _configured_cache = 3600

JWKS_CACHE_SECONDS = min(max(_configured_cache, 60), _MAX_CACHE_SECONDS)

_jwks_cache: list[dict[str, object]] | None = None
_jwks_cache_expiry: float = 0.0


class FalWebhookVerificationError(RuntimeError):
    """Raised when a fal.ai webhook signature cannot be validated."""


def _decode_base64url(data: str) -> bytes:
    padding = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def fetch_jwks(force_refresh: bool = False) -> list[dict[str, object]]:
    """Fetch the JSON Web Key Set used to verify webhook signatures."""

    global _jwks_cache, _jwks_cache_expiry
    now = time.time()
    if (
        force_refresh
        or _jwks_cache is None
        or now >= _jwks_cache_expiry
    ):
        response = requests.get(JWKS_URL, timeout=10)
        response.raise_for_status()
        payload = response.json()
        keys = payload.get("keys")
        if not isinstance(keys, list):
            keys = []
        _jwks_cache = keys  # type: ignore[assignment]
        _jwks_cache_expiry = now + JWKS_CACHE_SECONDS
    return _jwks_cache or []


def _required_header(headers: Mapping[str, str], name: str) -> str:
    value = headers.get(name)
    if value is None:
        raise FalWebhookVerificationError(f"missing header: {name}")
    return value


def verify_fal_webhook(
    headers: Mapping[str, str],
    body: bytes,
    *,
    fetch_keys: Callable[[], Sequence[Mapping[str, object]]] = fetch_jwks,
    now: float | None = None,
) -> None:
    """Validate the webhook signature.

    Raises ``FalWebhookVerificationError`` when the request cannot be
    authenticated.
    """

    request_id = _required_header(headers, "X-Fal-Webhook-Request-Id")
    user_id = _required_header(headers, "X-Fal-Webhook-User-Id")
    timestamp_raw = _required_header(headers, "X-Fal-Webhook-Timestamp")
    signature_hex = _required_header(headers, "X-Fal-Webhook-Signature")

    try:
        timestamp_int = int(timestamp_raw)
    except ValueError as exc:
        raise FalWebhookVerificationError("invalid timestamp") from exc

    current_time = int(now if now is not None else time.time())
    if abs(current_time - timestamp_int) > 300:
        raise FalWebhookVerificationError("timestamp outside tolerance")

    digest = hashlib.sha256(body).hexdigest()
    message = "\n".join([request_id, user_id, timestamp_raw, digest]).encode("utf-8")

    try:
        signature = bytes.fromhex(signature_hex)
    except ValueError as exc:
        raise FalWebhookVerificationError("invalid signature format") from exc

    try:
        keys: Iterable[Mapping[str, object]] = fetch_keys()
    except Exception as exc:  # pragma: no cover - network failure propagated
        raise FalWebhookVerificationError("unable to fetch JWKS") from exc
    verified = False
    for key_info in keys:
        key_data = key_info.get("x")
        if not isinstance(key_data, str):
            continue
        try:
            public_key = _decode_base64url(key_data)
            verify_key = VerifyKey(public_key)
            verify_key.verify(message, signature)
            verified = True
            break
        except (BadSignatureError, ValueError, TypeError):
            continue

    if not verified:
        raise FalWebhookVerificationError("signature verification failed")
