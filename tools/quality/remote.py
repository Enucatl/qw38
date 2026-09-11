"""Optional OpenRouter scorer for qwen/qwen3.8-27b. Default off. Never pytest network."""

from __future__ import annotations

import os
from typing import Any

from tools.quality.errors import QualityFrameworkError

OPENROUTER_MODEL = "qwen/qwen3.8-27b"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
API_KEY_ENV = "OPENROUTER_API_KEY"
LABEL = "external_scorer"


def in_pytest() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


def openrouter_status(
    *,
    enabled: bool = False,
    api_key: str | None = None,
    allow_network: bool = False,
) -> dict[str, Any]:
    """Inspect the remote scorer. Missing credentials are success, not failure."""
    key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
    payload: dict[str, Any] = {
        "enabled": bool(enabled),
        "default_off": True,
        "model": OPENROUTER_MODEL,
        "endpoint": OPENROUTER_ENDPOINT,
        "api_key_env": API_KEY_ENV,
        "label": LABEL,
        "authoritative": False,
        "network": False,
        "credentials_present": bool(key),
        "status": "disabled",
        "success": True,
    }
    if in_pytest() and allow_network:
        raise QualityFrameworkError("OpenRouter scorer cannot run in pytest")
    if not enabled:
        payload["status"] = "disabled"
        payload["reason"] = "default_off"
        return payload
    if not key:
        payload["status"] = "skipped_missing_credentials"
        payload["reason"] = "missing_OPENROUTER_API_KEY"
        payload["success"] = True
        return payload
    if not allow_network:
        payload["status"] = "not_invoked"
        payload["reason"] = "batch_does_not_call_openrouter"
        payload["success"] = True
        return payload
    raise QualityFrameworkError(
        "OpenRouter network collection is out of scope for OPT-083"
    )


def collect_openrouter(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """Never called automatically. OPT-083 does not collect remote fixtures."""
    return openrouter_status(enabled=True, allow_network=True)
