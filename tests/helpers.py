"""Dependency-free fixture builders for the VoiceLabs SDK tests.

Every builder returns plain dicts shaped exactly like the wire payloads in
``openapi.json``, so a fixture that drifts from the contract is a test failure rather than a
false green.
"""

from __future__ import annotations

from typing import Any

API_KEY = "vl_sandbox_test_key"
BASE_URL = "https://app.voicelabs.now"


def problem(
    *,
    code: str,
    status: int,
    detail: str = "Something the API wants to explain.",
    title: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """An RFC 9457 problem document as the API emits it."""
    document: dict[str, Any] = {
        "type": f"https://voicelabs.now/errors/{code}",
        "title": title or code.replace("_", " ").title(),
        "status": status,
        "detail": detail,
        "code": code,
    }
    document.update(extra)
    return document


def voice(**overrides: Any) -> dict[str, Any]:
    """A ``Voice`` resource."""
    payload = {
        "id": "voice_1",
        "name": "Narrator",
        "description": None,
        "language": "en",
        "voice_type": "cloned",
        "default_engine": None,
        "has_personality": False,
        "generation_count": 3,
        "sample_count": 2,
        "created_at": "2026-01-01T00:00:00.000Z",
        "updated_at": "2026-01-02T00:00:00.000Z",
    }
    payload.update(overrides)
    return payload


def capture(**overrides: Any) -> dict[str, Any]:
    """A ``Capture`` resource."""
    payload = {
        "id": "cap_1",
        "source": "desktop",
        "language": "en",
        "duration_ms": 1500,
        "transcript_raw": "hello there",
        "transcript_refined": "Hello there.",
        "created_at": "2026-01-01T00:00:00.000Z",
    }
    payload.update(overrides)
    return payload


def generation(**overrides: Any) -> dict[str, Any]:
    """A ``Generation`` resource, mid-flight by default."""
    payload = {
        "id": "gen_1",
        "status": "generating",
        "profile": "Narrator",
        "has_audio": False,
        "audio_url": None,
        "error": None,
    }
    payload.update(overrides)
    return payload


def completed_generation(**overrides: Any) -> dict[str, Any]:
    """A ``Generation`` that finished, carrying a signed capability URL."""
    return generation(
        status="completed",
        has_audio=True,
        audio_url=f"{BASE_URL}/v1/audio/gen_1?sig=abc123",
        **overrides,
    )


def draft_11_headers(
    *,
    quota: int = 600,
    window: int = 3600,
    remaining: int = 599,
    reset: int = 2718,
) -> dict[str, str]:
    """The IETF draft-11 rate-limit header pair."""
    return {
        "RateLimit-Policy": f'"default";q={quota};w={window}',
        "RateLimit": f'"default";r={remaining};t={reset}',
    }
