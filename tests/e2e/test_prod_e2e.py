"""Real end-to-end checks against production ``https://app.voicelabs.now``.

Most of this file is gated on ``VOICELABS_API_KEY``. When it is absent the tests SKIP LOUDLY
rather than passing silently — a skipped e2e is not a green e2e.

Two checks deliberately need no credential and therefore run everywhere, including on forks:
the published OpenAPI document, and the unauthenticated 401 problem envelope. They are the
cheapest possible proof that the contract this SDK was written against is still the one
production serves.

The round-trip test generates real audio and spends the account's real allowance. It is kept to
one short utterance on purpose.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import httpx
import pytest

from voicelabs_py import (
    GENERATION_STATUS,
    AuthenticationError,
    IdempotencyError,
    InsufficientScopeError,
    VoiceLabs,
)
from voicelabs_py._client import USER_AGENT

PROD = "https://app.voicelabs.now"
CONTRACT_PATH = Path(__file__).resolve().parents[2] / "openapi.json"

API_KEY = os.environ.get("VOICELABS_API_KEY")
READONLY_API_KEY = os.environ.get("VOICELABS_READONLY_API_KEY")

requires_key = pytest.mark.skipif(
    not API_KEY,
    reason="LOUD SKIP: VOICELABS_API_KEY is not set — a skipped e2e is not a green e2e.",
)


@pytest.fixture
def client() -> VoiceLabs:
    with VoiceLabs(api_key=API_KEY, timeout=120.0) as sdk:
        yield sdk


# ── unauthenticated: these run everywhere, including on forks ────────────────


def test_the_published_openapi_document_is_reachable_and_describes_the_shipped_surface():
    response = httpx.get(f"{PROD}/openapi.json", headers={"user-agent": USER_AGENT}, timeout=30.0)

    assert response.status_code == 200
    document = response.json()
    assert document["openapi"] == "3.1.1"
    assert document["info"]["version"] == "1.0.0"
    assert len(document["paths"]) == 6
    assert set(document["paths"]) == {
        "/v1/voices",
        "/v1/captures",
        "/v1/generations/{generationId}",
        "/v1/speech",
        "/v1/transcriptions",
        "/v1/audio/{generationId}",
    }


def test_the_vendored_contract_matches_production():
    live = httpx.get(
        f"{PROD}/openapi.json", headers={"user-agent": USER_AGENT}, timeout=30.0
    ).json()
    vendored = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    assert live == vendored, (
        "The production OpenAPI document no longer matches the vendored snapshot. Re-vendor it:\n"
        "  curl -s https://app.voicelabs.now/openapi.json -o openapi.json\n"
        "then re-run tests/test_contract_coverage.py and decide what the SDK owes the change."
    )


def test_an_unauthenticated_call_returns_an_rfc_9457_problem_with_code_unauthorized():
    response = httpx.get(f"{PROD}/v1/voices", headers={"user-agent": USER_AGENT}, timeout=30.0)

    assert response.status_code == 401
    assert response.headers["content-type"].startswith("application/problem+json")
    problem = response.json()
    assert problem["code"] == "unauthorized"
    assert problem["type"] == "https://voicelabs.now/errors/unauthorized"
    assert problem["status"] == 401
    assert problem["detail"]


def test_the_sdk_turns_that_unauthenticated_problem_into_an_authentication_error():
    bogus = VoiceLabs(api_key="vl_sandbox_definitely_not_a_real_key", max_retries=0)
    with bogus, pytest.raises(AuthenticationError) as caught:
        bogus.list_voices()

    assert caught.value.code == "unauthorized"
    assert caught.value.status == 401
    assert caught.value.problem is not None


# ── authenticated ────────────────────────────────────────────────────────────


@requires_key
def test_listing_voices_with_a_real_key_returns_voices_and_draft_11_rate_limit_headers(client):
    voices = client.list_voices()

    assert voices.total == len(voices.data)
    for voice in voices.data:
        assert voice.id
        assert voice.name

    info = client.rate_limit
    assert info is not None, (
        "An authenticated 200 carried no RateLimit headers. Either limiting is off for this key "
        "or the API stopped emitting draft-11 fields — worth filing against the app repo."
    )
    assert isinstance(info.limit, int)
    assert isinstance(info.remaining, int)


@requires_key
def test_listing_captures_echoes_the_limit_and_offset_the_server_applied(client):
    page = client.list_captures(limit=2, offset=0)

    assert page.offset == 0
    assert page.limit <= 200
    assert len(page.data) <= page.limit


@requires_key
def test_a_full_speech_round_trip_creates_polls_downloads_and_yields_real_audio_bytes(
    client, tmp_path
):
    voices = client.list_voices()
    if not voices.data:
        pytest.skip("LOUD SKIP: the account has no voice profiles to generate with.")

    generation = client.generate_speech(
        text="VoiceLabs Python SDK end to end check.",
        voice_name=voices.data[0].name,
        poll_interval=2.0,
        timeout=240.0,
    )

    assert generation.status == GENERATION_STATUS.completed
    assert generation.has_audio is True
    assert generation.audio_url

    audio = client.download_audio(generation)
    assert len(audio.content) > 1024
    assert audio.content_type.startswith("audio/")

    written = client.write_audio(generation, tmp_path / "e2e.audio")
    assert written.stat().st_size == len(audio.content)


@requires_key
def test_replaying_an_idempotency_key_returns_the_original_response_without_new_work(client):
    key = f"voicelabs-sdk-e2e-{uuid.uuid4()}"
    text = "Idempotency check."

    voices = client.list_voices()
    voice_name = voices.data[0].name
    first = client.create_speech(text=text, voice_name=voice_name, idempotency_key=key)
    replay = client.create_speech(text=text, voice_name=voice_name, idempotency_key=key)

    assert first.id == replay.id

    with pytest.raises(IdempotencyError) as caught:
        client.create_speech(
            text="A different body entirely.", voice_name=voice_name, idempotency_key=key
        )

    assert caught.value.code == "idempotency_key_reused"


@pytest.mark.skipif(
    not READONLY_API_KEY,
    reason=(
        "LOUD SKIP: VOICELABS_READONLY_API_KEY is not set — the insufficient_scope path is "
        "unproven without a key that lacks voice:generate."
    ),
)
def test_a_key_without_the_generate_scope_is_refused_with_insufficient_scope():
    with VoiceLabs(api_key=READONLY_API_KEY, max_retries=0) as sdk:
        sdk.list_voices()  # voice:read works

        with pytest.raises(InsufficientScopeError) as caught:
            sdk.create_speech(text="This key may not generate.")

    assert caught.value.code == "insufficient_scope"
    assert caught.value.required_scope == "voice:generate"


@requires_key
def test_wire_probe_of_a_keyed_speech_request_and_its_replay(client):
    """TEMPORARY diagnostic for voicelabs#64 - prints status and framing headers, never a secret."""
    import datetime

    voice_name = client.list_voices().data[0].name
    key = f"voicelabs-wire-probe-{uuid.uuid4()}"
    for label in ("first", "replay"):
        stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        response = httpx.post(
            f"{PROD}/v1/speech",
            headers={"x-api-key": API_KEY, "idempotency-key": key, "user-agent": USER_AGENT},
            json={"text": "Wire probe.", "voice_name": voice_name},
            timeout=60.0,
        )
        framing = {
            name: response.headers.get(name)
            for name in ("content-type", "content-length", "content-encoding", "transfer-encoding")
        }
        print(
            f"WIREPROBE {label} at={stamp} status={response.status_code} {framing} "
            f"bytes={len(response.content)}"
        )
        assert response.status_code == 200
