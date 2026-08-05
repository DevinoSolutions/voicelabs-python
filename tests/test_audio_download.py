"""Polling to a terminal state, and downloading audio without leaking the API key."""

from __future__ import annotations

import httpx
import pytest
import respx

from tests.helpers import API_KEY, BASE_URL, completed_generation, generation, problem
from voicelabs_py import (
    AuthenticationError,
    Generation,
    GenerationFailedError,
    GenerationTimeoutError,
    VoiceLabs,
    VoiceLabsConfigError,
)

AUDIO_URL = f"{BASE_URL}/v1/audio/gen_1?sig=abc123"
AUDIO_BYTES = b"ID3fake mp3 payload" * 8


def client() -> VoiceLabs:
    return VoiceLabs(api_key=API_KEY, max_retries=0)


# ── polling ──────────────────────────────────────────────────────────────────


@respx.mock
def test_wait_for_generation_polls_to_a_terminal_state_and_never_forever():
    route = respx.get(f"{BASE_URL}/v1/generations/gen_1").mock(
        side_effect=[
            httpx.Response(200, json=generation()),
            httpx.Response(200, json=generation()),
            httpx.Response(200, json=completed_generation()),
        ]
    )
    seen = []

    with client() as sdk:
        done = sdk.wait_for_generation(
            "gen_1", poll_interval=0.0, timeout=30.0, on_poll=seen.append
        )

    assert done.status == "completed"
    assert done.audio_url == AUDIO_URL
    assert route.call_count == 3
    # on_poll fires for the intermediate results only — the terminal one is the return value.
    assert len(seen) == 2


@respx.mock
def test_wait_for_generation_accepts_a_generation_object_a_handle_or_a_bare_id():
    respx.get(f"{BASE_URL}/v1/generations/gen_1").mock(
        return_value=httpx.Response(200, json=completed_generation())
    )

    with client() as sdk:
        from_id = sdk.wait_for_generation("gen_1", poll_interval=0.0)
        from_object = sdk.wait_for_generation(from_id, poll_interval=0.0)

    assert from_id == from_object


@respx.mock
def test_wait_for_generation_raises_a_timeout_that_says_the_work_was_not_cancelled():
    respx.get(f"{BASE_URL}/v1/generations/gen_1").mock(
        return_value=httpx.Response(200, json=generation())
    )

    with client() as sdk, pytest.raises(GenerationTimeoutError) as caught:
        sdk.wait_for_generation("gen_1", poll_interval=0.01, timeout=0.0)

    message = str(caught.value)
    assert "has NOT been cancelled" in message
    assert "re-poll it with get_generation()" in message
    # The last-seen generation rides along so the caller can keep the id without re-parsing text.
    assert caught.value.generation.id == "gen_1"
    assert caught.value.generation.status == "generating"


@respx.mock
def test_a_failed_generation_raises_with_the_servers_error_message_attached():
    respx.get(f"{BASE_URL}/v1/generations/gen_1").mock(
        return_value=httpx.Response(
            200, json=generation(status="failed", error="the engine ran out of voice")
        )
    )

    with client() as sdk, pytest.raises(GenerationFailedError) as caught:
        sdk.wait_for_generation("gen_1", poll_interval=0.0)

    assert caught.value.reason == "the engine ran out of voice"
    assert caught.value.generation_id == "gen_1"
    assert "the engine ran out of voice" in str(caught.value)


@respx.mock
def test_generate_speech_creates_then_waits_and_returns_the_finished_generation():
    create = respx.post(f"{BASE_URL}/v1/speech").mock(
        return_value=httpx.Response(
            200, json={"id": "gen_1", "status": "generating", "voice": "Narrator"}
        )
    )
    poll = respx.get(f"{BASE_URL}/v1/generations/gen_1").mock(
        return_value=httpx.Response(200, json=completed_generation())
    )

    with client() as sdk:
        done = sdk.generate_speech(text="Hello.", voice_name="Narrator", poll_interval=0.0)

    assert create.call_count == 1
    assert poll.call_count == 1
    assert done.status == "completed"


# ── download ─────────────────────────────────────────────────────────────────


@respx.mock
def test_download_audio_sends_no_api_key_to_the_signed_capability_url():
    route = respx.get(AUDIO_URL).mock(
        return_value=httpx.Response(
            200, content=AUDIO_BYTES, headers={"content-type": "audio/mpeg"}
        )
    )

    with client() as sdk:
        audio = sdk.download_audio(_completed())

    request = route.calls.last.request
    # THE non-negotiable: the signature in the URL is the whole authorization, so no credential
    # of ours may travel to whatever host that URL points at.
    assert "x-api-key" not in request.headers
    assert "authorization" not in request.headers
    assert audio.content == AUDIO_BYTES
    assert audio.content_type == "audio/mpeg"
    assert len(audio) == len(AUDIO_BYTES)


@respx.mock
def test_download_audio_accepts_a_bare_url_string_and_still_sends_no_credential():
    route = respx.get(AUDIO_URL).mock(
        return_value=httpx.Response(200, content=AUDIO_BYTES, headers={"content-type": "audio/wav"})
    )

    with client() as sdk:
        audio = sdk.download_audio(AUDIO_URL)

    assert "x-api-key" not in route.calls.last.request.headers
    assert audio.content_type == "audio/wav"


@respx.mock
def test_download_audio_sends_no_credential_even_to_a_url_on_another_host():
    foreign = "https://cdn.example.test/audio/gen_1?sig=abc123"
    route = respx.get(foreign).mock(
        return_value=httpx.Response(200, content=AUDIO_BYTES, headers={"content-type": "audio/wav"})
    )

    with client() as sdk:
        sdk.download_audio(foreign)

    assert "x-api-key" not in route.calls.last.request.headers


def test_download_audio_refuses_a_generation_whose_audio_url_is_still_null():
    running = _running()

    with client() as sdk, pytest.raises(VoiceLabsConfigError) as caught:
        sdk.download_audio(running)

    assert "wait_for_generation" in str(caught.value)


@respx.mock
def test_a_stale_signature_surfaces_as_a_typed_authentication_error():
    respx.get(AUDIO_URL).mock(
        return_value=httpx.Response(
            401,
            json=problem(code="unauthorized", status=401, detail="The signature has expired."),
        )
    )

    with client() as sdk, pytest.raises(AuthenticationError) as caught:
        sdk.download_audio(AUDIO_URL)

    assert caught.value.detail == "The signature has expired."


@respx.mock
def test_write_audio_writes_the_bytes_and_returns_the_path(tmp_path):
    respx.get(AUDIO_URL).mock(
        return_value=httpx.Response(
            200, content=AUDIO_BYTES, headers={"content-type": "audio/mpeg"}
        )
    )
    destination = tmp_path / "nested" / "speech.mp3"

    with client() as sdk:
        written = sdk.write_audio(_completed(), destination)

    assert written == destination
    assert destination.read_bytes() == AUDIO_BYTES


def _completed() -> Generation:
    return Generation.from_payload(completed_generation())


def _running() -> Generation:
    return Generation.from_payload(generation())
