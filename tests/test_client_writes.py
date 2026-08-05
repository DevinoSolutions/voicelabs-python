"""The two write operations: body shape, idempotency headers, and local validation."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from tests.helpers import API_KEY, BASE_URL
from voicelabs_py import IdempotencyError, VoiceLabs, VoiceLabsConfigError
from voicelabs_py.errors import error_from_response

SPEECH_HANDLE = {"id": "gen_1", "status": "generating", "voice": "Narrator"}


def client() -> VoiceLabs:
    return VoiceLabs(api_key=API_KEY, max_retries=0)


@respx.mock
def test_creating_speech_puts_the_documented_body_on_the_wire_and_returns_a_handle():
    route = respx.post(f"{BASE_URL}/v1/speech").mock(
        return_value=httpx.Response(200, json=SPEECH_HANDLE)
    )

    with client() as sdk:
        handle = sdk.create_speech(text="Hello from Python.", voice_name="Narrator", language="en")

    body = json.loads(route.calls.last.request.content)
    assert body == {"text": "Hello from Python.", "voice_name": "Narrator", "language": "en"}
    assert route.calls.last.request.headers["content-type"] == "application/json"
    assert handle.id == "gen_1"
    assert handle.voice == "Narrator"


@respx.mock
def test_optional_speech_fields_are_omitted_rather_than_sent_as_null():
    route = respx.post(f"{BASE_URL}/v1/speech").mock(
        return_value=httpx.Response(200, json=SPEECH_HANDLE)
    )

    with client() as sdk:
        sdk.create_speech(text="Just the text.")

    assert json.loads(route.calls.last.request.content) == {"text": "Just the text."}


def test_naming_a_voice_twice_is_refused_locally_so_the_choice_stays_the_callers():
    with client() as sdk, pytest.raises(VoiceLabsConfigError) as caught:
        sdk.create_speech(text="Hello.", voice_id="v_1", voice_name="Narrator")

    assert "not both" in str(caught.value)


def test_empty_speech_text_is_refused_before_the_request():
    with client() as sdk, pytest.raises(VoiceLabsConfigError):
        sdk.create_speech(text="")


@respx.mock
def test_an_idempotency_key_is_sent_as_a_header_and_only_on_writes():
    speech = respx.post(f"{BASE_URL}/v1/speech").mock(
        return_value=httpx.Response(200, json=SPEECH_HANDLE)
    )
    voices = respx.get(f"{BASE_URL}/v1/voices").mock(
        return_value=httpx.Response(200, json={"data": [], "total": 0})
    )

    with client() as sdk:
        sdk.create_speech(text="Hello.", idempotency_key="order-42")
        sdk.list_voices()

    assert speech.calls.last.request.headers["idempotency-key"] == "order-42"
    assert "idempotency-key" not in voices.calls.last.request.headers


@respx.mock
def test_a_write_without_an_idempotency_key_sends_no_such_header():
    route = respx.post(f"{BASE_URL}/v1/speech").mock(
        return_value=httpx.Response(200, json=SPEECH_HANDLE)
    )

    with client() as sdk:
        sdk.create_speech(text="Hello.")

    assert "idempotency-key" not in route.calls.last.request.headers


def test_an_over_long_idempotency_key_is_refused_before_the_request():
    with client() as sdk, pytest.raises(VoiceLabsConfigError) as caught:
        sdk.create_speech(text="Hello.", idempotency_key="k" * 256)

    assert "255" in str(caught.value)


@respx.mock
def test_replaying_an_idempotency_key_with_a_different_body_raises_idempotency_error():
    respx.post(f"{BASE_URL}/v1/speech").mock(
        return_value=httpx.Response(
            422,
            json={
                "type": "https://voicelabs.now/errors/idempotency_key_reused",
                "title": "Idempotency key reused",
                "status": 422,
                "detail": "This Idempotency-Key was already used with a different request body.",
                "code": "idempotency_key_reused",
            },
        )
    )

    with client() as sdk, pytest.raises(IdempotencyError) as caught:
        sdk.create_speech(text="Different text.", idempotency_key="order-42")

    assert caught.value.code == "idempotency_key_reused"


def test_an_in_progress_replay_and_a_reused_key_share_a_class_but_not_a_code():
    in_progress = error_from_response(
        409,
        {
            "type": "https://voicelabs.now/errors/idempotency_key_in_progress",
            "title": "In progress",
            "status": 409,
            "detail": "The first request with this key is still running.",
            "code": "idempotency_key_in_progress",
        },
    )
    reused = error_from_response(
        422,
        {
            "type": "https://voicelabs.now/errors/idempotency_key_reused",
            "title": "Reused",
            "status": 422,
            "detail": "Different body.",
            "code": "idempotency_key_reused",
        },
    )

    assert isinstance(in_progress, IdempotencyError)
    assert isinstance(reused, IdempotencyError)
    # Only `retryable` tells them apart, and it must: waiting fixes one and never the other.
    assert in_progress.retryable is True
    assert reused.retryable is False
