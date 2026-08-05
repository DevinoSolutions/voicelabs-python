"""Transcription: base64 in JSON, never multipart, and size-checked before the wire."""

from __future__ import annotations

import base64
import io
import json

import httpx
import pytest
import respx

from tests.helpers import API_KEY, BASE_URL
from voicelabs_py import MAX_AUDIO_BASE64_CHARS, VoiceLabs, VoiceLabsConfigError

AUDIO = b"RIFF____WAVEfmt fake audio bytes"

TRANSCRIPT = {"id": "cap_9", "text": "Hello there.", "language": "en", "duration_ms": 1200}


def client() -> VoiceLabs:
    return VoiceLabs(api_key=API_KEY, max_retries=0)


def sent_body(route) -> dict:
    return json.loads(route.calls.last.request.content)


@respx.mock
def test_transcription_accepts_raw_bytes_and_base64_encodes_them_for_the_wire():
    route = respx.post(f"{BASE_URL}/v1/transcriptions").mock(
        return_value=httpx.Response(200, json=TRANSCRIPT)
    )

    with client() as sdk:
        transcription = sdk.create_transcription(audio=AUDIO, language="en")

    body = sent_body(route)
    assert base64.b64decode(body["audio"]) == AUDIO
    assert body["language"] == "en"
    assert transcription.text == "Hello there."
    assert transcription.duration_ms == 1200


@respx.mock
def test_the_transcription_request_is_json_and_never_multipart():
    route = respx.post(f"{BASE_URL}/v1/transcriptions").mock(
        return_value=httpx.Response(200, json=TRANSCRIPT)
    )

    with client() as sdk:
        sdk.create_transcription(audio=AUDIO)

    request = route.calls.last.request
    assert request.headers["content-type"] == "application/json"
    assert "multipart" not in request.headers["content-type"]
    assert set(sent_body(route)) == {"audio"}


@respx.mock
def test_transcription_accepts_a_file_path_and_a_file_object(tmp_path):
    route = respx.post(f"{BASE_URL}/v1/transcriptions").mock(
        return_value=httpx.Response(200, json=TRANSCRIPT)
    )
    path = tmp_path / "clip.wav"
    path.write_bytes(AUDIO)

    with client() as sdk:
        sdk.create_transcription(audio=path)
        from_path = sent_body(route)["audio"]

        sdk.create_transcription(audio=str(path))
        from_str = sent_body(route)["audio"]

        with path.open("rb") as handle:
            sdk.create_transcription(audio=handle)
        from_handle = sent_body(route)["audio"]

        sdk.create_transcription(audio=io.BytesIO(AUDIO))
        from_buffer = sent_body(route)["audio"]

    expected = base64.b64encode(AUDIO).decode()
    assert from_path == from_str == from_handle == from_buffer == expected


@respx.mock
def test_already_encoded_base64_is_sent_through_untouched():
    route = respx.post(f"{BASE_URL}/v1/transcriptions").mock(
        return_value=httpx.Response(200, json=TRANSCRIPT)
    )
    encoded = base64.b64encode(AUDIO).decode()

    with client() as sdk:
        sdk.create_transcription(audio_base64=encoded)

    assert sent_body(route)["audio"] == encoded


def test_audio_larger_than_the_documented_cap_is_refused_before_the_request_is_made():
    # respx is not installed here on purpose: if the SDK made a request this test would blow up
    # with a connection error instead of the config error we are asserting.
    oversized = b"\0" * (MAX_AUDIO_BASE64_CHARS)  # base64 of this is ~4/3 the cap

    with client() as sdk, pytest.raises(VoiceLabsConfigError) as caught:
        sdk.create_transcription(audio=oversized)

    message = str(caught.value)
    assert str(MAX_AUDIO_BASE64_CHARS) in message
    assert "base64 characters" in message


def test_passing_neither_or_both_audio_arguments_is_refused_with_a_clear_message():
    with client() as sdk:
        with pytest.raises(VoiceLabsConfigError) as neither:
            sdk.create_transcription()
        with pytest.raises(VoiceLabsConfigError) as both:
            sdk.create_transcription(audio=AUDIO, audio_base64="AAAA")

    assert "exactly one of" in str(neither.value)
    assert "exactly one of" in str(both.value)


def test_malformed_base64_is_refused_locally_rather_than_by_the_server():
    with client() as sdk, pytest.raises(VoiceLabsConfigError) as caught:
        sdk.create_transcription(audio_base64="not base64 at all!!")

    assert "not valid base64" in str(caught.value)


def test_a_text_mode_file_object_is_refused_with_advice_to_open_it_in_binary(tmp_path):
    path = tmp_path / "clip.wav"
    path.write_text("not bytes")

    with client() as sdk, path.open("r") as handle, pytest.raises(VoiceLabsConfigError) as caught:
        sdk.create_transcription(audio=handle)

    assert "binary mode" in str(caught.value)


def test_an_unreadable_audio_path_is_refused_before_the_request(tmp_path):
    with client() as sdk, pytest.raises(VoiceLabsConfigError) as caught:
        sdk.create_transcription(audio=tmp_path / "does-not-exist.wav")

    assert "Could not read the audio file" in str(caught.value)
