"""The async client: full parity with the sync one, over a native httpx.AsyncClient."""

from __future__ import annotations

import inspect

import httpx
import pytest
import respx

from tests.helpers import API_KEY, BASE_URL, capture, completed_generation, generation, voice
from voicelabs_py import (
    AsyncVoiceLabs,
    GenerationTimeoutError,
    NotFoundError,
    QuotaExhaustedError,
    VoiceLabs,
)

AUDIO_URL = f"{BASE_URL}/v1/audio/gen_1?sig=abc123"
AUDIO_BYTES = b"ID3fake mp3 payload" * 8

# The one deliberate difference: closing is `close()` on the sync client and `aclose()` on the
# async one, because an awaitable named `close` reads like a synchronous call that silently
# does nothing.
CLOSE_METHODS = {"close", "aclose"}


def public_names(cls: type) -> set[str]:
    return {name for name in dir(cls) if not name.startswith("_")}


def client(**kwargs) -> AsyncVoiceLabs:
    return AsyncVoiceLabs(api_key=API_KEY, max_retries=0, **kwargs)


def test_the_async_client_exposes_every_method_the_sync_client_does():
    missing = public_names(VoiceLabs) - public_names(AsyncVoiceLabs) - CLOSE_METHODS

    assert missing == set()
    assert "aclose" in public_names(AsyncVoiceLabs)


def test_every_shared_method_takes_exactly_the_same_arguments_on_both_clients():
    # Parameters only: the return annotations differ by design (Iterator vs AsyncIterator, and
    # everything else is wrapped in a coroutine), but a caller's call site must port unchanged.
    shared = (public_names(VoiceLabs) & public_names(AsyncVoiceLabs)) - CLOSE_METHODS

    for name in sorted(shared):
        sync_attr = getattr(VoiceLabs, name)
        async_attr = getattr(AsyncVoiceLabs, name)
        if not callable(sync_attr) or isinstance(sync_attr, property):
            continue
        sync_params = inspect.signature(sync_attr).parameters
        async_params = inspect.signature(async_attr).parameters
        assert list(sync_params) == list(async_params), name
        for parameter, expected in sync_params.items():
            assert async_params[parameter].default == expected.default, f"{name}.{parameter}"
            assert async_params[parameter].kind == expected.kind, f"{name}.{parameter}"


def test_the_async_client_uses_a_native_async_transport_and_not_a_thread_pool():
    sdk = client()

    assert isinstance(sdk._http, httpx.AsyncClient)
    assert inspect.iscoroutinefunction(sdk.list_voices)
    assert inspect.isasyncgenfunction(AsyncVoiceLabs.iter_captures)


@respx.mock
async def test_the_async_client_returns_the_same_models_for_the_same_wire_bytes():
    payload = {"data": [voice()], "total": 1}
    respx.get(f"{BASE_URL}/v1/voices").mock(return_value=httpx.Response(200, json=payload))

    with VoiceLabs(api_key=API_KEY, max_retries=0) as sync_sdk:
        from_sync = sync_sdk.list_voices()
    async with client() as async_sdk:
        from_async = await async_sdk.list_voices()

    assert from_sync == from_async


@respx.mock
async def test_the_async_client_sends_the_api_key_on_every_authenticated_call():
    route = respx.get(f"{BASE_URL}/v1/voices").mock(
        return_value=httpx.Response(200, json={"data": [], "total": 0})
    )

    async with client() as sdk:
        await sdk.list_voices()

    assert route.calls.last.request.headers["x-api-key"] == API_KEY


@respx.mock
async def test_the_async_iterator_walks_every_page_and_stops_without_an_extra_request():
    pages = [
        {"data": [capture(id="a"), capture(id="b")], "total": 3, "limit": 2, "offset": 0},
        {"data": [capture(id="c")], "total": 3, "limit": 2, "offset": 2},
    ]
    route = respx.get(f"{BASE_URL}/v1/captures").mock(
        side_effect=[httpx.Response(200, json=page) for page in pages]
    )

    async with client() as sdk:
        ids = [c.id async for c in sdk.iter_captures(page_size=2)]

    assert ids == ["a", "b", "c"]
    assert route.call_count == 2


@respx.mock
async def test_the_async_client_raises_the_same_typed_errors():
    respx.get(f"{BASE_URL}/v1/generations/nope").mock(
        return_value=httpx.Response(
            404,
            json={
                "type": "https://voicelabs.now/errors/not_found",
                "title": "Not Found",
                "status": 404,
                "detail": "No such generation.",
                "code": "not_found",
            },
        )
    )
    respx.get(f"{BASE_URL}/v1/voices").mock(
        return_value=httpx.Response(
            429,
            json={
                "type": "https://voicelabs.now/errors/quota_exhausted",
                "title": "Quota exhausted",
                "status": 429,
                "detail": "The allowance is spent.",
                "code": "quota_exhausted",
                "settings_url": "https://voicelabs.now/settings/billing",
            },
        )
    )

    async with client() as sdk:
        with pytest.raises(NotFoundError):
            await sdk.get_generation("nope")
        with pytest.raises(QuotaExhaustedError) as caught:
            await sdk.list_voices()

    assert caught.value.settings_url == "https://voicelabs.now/settings/billing"


@respx.mock
async def test_the_async_polling_helper_reaches_a_terminal_state():
    respx.get(f"{BASE_URL}/v1/generations/gen_1").mock(
        side_effect=[
            httpx.Response(200, json=generation()),
            httpx.Response(200, json=completed_generation()),
        ]
    )

    async with client() as sdk:
        done = await sdk.wait_for_generation("gen_1", poll_interval=0.0)

    assert done.status == "completed"


@respx.mock
async def test_the_async_polling_helper_times_out_without_returning_unfinished_work():
    respx.get(f"{BASE_URL}/v1/generations/gen_1").mock(
        return_value=httpx.Response(200, json=generation())
    )

    async with client() as sdk:
        with pytest.raises(GenerationTimeoutError) as caught:
            await sdk.wait_for_generation("gen_1", poll_interval=0.01, timeout=0.0)

    assert "has NOT been cancelled" in str(caught.value)


@respx.mock
async def test_the_async_download_sends_no_api_key_to_the_signed_capability_url():
    route = respx.get(AUDIO_URL).mock(
        return_value=httpx.Response(200, content=AUDIO_BYTES, headers={"content-type": "audio/wav"})
    )

    async with client() as sdk:
        audio = await sdk.download_audio(AUDIO_URL)

    assert "x-api-key" not in route.calls.last.request.headers
    assert "authorization" not in route.calls.last.request.headers
    assert audio.content == AUDIO_BYTES


@respx.mock
async def test_the_async_write_audio_writes_the_bytes_and_returns_the_path(tmp_path):
    respx.get(AUDIO_URL).mock(
        return_value=httpx.Response(200, content=AUDIO_BYTES, headers={"content-type": "audio/wav"})
    )
    destination = tmp_path / "out" / "speech.wav"

    async with client() as sdk:
        written = await sdk.write_audio(AUDIO_URL, destination)

    assert written == destination
    assert destination.read_bytes() == AUDIO_BYTES


async def test_the_async_context_manager_closes_the_transport():
    sdk = client()

    async with sdk:
        assert sdk._http.is_closed is False

    assert sdk._http.is_closed is True
