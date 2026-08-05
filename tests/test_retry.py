"""The retry policy — narrow on purpose, and never retrying what cannot succeed."""

from __future__ import annotations

import httpx
import pytest
import respx

from tests.helpers import API_KEY, BASE_URL, problem, voice
from voicelabs_py import (
    AsyncVoiceLabs,
    AuthenticationError,
    IdempotencyError,
    QuotaExhaustedError,
    RateLimitError,
    ServerError,
    VoiceLabs,
    VoiceLabsConnectionError,
    _retry,
)

SPEECH_HANDLE = {"id": "gen_1", "status": "generating", "voice": "Narrator"}
VOICES = {"data": [voice()], "total": 1}


@pytest.fixture(autouse=True)
def instant_and_deterministic(monkeypatch):
    """Make retries instant and their jitter predictable, so the tests assert behaviour."""
    slept: list[float] = []

    def record(seconds):
        slept.append(seconds)

    async def record_async(seconds):
        slept.append(seconds)

    monkeypatch.setattr("time.sleep", record)
    monkeypatch.setattr("asyncio.sleep", record_async)
    monkeypatch.setattr(_retry, "_full_jitter", lambda ceiling: ceiling)
    return slept


def client(**kwargs) -> VoiceLabs:
    return VoiceLabs(api_key=API_KEY, **kwargs)


@respx.mock
def test_a_rate_limited_request_is_retried_after_the_server_supplied_delay(
    instant_and_deterministic,
):
    route = respx.get(f"{BASE_URL}/v1/voices").mock(
        side_effect=[
            httpx.Response(
                429,
                json=problem(code="rate_limit_exceeded", status=429),
                headers={"Retry-After": "7", "RateLimit": '"default";r=0;t=41'},
            ),
            httpx.Response(200, json=VOICES),
        ]
    )

    with client(max_retries=2) as sdk:
        voices = sdk.list_voices()

    assert voices.total == 1
    assert route.call_count == 2
    # Retry-After wins over the draft-11 reset: it is the API's own instruction.
    assert instant_and_deterministic == [7.0]


@respx.mock
def test_a_rate_limit_without_retry_after_falls_back_to_the_draft_11_reset(
    instant_and_deterministic,
):
    respx.get(f"{BASE_URL}/v1/voices").mock(
        side_effect=[
            httpx.Response(
                429,
                json=problem(code="rate_limit_exceeded", status=429),
                headers={"RateLimit": '"default";r=0;t=13'},
            ),
            httpx.Response(200, json=VOICES),
        ]
    )

    with client(max_retries=2) as sdk:
        sdk.list_voices()

    assert instant_and_deterministic == [13.0]


@respx.mock
def test_a_rate_limit_with_no_server_hint_at_all_backs_off_exponentially(
    instant_and_deterministic,
):
    respx.get(f"{BASE_URL}/v1/voices").mock(
        side_effect=[
            httpx.Response(429, json=problem(code="rate_limit_exceeded", status=429)),
            httpx.Response(429, json=problem(code="rate_limit_exceeded", status=429)),
            httpx.Response(200, json=VOICES),
        ]
    )

    with client(max_retries=2) as sdk:
        sdk.list_voices()

    assert instant_and_deterministic == [0.5, 1.0]


@respx.mock
def test_a_quota_exhausted_request_is_never_retried_because_retrying_is_futile():
    route = respx.get(f"{BASE_URL}/v1/voices").mock(
        return_value=httpx.Response(
            429,
            json=problem(
                code="quota_exhausted",
                status=429,
                settings_url="https://voicelabs.now/settings/billing",
            ),
            headers={"Retry-After": "1"},
        )
    )

    with client(max_retries=5) as sdk, pytest.raises(QuotaExhaustedError):
        sdk.list_voices()

    # Exactly one. A backoff loop here would burn the caller's rate budget until the billing
    # period rolled over, and never succeed.
    assert route.call_count == 1


@respx.mock
def test_server_errors_are_retried_with_exponential_backoff(instant_and_deterministic):
    route = respx.get(f"{BASE_URL}/v1/voices").mock(
        side_effect=[
            httpx.Response(503, json=problem(code="service_unavailable", status=503)),
            httpx.Response(502, json=problem(code="upstream_error", status=502)),
            httpx.Response(200, json=VOICES),
        ]
    )

    with client(max_retries=2) as sdk:
        sdk.list_voices()

    assert route.call_count == 3
    assert instant_and_deterministic == [0.5, 1.0]


@respx.mock
def test_a_client_error_that_is_not_a_rate_limit_is_never_retried():
    route = respx.get(f"{BASE_URL}/v1/voices").mock(
        return_value=httpx.Response(401, json=problem(code="unauthorized", status=401))
    )

    with client(max_retries=3) as sdk, pytest.raises(AuthenticationError):
        sdk.list_voices()

    assert route.call_count == 1


@respx.mock
def test_a_read_is_retried_on_a_connection_error(instant_and_deterministic):
    route = respx.get(f"{BASE_URL}/v1/voices").mock(
        side_effect=[httpx.ConnectError("no route to host"), httpx.Response(200, json=VOICES)]
    )

    with client(max_retries=2) as sdk:
        sdk.list_voices()

    assert route.call_count == 2


@respx.mock
def test_a_write_without_an_idempotency_key_is_not_retried_on_a_connection_error():
    route = respx.post(f"{BASE_URL}/v1/speech").mock(side_effect=httpx.ConnectError("reset"))

    with client(max_retries=3) as sdk, pytest.raises(VoiceLabsConnectionError) as caught:
        sdk.create_speech(text="Hello.")

    # Exactly one attempt: the write may have succeeded server-side, and a blind replay would
    # generate — and charge for — the audio twice.
    assert route.call_count == 1
    assert isinstance(caught.value.__cause__, httpx.ConnectError)


@respx.mock
def test_a_write_with_an_idempotency_key_is_retried_on_a_connection_error():
    route = respx.post(f"{BASE_URL}/v1/speech").mock(side_effect=httpx.ConnectError("reset"))

    with client(max_retries=2) as sdk, pytest.raises(VoiceLabsConnectionError):
        sdk.create_speech(text="Hello.", idempotency_key="order-42")

    assert route.call_count == 3  # max_retries + 1


@respx.mock
def test_a_write_with_an_idempotency_key_recovers_when_the_replay_succeeds():
    route = respx.post(f"{BASE_URL}/v1/speech").mock(
        side_effect=[httpx.ConnectError("reset"), httpx.Response(200, json=SPEECH_HANDLE)]
    )

    with client(max_retries=2) as sdk:
        handle = sdk.create_speech(text="Hello.", idempotency_key="order-42")

    assert handle.id == "gen_1"
    assert route.call_count == 2


@respx.mock
def test_retries_stop_at_max_retries_and_raise_the_last_error():
    route = respx.get(f"{BASE_URL}/v1/voices").mock(
        return_value=httpx.Response(
            503, json=problem(code="service_unavailable", status=503, detail="Still down.")
        )
    )

    with client(max_retries=2) as sdk, pytest.raises(ServerError) as caught:
        sdk.list_voices()

    assert route.call_count == 3
    assert caught.value.detail == "Still down."


@respx.mock
def test_max_retries_zero_disables_retrying_entirely():
    route = respx.get(f"{BASE_URL}/v1/voices").mock(
        return_value=httpx.Response(503, json=problem(code="service_unavailable", status=503))
    )

    with client(max_retries=0) as sdk, pytest.raises(ServerError):
        sdk.list_voices()

    assert route.call_count == 1


@respx.mock
def test_a_wait_that_would_outlast_the_client_timeout_raises_instead_of_sleeping_past_it():
    route = respx.get(f"{BASE_URL}/v1/voices").mock(
        return_value=httpx.Response(
            429,
            json=problem(code="rate_limit_exceeded", status=429),
            headers={"Retry-After": "600"},
        )
    )

    # The server asks for ten minutes; the caller budgeted five seconds. Sleeping anyway would
    # make "this call takes at most 5s" a quiet lie.
    with client(max_retries=3, timeout=5.0) as sdk, pytest.raises(RateLimitError) as caught:
        sdk.list_voices()

    assert route.call_count == 1
    assert caught.value.retry_after == 600.0


@respx.mock
def test_an_idempotency_key_in_progress_is_left_for_the_caller_to_time():
    route = respx.post(f"{BASE_URL}/v1/speech").mock(
        return_value=httpx.Response(
            409, json=problem(code="idempotency_key_in_progress", status=409)
        )
    )

    with client(max_retries=3) as sdk, pytest.raises(IdempotencyError) as caught:
        sdk.create_speech(text="Hello.", idempotency_key="order-42")

    assert route.call_count == 1
    # Advisory-only: the error says "retryable", but only the caller knows how long their own
    # first request takes, and an immediate replay just collects another 409.
    assert caught.value.retryable is True


@respx.mock
async def test_the_async_client_retries_on_exactly_the_same_terms(instant_and_deterministic):
    route = respx.get(f"{BASE_URL}/v1/voices").mock(
        side_effect=[
            httpx.Response(
                429,
                json=problem(code="rate_limit_exceeded", status=429),
                headers={"Retry-After": "3"},
            ),
            httpx.Response(200, json=VOICES),
        ]
    )
    quota = respx.get(f"{BASE_URL}/v1/captures").mock(
        return_value=httpx.Response(429, json=problem(code="quota_exhausted", status=429))
    )

    async with AsyncVoiceLabs(api_key=API_KEY, max_retries=3) as sdk:
        await sdk.list_voices()
        with pytest.raises(QuotaExhaustedError):
            await sdk.list_captures()

    assert route.call_count == 2
    assert quota.call_count == 1
    assert instant_and_deterministic == [3.0]
