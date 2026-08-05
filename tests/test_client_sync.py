"""The synchronous client: authentication, reads, pagination, and model tolerance."""

from __future__ import annotations

import httpx
import pytest
import respx

from tests.helpers import API_KEY, BASE_URL, capture, problem, voice
from voicelabs_py import (
    GENERATION_STATUS,
    NotFoundError,
    VoiceLabs,
    VoiceLabsConfigError,
)
from voicelabs_py.models import Voice


def client(**kwargs) -> VoiceLabs:
    return VoiceLabs(api_key=API_KEY, max_retries=0, **kwargs)


@respx.mock
def test_the_api_key_travels_in_the_x_api_key_header_on_every_authenticated_call():
    route = respx.get(f"{BASE_URL}/v1/voices").mock(
        return_value=httpx.Response(200, json={"data": [voice()], "total": 1})
    )

    with client() as sdk:
        sdk.list_voices()

    request = route.calls.last.request
    assert request.headers["x-api-key"] == API_KEY
    assert request.headers["user-agent"].startswith("voicelabs-py/")
    assert "authorization" not in request.headers


@respx.mock
def test_the_bearer_auth_style_sends_the_same_key_as_an_authorization_header_instead():
    route = respx.get(f"{BASE_URL}/v1/voices").mock(
        return_value=httpx.Response(200, json={"data": [], "total": 0})
    )

    with client(auth_style="bearer") as sdk:
        sdk.list_voices()

    request = route.calls.last.request
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    assert "x-api-key" not in request.headers


@respx.mock
def test_the_client_reads_the_api_key_from_the_environment_when_none_is_passed(monkeypatch):
    monkeypatch.setenv("VOICELABS_API_KEY", "vl_sandbox_from_env")
    route = respx.get(f"{BASE_URL}/v1/voices").mock(
        return_value=httpx.Response(200, json={"data": [], "total": 0})
    )

    with VoiceLabs() as sdk:
        sdk.list_voices()

    assert route.calls.last.request.headers["x-api-key"] == "vl_sandbox_from_env"


def test_a_client_with_no_key_anywhere_fails_loudly_before_any_request(monkeypatch):
    monkeypatch.delenv("VOICELABS_API_KEY", raising=False)

    with pytest.raises(VoiceLabsConfigError) as caught:
        VoiceLabs()

    assert "voicelabs.now/connections" in str(caught.value)


@respx.mock
def test_listing_voices_returns_typed_voices_and_the_servers_total():
    respx.get(f"{BASE_URL}/v1/voices").mock(
        return_value=httpx.Response(
            200, json={"data": [voice(), voice(id="voice_2", name="Reader")], "total": 2}
        )
    )

    with client() as sdk:
        voices = sdk.list_voices()

    assert voices.total == 2
    assert [v.name for v in voices.data] == ["Narrator", "Reader"]
    assert isinstance(voices.data[0], Voice)


@respx.mock
def test_the_base_url_is_right_stripped_so_a_trailing_slash_does_not_double_up():
    route = respx.get(f"{BASE_URL}/v1/voices").mock(
        return_value=httpx.Response(200, json={"data": [], "total": 0})
    )

    with VoiceLabs(api_key=API_KEY, base_url=f"{BASE_URL}///") as sdk:
        sdk.list_voices()

    assert str(route.calls.last.request.url) == f"{BASE_URL}/v1/voices"


@respx.mock
def test_listing_captures_clamps_the_page_size_to_the_documented_maximum_of_200():
    route = respx.get(f"{BASE_URL}/v1/captures").mock(
        return_value=httpx.Response(
            200, json={"data": [capture()], "total": 1, "limit": 200, "offset": 0}
        )
    )

    with client() as sdk:
        sdk.list_captures(limit=500)
        sdk.list_captures(limit=0)

    assert route.calls[0].request.url.params["limit"] == "200"
    assert route.calls[1].request.url.params["limit"] == "1"


@respx.mock
def test_listing_captures_omits_unset_query_parameters_entirely():
    route = respx.get(f"{BASE_URL}/v1/captures").mock(
        return_value=httpx.Response(200, json={"data": [], "total": 0, "limit": 50, "offset": 0})
    )

    with client() as sdk:
        sdk.list_captures()

    assert route.calls.last.request.url.params == httpx.QueryParams()


@respx.mock
def test_iter_captures_walks_every_page_and_stops_without_a_extra_request():
    pages = [
        {
            "data": [capture(id=f"cap_{i}") for i in range(1, 3)],
            "total": 5,
            "limit": 2,
            "offset": 0,
        },
        {
            "data": [capture(id=f"cap_{i}") for i in range(3, 5)],
            "total": 5,
            "limit": 2,
            "offset": 2,
        },
        {"data": [capture(id="cap_5")], "total": 5, "limit": 2, "offset": 4},
    ]
    route = respx.get(f"{BASE_URL}/v1/captures").mock(
        side_effect=[httpx.Response(200, json=page) for page in pages]
    )

    with client() as sdk:
        ids = [c.id for c in sdk.iter_captures(page_size=2)]

    assert ids == ["cap_1", "cap_2", "cap_3", "cap_4", "cap_5"]
    # Exactly three requests: the third page exhausts `total`, so there is no wasted fourth call.
    assert route.call_count == 3
    assert [call.request.url.params["offset"] for call in route.calls] == ["0", "2", "4"]


@respx.mock
def test_iter_captures_pages_by_the_offset_the_server_echoed_not_the_one_requested():
    # The server clamped the page to 2 rows even though 200 were asked for. A pager that trusted
    # its own number would jump to offset=200 and silently skip rows 3 and 4.
    pages = [
        {"data": [capture(id="a"), capture(id="b")], "total": 4, "limit": 2, "offset": 0},
        {"data": [capture(id="c"), capture(id="d")], "total": 4, "limit": 2, "offset": 2},
    ]
    route = respx.get(f"{BASE_URL}/v1/captures").mock(
        side_effect=[httpx.Response(200, json=page) for page in pages]
    )

    with client() as sdk:
        ids = [c.id for c in sdk.iter_captures()]

    assert ids == ["a", "b", "c", "d"]
    assert [call.request.url.params["offset"] for call in route.calls] == ["0", "2"]


@respx.mock
def test_iter_captures_stops_on_an_empty_page_even_when_the_total_disagrees():
    route = respx.get(f"{BASE_URL}/v1/captures").mock(
        return_value=httpx.Response(200, json={"data": [], "total": 99, "limit": 200, "offset": 0})
    )

    with client() as sdk:
        assert list(sdk.iter_captures()) == []

    assert route.call_count == 1


@respx.mock
def test_reading_a_generation_returns_its_status_profile_and_signed_audio_url():
    respx.get(f"{BASE_URL}/v1/generations/gen_1").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "gen_1",
                "status": "completed",
                "profile": "Narrator",
                "has_audio": True,
                "audio_url": f"{BASE_URL}/v1/audio/gen_1?sig=abc",
                "error": None,
            },
        )
    )

    with client() as sdk:
        generation = sdk.get_generation("gen_1")

    assert generation.status == GENERATION_STATUS.completed
    assert generation.has_audio is True
    assert generation.audio_url is not None


@respx.mock
def test_a_generation_id_with_a_slash_cannot_reshape_the_request_path():
    route = respx.get(url__regex=rf"{BASE_URL}/v1/generations/.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "x",
                "status": "generating",
                "profile": "p",
                "has_audio": False,
                "audio_url": None,
                "error": None,
            },
        )
    )

    with client() as sdk:
        sdk.get_generation("../voices")

    url = str(route.calls.last.request.url)
    assert url == f"{BASE_URL}/v1/generations/..%2Fvoices"


@respx.mock
def test_a_missing_generation_raises_not_found_with_the_servers_detail():
    respx.get(f"{BASE_URL}/v1/generations/nope").mock(
        return_value=httpx.Response(
            404,
            json=problem(code="not_found", status=404, detail="No such generation."),
            headers={"content-type": "application/problem+json"},
        )
    )

    with client() as sdk, pytest.raises(NotFoundError) as caught:
        sdk.get_generation("nope")

    assert caught.value.code == "not_found"
    assert caught.value.status == 404
    assert caught.value.detail == "No such generation."


@respx.mock
def test_the_last_seen_rate_limit_state_is_available_on_the_client():
    respx.get(f"{BASE_URL}/v1/voices").mock(
        return_value=httpx.Response(
            200,
            json={"data": [], "total": 0},
            headers={
                "RateLimit-Policy": '"default";q=600;w=3600',
                "RateLimit": '"default";r=598;t=2700',
            },
        )
    )

    seen = []
    with client(on_rate_limit=seen.append) as sdk:
        assert sdk.rate_limit is None
        sdk.list_voices()

        assert sdk.rate_limit is not None
        assert sdk.rate_limit.limit == 600
        assert sdk.rate_limit.remaining == 598
    assert len(seen) == 1


@respx.mock
def test_a_response_without_rate_limit_headers_leaves_the_state_unknown_rather_than_zero():
    respx.get(f"{BASE_URL}/v1/voices").mock(
        return_value=httpx.Response(200, json={"data": [], "total": 0})
    )

    with client() as sdk:
        sdk.list_voices()
        assert sdk.rate_limit is None


@respx.mock
def test_models_tolerate_unknown_fields_so_an_additive_server_change_is_not_an_outage():
    respx.get(f"{BASE_URL}/v1/voices").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [voice(accent="northern", preview_url="https://example.test/p.wav")],
                "total": 1,
                "next_cursor": "someday",
            },
        )
    )

    with client() as sdk:
        voices = sdk.list_voices()

    assert voices.data[0].name == "Narrator"


@respx.mock
def test_generation_status_is_a_plain_string_so_a_future_status_does_not_raise():
    respx.get(f"{BASE_URL}/v1/generations/gen_1").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "gen_1",
                "status": "queued_behind_a_new_status_nobody_shipped_yet",
                "profile": "Narrator",
                "has_audio": False,
                "audio_url": None,
                "error": None,
            },
        )
    )

    with client() as sdk:
        generation = sdk.get_generation("gen_1")

    assert isinstance(generation.status, str)
    assert generation.status == "queued_behind_a_new_status_nobody_shipped_yet"
