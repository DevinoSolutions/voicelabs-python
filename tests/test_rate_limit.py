"""The draft-11 rate-limit header parser."""

from __future__ import annotations

import datetime as dt
from email.utils import format_datetime

from tests.helpers import draft_11_headers
from voicelabs_py.rate_limit import RateLimitInfo, rate_limit_from_headers, retry_after_seconds


def test_draft_11_structured_fields_parse_into_limit_window_remaining_and_reset():
    info = rate_limit_from_headers(draft_11_headers())

    assert info == RateLimitInfo(
        policy="default",
        limit=600,
        window_seconds=3600,
        remaining=599,
        reset_seconds=2718,
    )


def test_the_state_header_alone_yields_remaining_and_reset_and_leaves_the_policy_unknown():
    info = rate_limit_from_headers({"RateLimit": '"default";r=599;t=2718'})

    assert info == RateLimitInfo(
        policy="default",
        limit=None,
        window_seconds=None,
        remaining=599,
        reset_seconds=2718,
    )


def test_absent_rate_limit_headers_read_as_unknown_and_never_as_zero():
    assert rate_limit_from_headers({}) is None
    assert rate_limit_from_headers({"content-type": "application/json"}) is None


def test_a_non_numeric_structured_parameter_yields_none_rather_than_nan():
    info = rate_limit_from_headers({"RateLimit": '"default";r=soon;t=nan'})

    assert info is not None
    assert info.remaining is None
    assert info.reset_seconds is None


def test_structured_field_parsing_tolerates_spacing_and_an_unquoted_policy():
    info = rate_limit_from_headers({"ratelimit": '"burst"; r=5 ; t=10'})

    assert info is not None
    assert info.policy == "burst"
    assert info.remaining == 5
    assert info.reset_seconds == 10


def test_retry_after_accepts_delta_seconds_and_http_date_and_clamps_negative_to_zero():
    assert retry_after_seconds({"Retry-After": "42"}) == 42.0
    assert retry_after_seconds({"retry-after": " 7 "}) == 7.0

    future = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=120)
    parsed = retry_after_seconds({"Retry-After": format_datetime(future)})
    assert parsed is not None
    assert 100 <= parsed <= 125

    past = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)
    assert retry_after_seconds({"Retry-After": format_datetime(past)}) == 0.0

    assert retry_after_seconds({}) is None
    assert retry_after_seconds({"Retry-After": "whenever"}) is None
