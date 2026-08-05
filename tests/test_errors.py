"""The RFC 9457 error hierarchy — the part of the SDK that earns its keep."""

from __future__ import annotations

import pytest

from tests.helpers import problem
from voicelabs_py.errors import (
    ERROR_CODES,
    AuthenticationError,
    FeatureNotEnabledError,
    IdempotencyError,
    InsufficientScopeError,
    InvalidRequestError,
    MethodNotAllowedError,
    NotFoundError,
    QuotaExhaustedError,
    RateLimitError,
    ServerError,
    VoiceLabsAPIError,
    VoiceLabsError,
    error_from_response,
)
from voicelabs_py.rate_limit import RateLimitInfo

CODE_TO_CLASS = {
    "invalid_request": InvalidRequestError,
    "unauthorized": AuthenticationError,
    "insufficient_scope": InsufficientScopeError,
    "feature_not_enabled": FeatureNotEnabledError,
    "not_found": NotFoundError,
    "method_not_allowed": MethodNotAllowedError,
    "idempotency_key_in_progress": IdempotencyError,
    "idempotency_key_reused": IdempotencyError,
    "rate_limit_exceeded": RateLimitError,
    "quota_exhausted": QuotaExhaustedError,
    "upstream_error": ServerError,
    "internal_error": ServerError,
    "service_unavailable": ServerError,
}


def test_a_429_rate_limit_exceeded_and_a_429_quota_exhausted_raise_different_classes():
    too_fast = error_from_response(429, problem(code="rate_limit_exceeded", status=429))
    spent = error_from_response(429, problem(code="quota_exhausted", status=429))

    assert isinstance(too_fast, RateLimitError)
    assert isinstance(spent, QuotaExhaustedError)
    # The whole point: a caller catching RateLimitError to back off must not catch the one that
    # will never succeed no matter how long they wait.
    assert not isinstance(spent, RateLimitError)
    assert not isinstance(too_fast, QuotaExhaustedError)
    assert too_fast.retryable is True
    assert spent.retryable is False


def test_a_403_insufficient_scope_and_a_403_feature_not_enabled_raise_different_classes():
    scope = error_from_response(403, problem(code="insufficient_scope", status=403))
    feature = error_from_response(403, problem(code="feature_not_enabled", status=403))

    assert isinstance(scope, InsufficientScopeError)
    assert isinstance(feature, FeatureNotEnabledError)
    assert not isinstance(scope, FeatureNotEnabledError)
    assert not isinstance(feature, InsufficientScopeError)


def test_insufficient_scope_carries_the_required_scope_off_the_problem_document():
    error = error_from_response(
        403, problem(code="insufficient_scope", status=403, required_scope="voice:generate")
    )

    assert isinstance(error, InsufficientScopeError)
    assert error.required_scope == "voice:generate"


def test_quota_exhausted_carries_the_settings_url_a_human_can_act_on():
    error = error_from_response(
        429,
        problem(
            code="quota_exhausted",
            status=429,
            settings_url="https://voicelabs.now/settings/billing",
        ),
    )

    assert isinstance(error, QuotaExhaustedError)
    assert error.settings_url == "https://voicelabs.now/settings/billing"


def test_an_unknown_error_code_degrades_to_the_base_api_error_instead_of_crashing():
    error = error_from_response(402, problem(code="payment_required", status=402))

    assert type(error) is VoiceLabsAPIError
    assert error.code == "payment_required"
    assert error.status == 402
    # Not silently caught as something familiar — a future code caught as a rate limit would be a
    # retry loop nobody wrote.
    assert not isinstance(error, RateLimitError)


def test_a_non_json_error_body_still_raises_a_typed_api_error():
    error = error_from_response(502, None)

    assert isinstance(error, ServerError)
    assert error.code == "upstream_error"
    assert error.problem is None
    assert "without a problem document" in error.detail

    html = error_from_response(404, "<html>nope</html>")
    assert isinstance(html, NotFoundError)
    assert html.problem is None


def test_isinstance_and_the_code_attribute_always_agree():
    for code, expected in CODE_TO_CLASS.items():
        error = error_from_response(400, problem(code=code, status=400))
        assert isinstance(error, expected), code
        assert error.code == code


def test_every_documented_error_code_maps_to_an_exception_class():
    assert set(ERROR_CODES) == set(CODE_TO_CLASS)
    for code in ERROR_CODES:
        error = error_from_response(500, problem(code=code, status=500))
        assert type(error) is not VoiceLabsAPIError, code


def test_every_api_error_is_catchable_as_the_one_base_class():
    error = error_from_response(401, problem(code="unauthorized", status=401))

    with pytest.raises(VoiceLabsError):
        raise error


def test_the_rate_limit_state_of_the_failing_response_rides_along_on_the_error():
    info = RateLimitInfo(
        policy="default", limit=600, window_seconds=3600, remaining=0, reset_seconds=30
    )
    error = error_from_response(
        429, problem(code="rate_limit_exceeded", status=429), rate_limit=info, retry_after=30.0
    )

    assert error.rate_limit == info
    assert error.retry_after == 30.0


def test_the_error_message_names_the_title_the_code_and_the_servers_detail():
    error = error_from_response(
        401,
        problem(
            code="unauthorized",
            status=401,
            title="Unauthorized",
            detail="This endpoint requires an API key.",
        ),
    )

    assert str(error) == "Unauthorized (unauthorized): This endpoint requires an API key."
