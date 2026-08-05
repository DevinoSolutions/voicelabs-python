"""Parse the IETF draft-11 rate-limit headers the VoiceLabs API emits.

Draft 11 is a DIFFERENT wire format from the widely-copied ``X-RateLimit-*`` triple, and this is
where most hand-written clients get it wrong. It is two RFC 8941 structured fields, each keyed by
a quoted policy name::

    RateLimit-Policy: "default";q=600;w=3600    <- the standing policy (quota, window seconds)
    RateLimit:        "default";r=599;t=2718    <- this caller's state (remaining, seconds to reset)

The headers describe an API KEY'S budget, so they are absent on the OAuth 2.1 lane and absent for
a key with limiting switched off. Absent means UNKNOWN, never zero — a client that read a missing
header as "0 remaining" would stop making requests it is perfectly entitled to make.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime

__all__ = ["RateLimitInfo", "rate_limit_from_headers", "retry_after_seconds"]

_POLICY_NAME = re.compile(r'^\s*"([^"]*)"')


@dataclass(frozen=True, slots=True)
class RateLimitInfo:
    """What the API said about this credential's request budget.

    Every numeric field is ``None`` when the API did not report it. ``None`` means UNKNOWN, never
    zero: a caller that treated a missing ``remaining`` as "no requests left" would throttle
    itself out of requests it is entitled to make.

    Attributes:
        policy: The policy name the API keys its limits by. Today always ``"default"``.
        limit: Requests permitted per window, or ``None`` when the policy header was absent.
        window_seconds: Window length in seconds, or ``None`` when the policy header was absent.
        remaining: Requests left in the current window, or ``None`` when the state header was
            absent.
        reset_seconds: Seconds until the window resets, or ``None`` when the state header was
            absent.
    """

    policy: str
    limit: int | None
    window_seconds: int | None
    remaining: int | None
    reset_seconds: int | None


def _header(headers: Mapping[str, str], name: str) -> str | None:
    """Read one header case-insensitively, whether ``headers`` is httpx's or a plain dict."""
    value = headers.get(name)
    if value is not None:
        return value
    for key, candidate in headers.items():
        if key.lower() == name:
            return candidate
    return None


def _structured_parameter(header: str, key: str) -> int | None:
    """Read one parameter out of an RFC 8941 structured field of the shape ``"name";k=v;k2=v2``.

    Deliberately tolerant about spacing (``"default"; r=5`` is legal) and deliberately strict
    about the value: a non-numeric parameter yields ``None`` rather than a NaN float, because NaN
    silently poisons every arithmetic comparison a caller would write against it.

    Args:
        header: The raw header value.
        key: The parameter name to read, e.g. ``"r"``.

    Returns:
        The parameter as an int, or ``None`` when it is absent or not a finite number.
    """
    for segment in header.split(";")[1:]:
        separator = segment.find("=")
        if separator == -1:
            continue
        if segment[:separator].strip() != key:
            continue
        try:
            value = float(segment[separator + 1 :].strip())
        except ValueError:
            return None
        # float() happily accepts "nan" and "inf"; neither is a budget a caller can subtract from.
        return int(value) if math.isfinite(value) else None
    return None


def _policy_name(header: str) -> str | None:
    """The quoted policy name at the head of a structured field, or ``None`` when it is absent."""
    match = _POLICY_NAME.match(header)
    return match.group(1) if match else None


def rate_limit_from_headers(headers: Mapping[str, str]) -> RateLimitInfo | None:
    """Read the rate-limit state off a response, or ``None`` when the response carried none.

    Returning ``None`` rather than an all-``None`` object is the point: "this response said
    nothing about limits" and "this response said the limit is unknown" are the same fact, and
    there is no reason to hand a caller an object whose every field they must null-check before
    use.

    Args:
        headers: The response headers.

    Returns:
        A :class:`RateLimitInfo`, or ``None`` when neither draft-11 header was present.
    """
    state = _header(headers, "ratelimit")
    policy = _header(headers, "ratelimit-policy")
    if not state and not policy:
        return None

    return RateLimitInfo(
        policy=_policy_name(state or policy or "") or "default",
        limit=_structured_parameter(policy, "q") if policy else None,
        window_seconds=_structured_parameter(policy, "w") if policy else None,
        remaining=_structured_parameter(state, "r") if state else None,
        reset_seconds=_structured_parameter(state, "t") if state else None,
    )


def retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    """Seconds from a ``Retry-After`` header.

    The delta-seconds form is honoured first; the HTTP-date form is read as a date and converted.
    Both are clamped at zero so a skewed clock cannot produce a negative wait that a caller would
    pass straight to :func:`time.sleep`.

    Args:
        headers: The response headers.

    Returns:
        Seconds to wait, or ``None`` when the header was absent or unparseable.
    """
    value = _header(headers, "retry-after")
    if not value:
        return None

    try:
        seconds = float(value.strip())
    except ValueError:
        pass
    else:
        return max(0.0, seconds) if math.isfinite(seconds) else None

    try:
        parsed = parsedate_to_datetime(value.strip())
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    now = dt.datetime.now(dt.timezone.utc)
    return max(0.0, float(math.ceil((parsed - now).total_seconds())))
