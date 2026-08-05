"""When to try again, and how long to wait first.

The retry layer is deliberately narrow. Retrying the wrong failure is worse than not retrying at
all, and this API has two failures that look retryable and are not:

* ``quota_exhausted`` arrives as a 429, like a rate limit — but the account's allowance is spent,
  so a backoff loop would spin until the billing period rolls over, burning the caller's rate
  budget forever without ever succeeding. It is never retried.
* A connection error on a POST means the SDK does not know whether the write happened. Re-sending
  it blind can generate — and charge for — the same audio twice. A write is only replayed when
  the caller supplied an ``Idempotency-Key``, which is precisely what makes the server
  deduplicate it.
"""

from __future__ import annotations

import random

from .errors import VoiceLabsAPIError, VoiceLabsConnectionError, VoiceLabsError

__all__ = ["BASE_DELAY_SECONDS", "MAX_DELAY_SECONDS", "RETRYABLE_CODES", "plan_retry"]

#: First backoff step, doubled per attempt.
BASE_DELAY_SECONDS = 0.5

#: Ceiling on a computed backoff. A server-supplied ``Retry-After`` is honoured as given: the API
#: knows when its window rolls over and this SDK does not.
MAX_DELAY_SECONDS = 20.0

#: The problem codes the SDK retries on its own.
#:
#: ``idempotency_key_in_progress`` is NOT here even though :attr:`VoiceLabsAPIError.retryable` is
#: true for it: that flag is advice to a caller who knows how long their own first request takes,
#: while an immediate automatic replay would just collect another 409.
RETRYABLE_CODES = frozenset(
    {
        "rate_limit_exceeded",
        "upstream_error",
        "internal_error",
        "service_unavailable",
    }
)


def _full_jitter(ceiling: float) -> float:
    """A uniform sample in ``[0, ceiling]`` — full jitter, not equal jitter.

    Full jitter is what keeps a fleet of clients that were rate-limited by the same window from
    re-colliding on the same retry tick.
    """
    return random.uniform(0.0, ceiling)


def backoff_delay(attempt: int) -> float:
    """Exponential backoff with full jitter for a zero-based attempt number."""
    return _full_jitter(min(MAX_DELAY_SECONDS, BASE_DELAY_SECONDS * (2**attempt)))


def _server_requested_delay(error: VoiceLabsAPIError) -> float | None:
    """The wait the API itself asked for: ``Retry-After`` first, then the draft-11 reset."""
    if error.retry_after is not None:
        return error.retry_after
    if error.rate_limit is not None and error.rate_limit.reset_seconds is not None:
        return float(error.rate_limit.reset_seconds)
    return None


def plan_retry(
    error: VoiceLabsError,
    *,
    attempt: int,
    max_retries: int,
    retry_safe: bool,
    remaining_budget: float | None,
) -> float | None:
    """Decide whether to retry, and how long to wait first.

    Args:
        error: The failure from the attempt that just finished.
        attempt: Zero-based index of the attempt that failed.
        max_retries: Extra attempts allowed after the first.
        retry_safe: False for a write with no idempotency key — the server may already have done
            the work, and replaying it can double-charge the account.
        remaining_budget: Seconds left in the client's timeout budget, or ``None`` for no budget.
            A wait that would not fit is refused rather than taken silently.

    Returns:
        Seconds to wait before the next attempt, or ``None`` to give up and raise.
    """
    if attempt >= max_retries or not retry_safe:
        return None

    if isinstance(error, VoiceLabsConnectionError):
        # Nothing on the server is known to have happened. `retry_safe` above is what makes this
        # sound for writes.
        delay = backoff_delay(attempt)
    elif isinstance(error, VoiceLabsAPIError):
        if error.code not in RETRYABLE_CODES:
            return None
        if error.code == "rate_limit_exceeded":
            delay = _server_requested_delay(error) or backoff_delay(attempt)
        else:
            delay = backoff_delay(attempt)
    else:
        return None

    if remaining_budget is not None and delay > remaining_budget:
        # Sleeping past the caller's timeout would turn "this call takes at most N seconds" into
        # a quiet lie. Raise the error the caller can see instead.
        return None
    return delay
