"""Typed errors, parsed from the API's RFC 9457 problem documents.

THE REASON THIS FILE IS THE MOST IMPORTANT PART OF THE SDK: the VoiceLabs API deliberately
returns two DIFFERENT problems under one HTTP status, twice::

    429 -> rate_limit_exceeded  you are calling too fast. Wait and retry; it will succeed.
    429 -> quota_exhausted      the account's audio allowance is spent. Retrying is futile
                                until the period rolls over or somebody upgrades.

    403 -> insufficient_scope   this credential was not granted the scope. Mint a wider key.
    403 -> feature_not_enabled  the plan does not include the capability at all.

A client that branches on ``response.status_code`` gets both pairs wrong, and the second mistake
is expensive: retrying ``quota_exhausted`` on a backoff loop burns the caller's rate budget
forever without ever succeeding. So the SDK branches on the problem document's ``code`` extension
member, and gives each meaning its own class — ``isinstance`` and ``err.code`` agree, always.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .rate_limit import RateLimitInfo

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, types only
    from .models import Generation

__all__ = [
    "AuthenticationError",
    "ERROR_CODES",
    "FeatureNotEnabledError",
    "GenerationFailedError",
    "GenerationTimeoutError",
    "IdempotencyError",
    "InsufficientScopeError",
    "InvalidRequestError",
    "MethodNotAllowedError",
    "NotFoundError",
    "ProblemDocument",
    "QuotaExhaustedError",
    "RateLimitError",
    "ServerError",
    "VoiceLabsAPIError",
    "VoiceLabsConfigError",
    "VoiceLabsConnectionError",
    "VoiceLabsError",
    "error_from_response",
]

#: Every error code the /v1 surface can emit. Codes may be ADDED but never renamed, which is why
#: an unfamiliar one degrades to :class:`VoiceLabsAPIError` instead of raising a parse failure.
ERROR_CODES: tuple[str, ...] = (
    "invalid_request",
    "unauthorized",
    "insufficient_scope",
    "feature_not_enabled",
    "not_found",
    "method_not_allowed",
    "idempotency_key_in_progress",
    "idempotency_key_reused",
    "rate_limit_exceeded",
    "quota_exhausted",
    "upstream_error",
    "internal_error",
    "service_unavailable",
)


@dataclass(frozen=True, slots=True)
class ProblemDocument:
    """An RFC 9457 problem document as the VoiceLabs API emits it.

    Attributes:
        type: The problem type URI — a stable identifier, and a documentation page.
        title: A short human-readable summary.
        status: The HTTP status the API answered with.
        detail: The API's human-readable explanation. Safe to log; never contains a credential.
        code: The stable token to branch on. Prefer this over ``status``.
        instance: The request path the problem occurred on, when the API supplied one.
        required_scope: On ``insufficient_scope``, the scope this credential was missing.
        settings_url: On ``quota_exhausted`` / ``feature_not_enabled``, where the account owner
            resolves it.
    """

    type: str
    title: str
    status: int
    detail: str
    code: str
    instance: str | None = None
    required_scope: str | None = None
    settings_url: str | None = None

    @classmethod
    def from_payload(cls, payload: Any, *, status: int) -> ProblemDocument | None:
        """Build a problem document from a decoded response body.

        Args:
            payload: The decoded JSON body, or ``None`` when the body was empty or not JSON.
            status: The HTTP status, used when the body omits one.

        Returns:
            The problem document, or ``None`` when the body is not one — an HTML error page from
            a proxy, an empty 502, a gateway timeout. The caller synthesises a detail in that
            case rather than raising: "the error itself errored" is not something a caller's
            ``except`` block should have to handle.
        """
        if not isinstance(payload, dict):
            return None
        code = payload.get("code")
        detail = payload.get("detail")
        if not isinstance(code, str) or not isinstance(detail, str):
            return None
        raw_status = payload.get("status")
        return cls(
            type=payload["type"] if isinstance(payload.get("type"), str) else _type_uri(code),
            title=payload["title"] if isinstance(payload.get("title"), str) else code,
            status=raw_status if isinstance(raw_status, int) else status,
            detail=detail,
            code=code,
            instance=payload.get("instance") if isinstance(payload.get("instance"), str) else None,
            required_scope=(
                payload.get("required_scope")
                if isinstance(payload.get("required_scope"), str)
                else None
            ),
            settings_url=(
                payload.get("settings_url")
                if isinstance(payload.get("settings_url"), str)
                else None
            ),
        )


def _type_uri(code: str) -> str:
    return f"https://voicelabs.now/errors/{code}"


class VoiceLabsError(Exception):
    """Base class for everything this SDK raises.

    ``except VoiceLabsError`` catches every failure the SDK can produce, including the ones that
    never reached the network.
    """


class VoiceLabsConfigError(VoiceLabsError):
    """A local problem: no API key, oversized audio, or mutually exclusive arguments.

    Raised BEFORE any request is made, so nothing on the server has happened.
    """


class VoiceLabsConnectionError(VoiceLabsError):
    """The request never produced an HTTP response.

    DNS failure, connection reset, TLS problem, or a timeout. Distinct from every API error
    because nothing on the server is known to have happened — a write may or may not have been
    executed, which is exactly when an idempotency key earns its keep.

    The underlying httpx exception is available as ``__cause__``.
    """


class VoiceLabsAPIError(VoiceLabsError):
    """Any error the API reported as a problem document (or an unparseable non-2xx response).

    Attributes:
        code: The stable token to branch on. Prefer this over ``status``.
        status: The HTTP status code.
        detail: The API's human-readable explanation. Safe to log; never contains a credential.
        problem: The parsed problem document, or ``None`` when the body was not one (a proxy's
            HTML error page, an empty 502).
        rate_limit: The rate-limit state at the time of the failure, when the response
            advertised it.
        retry_after: Seconds the API asked the caller to wait, from ``Retry-After``.
    """

    def __init__(
        self,
        *,
        code: str,
        status: int,
        detail: str,
        title: str,
        type: str,
        instance: str | None = None,
        problem: ProblemDocument | None = None,
        rate_limit: RateLimitInfo | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(f"{title} ({code}): {detail}")
        self.code = code
        self.status = status
        self.detail = detail
        self.title = title
        self.type = type
        self.instance = instance
        self.problem = problem
        self.rate_limit = rate_limit
        self.retry_after = retry_after

    @property
    def required_scope(self) -> str | None:
        """The scope this credential was missing, on ``insufficient_scope``."""
        return self.problem.required_scope if self.problem else None

    @property
    def settings_url(self) -> str | None:
        """Where the account owner resolves a quota or entitlement problem."""
        return self.problem.settings_url if self.problem else None

    @property
    def retryable(self) -> bool:
        """Whether retrying this exact request could plausibly succeed on its own.

        ``quota_exhausted`` is deliberately ``False`` despite being a 429: the allowance is spent,
        and a backoff loop would spin until the billing period rolls over. ``insufficient_scope``
        is ``False`` for the same kind of reason — the credential has to change first.
        """
        return self.code in _RETRYABLE_CODES


_RETRYABLE_CODES = frozenset(
    {
        "rate_limit_exceeded",
        "idempotency_key_in_progress",
        "upstream_error",
        "internal_error",
        "service_unavailable",
    }
)


class InvalidRequestError(VoiceLabsAPIError):
    """400 ``invalid_request`` — the request did not satisfy the operation's schema."""


class AuthenticationError(VoiceLabsAPIError):
    """401 ``unauthorized`` — no credential, an unusable one, or a browser session.

    ``/v1`` never accepts a browser session cookie; it wants an API key.
    """


class InsufficientScopeError(VoiceLabsAPIError):
    """403 ``insufficient_scope`` — the credential is valid but lacks the required scope.

    ``required_scope`` names the scope the key needed. Mint a new key carrying it in the
    developer console at https://voicelabs.now/connections.
    """


class FeatureNotEnabledError(VoiceLabsAPIError):
    """403 ``feature_not_enabled`` — the account's plan does not include this capability at all.

    ``settings_url`` is where the account owner resolves it.
    """


class NotFoundError(VoiceLabsAPIError):
    """404 ``not_found`` — no such resource on THIS account.

    Another tenant's id is a 404, never a 403: existence is itself information.
    """


class MethodNotAllowedError(VoiceLabsAPIError):
    """405 ``method_not_allowed`` — the path exists but not with this HTTP method."""


class IdempotencyError(VoiceLabsAPIError):
    """409 / 422 — an ``Idempotency-Key`` is in flight, or was replayed with a different body.

    ``code`` tells the two apart: ``idempotency_key_in_progress`` is worth retrying shortly,
    ``idempotency_key_reused`` means the key is already bound to a different request body and a
    new key is needed.
    """


class RateLimitError(VoiceLabsAPIError):
    """429 ``rate_limit_exceeded`` — too many requests. Waiting fixes this.

    ``retry_after`` carries the server's requested delay when it supplied one.
    """


class QuotaExhaustedError(VoiceLabsAPIError):
    """429 ``quota_exhausted`` — the account's metered audio allowance is spent.

    NOT a subclass of :class:`RateLimitError`, on purpose: a caller who writes
    ``except RateLimitError: retry_later()`` must not silently pick this up and retry forever
    against an allowance that will not move until the period rolls over. ``settings_url`` is
    where the account owner upgrades or reviews usage.
    """


class ServerError(VoiceLabsAPIError):
    """5xx ``upstream_error`` / ``internal_error`` / ``service_unavailable``.

    The API or something it depends on failed. Not the caller's request.
    """


class GenerationFailedError(VoiceLabsError):
    """The generation finished in a ``failed`` state. Raised by the polling helper.

    Attributes:
        generation_id: The generation that failed.
        reason: The engine's reason, when it gave one.
        generation: The last-seen generation object.
    """

    def __init__(self, generation: Generation) -> None:
        reason = generation.error
        super().__init__(f"Generation {generation.id} failed{f': {reason}' if reason else '.'}")
        self.generation = generation
        self.generation_id = generation.id
        self.reason = reason


class GenerationTimeoutError(VoiceLabsError):
    """The polling helper gave up before the generation reached a terminal state.

    The timeout is a REAL deadline, not a poll count. A generation that was still running when it
    expired has NOT been cancelled, and re-polling with ``get_generation`` may still find it
    completed — which is what the message says, because silently returning an unfinished
    generation would push the failure into the caller's audio pipeline instead.

    Attributes:
        generation: The last-seen generation, still unfinished.
        generation_id: Its id, so it can be polled again.
        waited_seconds: How long the helper waited.
    """

    def __init__(self, generation: Generation, waited_seconds: float) -> None:
        super().__init__(
            f"Generation {generation.id} was still running after {waited_seconds:.1f}s. "
            "The generation has NOT been cancelled and may still complete — re-poll it with "
            "get_generation()."
        )
        self.generation = generation
        self.generation_id = generation.id
        self.waited_seconds = waited_seconds


_ERROR_CLASSES: dict[str, type[VoiceLabsAPIError]] = {
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


def _synthesized_code(status: int) -> str:
    """The closest honest code for a response that carried no problem document."""
    return {
        400: "invalid_request",
        401: "unauthorized",
        403: "insufficient_scope",
        404: "not_found",
        405: "method_not_allowed",
        429: "rate_limit_exceeded",
        502: "upstream_error",
        503: "service_unavailable",
    }.get(status, "internal_error")


def error_from_response(
    status: int,
    body: Any,
    *,
    rate_limit: RateLimitInfo | None = None,
    retry_after: float | None = None,
) -> VoiceLabsAPIError:
    """Turn a failed response into the right typed error.

    A body that is not a problem document — an HTML error page from a proxy, an empty 502, a
    gateway timeout — still produces a :class:`VoiceLabsAPIError` with a code SYNTHESISED from the
    status rather than an opaque parse failure, and with ``problem`` left ``None`` so the caller
    can tell a real problem document from a guess.

    An UNKNOWN code from a newer API version is kept verbatim on ``err.code`` and handled through
    the base class. Forcing it into a familiar class would be worse than admitting we do not know
    it: a future ``payment_required`` silently caught as a rate limit is a retry loop nobody
    wrote.

    Args:
        status: The HTTP status code of the failing response.
        body: The decoded JSON body, or ``None`` when it was empty or not JSON.
        rate_limit: Rate-limit state read off the same response, when present.
        retry_after: ``Retry-After`` seconds read off the same response, when present.

    Returns:
        A :class:`VoiceLabsAPIError` subclass matching the problem ``code``.
    """
    problem = ProblemDocument.from_payload(body, status=status)

    if problem is None:
        code = _synthesized_code(status)
        error_class = _ERROR_CLASSES.get(code, VoiceLabsAPIError)
        return error_class(
            code=code,
            status=status,
            detail=(
                f"The VoiceLabs API returned HTTP {status} without a problem document. This "
                "usually means something between your client and the API answered instead of "
                "the API itself."
            ),
            title=f"HTTP {status}",
            type=_type_uri(code),
            problem=None,
            rate_limit=rate_limit,
            retry_after=retry_after,
        )

    error_class = _ERROR_CLASSES.get(problem.code, VoiceLabsAPIError)
    return error_class(
        code=problem.code,
        status=problem.status,
        detail=problem.detail,
        title=problem.title,
        type=problem.type,
        instance=problem.instance,
        problem=problem,
        rate_limit=rate_limit,
        retry_after=retry_after,
    )
