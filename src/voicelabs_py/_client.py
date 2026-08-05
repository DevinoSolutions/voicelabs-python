"""Request/response plumbing shared by the sync and async clients.

Everything that does not need to know whether it is running under ``await`` lives here: URL
building, header assembly, argument validation, base64 audio encoding, problem parsing, and
rate-limit extraction. The sync and async clients differ only in the I/O call itself, which is
the only way two clients stay behaviourally identical over time.
"""

from __future__ import annotations

import base64
import binascii
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, cast

import httpx

from ._version import __version__
from .errors import (
    VoiceLabsAPIError,
    VoiceLabsConfigError,
    VoiceLabsConnectionError,
    error_from_response,
)
from .models import (
    GENERATION_STATUS,
    MAX_AUDIO_BASE64_CHARS,
    MAX_CAPTURES_PAGE_SIZE,
    AudioFile,
    Generation,
)
from .rate_limit import RateLimitInfo, rate_limit_from_headers, retry_after_seconds

#: Where the public API lives. Overridable so a fork or a staging host can be pointed at.
DEFAULT_BASE_URL = "https://app.voicelabs.now"

API_KEY_HEADER = "x-api-key"
IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"
ENV_API_KEY = "VOICELABS_API_KEY"

USER_AGENT = f"voicelabs-py/{__version__}"

_MAX_IDEMPOTENCY_KEY_CHARS = 255


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    """One outbound request, fully resolved but not yet sent."""

    method: str
    url: str
    params: dict[str, Any] = field(default_factory=dict)
    json: Any | None = None
    headers: dict[str, str] = field(default_factory=dict)
    #: Whether a connection-level failure may be retried. False for a write without an
    #: idempotency key: the request may have succeeded server-side, and re-sending it blind can
    #: double-charge the account's allowance.
    retry_safe: bool = True


class BaseVoiceLabs:
    """State and pure logic shared by :class:`~voicelabs_py.VoiceLabs` and its async twin."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = 60.0,
        max_retries: int = 2,
        auth_style: str = "api_key",
        on_rate_limit: Callable[[RateLimitInfo], None] | None = None,
    ) -> None:
        self._api_key = _resolve_api_key(api_key)
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        if max_retries < 0:
            raise VoiceLabsConfigError("max_retries cannot be negative.")
        self.max_retries = max_retries
        if auth_style not in ("api_key", "bearer"):
            raise VoiceLabsConfigError(
                f'auth_style must be "api_key" or "bearer", not {auth_style!r}.'
            )
        self._auth_style = auth_style
        self._on_rate_limit = on_rate_limit
        self._rate_limit: RateLimitInfo | None = None

    @property
    def rate_limit(self) -> RateLimitInfo | None:
        """The rate-limit state read off the most recent response that advertised one.

        ``None`` means the SDK has not seen one yet, or the responses carried none — which is the
        case on the OAuth 2.1 lane and for a key with limiting switched off. ``None`` is never a
        claim that the budget is spent.
        """
        return self._rate_limit

    # ── header + request assembly ────────────────────────────────────────────

    def _default_headers(self) -> dict[str, str]:
        """The headers every authenticated request carries."""
        headers = {"accept": "application/json", "user-agent": USER_AGENT}
        if self._auth_style == "bearer":
            headers["authorization"] = f"Bearer {self._api_key}"
        else:
            headers[API_KEY_HEADER] = self._api_key
        return headers

    def _prepare(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        body: Any | None = None,
        idempotency_key: str | None = None,
    ) -> PreparedRequest:
        """Resolve a call into a :class:`PreparedRequest`.

        Caller-supplied headers are never allowed to displace authentication — a request that
        silently swapped the key for one supplied elsewhere would be a confusing way to leak a
        credential to the wrong account, so the auth header is set on the client and not here.
        """
        headers: dict[str, str] = {}
        if body is not None:
            headers["content-type"] = "application/json"
        if idempotency_key is not None:
            if len(idempotency_key) > _MAX_IDEMPOTENCY_KEY_CHARS:
                raise VoiceLabsConfigError(
                    f"idempotency_key is {len(idempotency_key)} characters; the API accepts at "
                    f"most {_MAX_IDEMPOTENCY_KEY_CHARS}."
                )
            headers[IDEMPOTENCY_KEY_HEADER] = idempotency_key

        params = {k: v for k, v in (query or {}).items() if v is not None}
        return PreparedRequest(
            method=method,
            url=f"{self.base_url}{path}",
            params=params,
            json=body,
            headers=headers,
            # A write is only safe to replay when the caller gave us a key that makes the server
            # deduplicate it.
            retry_safe=method == "GET" or idempotency_key is not None,
        )

    # ── response handling ────────────────────────────────────────────────────

    def _record_rate_limit(self, headers: Mapping[str, str]) -> RateLimitInfo | None:
        """Parse the draft-11 headers once, cache them, and hand them to the callback."""
        info = rate_limit_from_headers(headers)
        if info is None:
            return None
        self._rate_limit = info
        if self._on_rate_limit is not None:
            self._on_rate_limit(info)
        return info

    def _outcome(self, response: httpx.Response) -> Any:
        """Turn a completed response into a decoded body, or raise the right typed error.

        Args:
            response: The response, already fully read.

        Returns:
            The decoded JSON body.

        Raises:
            VoiceLabsAPIError: The API answered non-2xx.
        """
        rate_limit = self._record_rate_limit(response.headers)
        body = _decode_json(response)
        if response.is_success:
            return body
        raise error_from_response(
            response.status_code,
            body,
            rate_limit=rate_limit,
            retry_after=retry_after_seconds(response.headers),
        )


def bare_headers() -> dict[str, str]:
    """Headers for the signed audio capability URL — deliberately carrying NO credential.

    ``GET /v1/audio/{id}`` is declared ``security: []`` in the contract: the signature in the URL
    IS the authorization. Sending ``x-api-key`` there would hand a long-lived credential to
    whatever host the URL points at, for a request that does not need it.
    """
    return {"user-agent": USER_AGENT, "accept": "*/*"}


def audio_outcome(response: httpx.Response) -> AudioFile:
    """Turn a capability-URL response into audio bytes, or raise the right typed error.

    Args:
        response: The response, already fully read.

    Returns:
        The audio and the content type the server served it as, so a caller can name the file
        without guessing at the extension.

    Raises:
        VoiceLabsAPIError: The download was refused (401 on a stale signature, 404, 502, 503).
    """
    if not response.is_success:
        raise error_from_response(
            response.status_code,
            _decode_json(response),
            retry_after=retry_after_seconds(response.headers),
        )
    return AudioFile(
        content=response.content,
        content_type=response.headers.get("content-type", "application/octet-stream"),
    )


def generation_id_of(generation: Any) -> str:
    """The id to poll, from a generation, a speech handle, or a bare id string."""
    if isinstance(generation, str):
        return generation
    identifier = getattr(generation, "id", None)
    if not isinstance(identifier, str) or not identifier:
        raise VoiceLabsConfigError(
            "Pass a Generation, a SpeechGeneration, or a generation id string — got "
            f"{type(generation).__name__}."
        )
    return identifier


def poll_verdict(
    generation: Generation, elapsed: float, poll_interval: float, timeout: float
) -> str:
    """Decide what a polling loop does next: ``done``, ``failed``, ``wait``, or ``give_up``.

    The deadline is checked BEFORE sleeping so an expired one reports promptly rather than after
    one more full interval of waiting.
    """
    if generation.status == GENERATION_STATUS.completed:
        return "done"
    if generation.status == GENERATION_STATUS.failed:
        return "failed"
    if elapsed + poll_interval > timeout:
        return "give_up"
    return "wait"


def _resolve_api_key(api_key: str | None) -> str:
    """Take the explicit key, else the environment, else say precisely what is missing.

    The SDK deliberately does NOT validate the key's prefix. Every key issued today begins with
    ``vl_sandbox_``, but that is an honest marker rather than a permanent format, and a prefix
    check here would break every customer the day a second prefix ships.
    """
    resolved = api_key if api_key else os.environ.get(ENV_API_KEY)
    if not resolved:
        raise VoiceLabsConfigError(
            "A VoiceLabs API key is required. Create one at https://voicelabs.now/connections "
            'on the "API keys" tab, then pass it as VoiceLabs(api_key="vl_sandbox_…") or set '
            f"the {ENV_API_KEY} environment variable."
        )
    return resolved


def _decode_json(response: httpx.Response) -> Any:
    """Read a JSON body, tolerating an empty or non-JSON one (a proxy's HTML error page)."""
    if not response.content:
        return None
    try:
        return response.json()
    except ValueError:
        return None


def connection_error(cause: Exception, method: str, url: str) -> VoiceLabsConnectionError:
    """Wrap a transport-level failure, naming the request so the message is actionable."""
    if isinstance(cause, httpx.TimeoutException):
        message = f"{method} {url} timed out before the VoiceLabs API answered."
    else:
        message = f"{method} {url} could not reach the VoiceLabs API: {cause}"
    error = VoiceLabsConnectionError(message)
    error.__cause__ = cause
    return error


def clamp_page_size(page_size: int | None) -> int | None:
    """Hold a requested page size inside the range the contract documents (1–200)."""
    if page_size is None:
        return None
    return max(1, min(int(page_size), MAX_CAPTURES_PAGE_SIZE))


def speech_body(
    *,
    text: str,
    voice_id: str | None,
    voice_name: str | None,
    language: str | None,
) -> dict[str, Any]:
    """Validate and assemble the ``POST /v1/speech`` request body.

    Raises:
        VoiceLabsConfigError: ``text`` is empty, or both ``voice_id`` and ``voice_name`` were
            supplied — the API treats them as alternatives, and sending both leaves the choice to
            the server rather than to the caller.
    """
    if not text:
        raise VoiceLabsConfigError("text is required and cannot be empty.")
    if voice_id is not None and voice_name is not None:
        raise VoiceLabsConfigError(
            "Pass voice_id or voice_name, not both — they are two ways of naming one profile. "
            "Omit both to let the account's default voice answer."
        )

    body: dict[str, Any] = {"text": text}
    if voice_id is not None:
        body["voice_id"] = voice_id
    if voice_name is not None:
        body["voice_name"] = voice_name
    if language is not None:
        body["language"] = language
    return body


def transcription_body(
    *,
    audio: bytes | str | os.PathLike[str] | BinaryIO | None,
    audio_base64: str | None,
    language: str | None,
) -> dict[str, Any]:
    """Validate, encode, and size-check a ``POST /v1/transcriptions`` request body.

    The public surface is JSON with base64 audio — NOT multipart, whatever the internal capture
    pipeline does. The contract's 10 MiB ceiling applies to the base64 TEXT, i.e. roughly 7.5 MiB
    of raw audio, and it is enforced here so an oversized clip fails instantly instead of after a
    long upload.

    Args:
        audio: Raw bytes, a filesystem path, or an open binary file object.
        audio_base64: Already-encoded base64 text, for callers who have it. Mutually exclusive
            with ``audio``.
        language: An ISO language code, or ``None`` to let the API detect it.

    Returns:
        The request body.

    Raises:
        VoiceLabsConfigError: Neither or both audio arguments were given, the file could not be
            read, the base64 was malformed, or the encoded clip exceeds the documented cap.
    """
    if (audio is None) == (audio_base64 is None):
        raise VoiceLabsConfigError(
            "Pass exactly one of audio= (bytes, a path, or a binary file object) or "
            "audio_base64= (text you already encoded)."
        )

    if audio is not None:
        encoded = base64.b64encode(_audio_bytes(audio)).decode("ascii")
    else:
        encoded = (audio_base64 or "").strip()
        try:
            base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise VoiceLabsConfigError(
                f"audio_base64 is not valid base64 text: {exc}. Pass raw bytes as audio= and let "
                "the SDK encode them."
            ) from exc

    if len(encoded) > MAX_AUDIO_BASE64_CHARS:
        raise VoiceLabsConfigError(
            f"The audio is {len(encoded)} base64 characters; the API accepts at most "
            f"{MAX_AUDIO_BASE64_CHARS} (about {MAX_AUDIO_BASE64_CHARS // (1024 * 1024)} MiB of "
            "base64 text, roughly 7.5 MiB of raw audio). Split or re-encode the clip."
        )

    body: dict[str, Any] = {"audio": encoded}
    if language is not None:
        body["language"] = language
    return body


def _audio_bytes(audio: bytes | str | os.PathLike[str] | BinaryIO) -> bytes:
    """Read audio out of bytes, a path, or a binary file object."""
    if isinstance(audio, (bytes, bytearray, memoryview)):
        return bytes(audio)
    if isinstance(audio, (str, os.PathLike)):
        # A file object could structurally satisfy os.PathLike, so the cast states the intent the
        # isinstance check already established.
        path = Path(cast("str | os.PathLike[str]", audio))
        try:
            return path.read_bytes()
        except OSError as exc:
            raise VoiceLabsConfigError(f"Could not read the audio file at {path}: {exc}") from exc
    read = getattr(audio, "read", None)
    if read is None:
        raise VoiceLabsConfigError(
            "audio must be bytes, a filesystem path, or an object with a .read() method that "
            f"returns bytes — got {type(audio).__name__}."
        )
    data = read()
    if isinstance(data, str):
        raise VoiceLabsConfigError(
            "The audio file object returned text. Open it in binary mode, e.g. open(path, 'rb')."
        )
    return bytes(data)


def audio_url_of(generation: Any) -> str:
    """The signed capability URL to download, from a generation or a bare URL string.

    Raises:
        VoiceLabsConfigError: The generation has not finished, so there is nothing to download.
    """
    url = generation if isinstance(generation, str) else getattr(generation, "audio_url", None)
    if not url:
        raise VoiceLabsConfigError(
            "This generation has no audio_url yet. Wait for it to complete "
            "(wait_for_generation) before downloading — a running generation has nothing to "
            "download."
        )
    return url


__all__ = [
    "API_KEY_HEADER",
    "BaseVoiceLabs",
    "DEFAULT_BASE_URL",
    "ENV_API_KEY",
    "IDEMPOTENCY_KEY_HEADER",
    "PreparedRequest",
    "USER_AGENT",
    "VoiceLabsAPIError",
    "audio_outcome",
    "audio_url_of",
    "bare_headers",
    "clamp_page_size",
    "connection_error",
    "generation_id_of",
    "poll_verdict",
    "speech_body",
    "transcription_body",
]
