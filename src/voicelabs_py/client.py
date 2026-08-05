"""The synchronous VoiceLabs API client.

Hand-written, not generated. Six operations do not need a code generator, and a generated client
could not make the judgement calls that are the whole value here: telling ``quota_exhausted``
apart from ``rate_limit_exceeded`` under a shared 429, fetching the audio capability URL WITHOUT
the API key, or refusing to poll a generation forever.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import quote

import httpx

from ._client import (
    BaseVoiceLabs,
    PreparedRequest,
    audio_outcome,
    audio_url_of,
    bare_headers,
    clamp_page_size,
    connection_error,
    generation_id_of,
    poll_verdict,
    speech_body,
    transcription_body,
)
from ._retry import plan_retry
from .errors import (
    GenerationFailedError,
    GenerationTimeoutError,
    VoiceLabsAPIError,
    VoiceLabsError,
)
from .models import (
    AudioFile,
    Capture,
    CaptureList,
    Generation,
    SpeechGeneration,
    Transcription,
    VoiceList,
)
from .rate_limit import RateLimitInfo

__all__ = ["VoiceLabs"]


class VoiceLabs(BaseVoiceLabs):
    """A synchronous client for the VoiceLabs public API.

    Args:
        api_key: Your API key, created at https://voicelabs.now/connections on the "API keys"
            tab. Falls back to the ``VOICELABS_API_KEY`` environment variable.
        base_url: Override the API origin. Defaults to ``https://app.voicelabs.now``; a trailing
            slash is stripped for you.
        timeout: Per-request timeout in seconds. Defaults to 60. Pass ``None`` to disable.
        max_retries: Extra attempts after the first for retryable failures. Defaults to 2, so a
            retryable request is tried three times in total. Pass 0 to disable retries entirely.
        auth_style: ``"api_key"`` (default, sends ``x-api-key``) or ``"bearer"`` (sends
            ``Authorization: Bearer``) for HTTP layers that only understand bearer tokens.
        on_rate_limit: Called after every response that advertised rate-limit headers — the place
            to implement pre-emptive backoff. Never called for responses that carried no budget
            to report.

    Raises:
        VoiceLabsConfigError: No API key was given and none was in the environment.
    """

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
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            auth_style=auth_style,
            on_rate_limit=on_rate_limit,
        )
        self._http = httpx.Client(
            timeout=self.timeout,
            headers=self._default_headers(),
            follow_redirects=True,
        )

    # ── lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying HTTP transport."""
        self._http.close()

    def __enter__(self) -> VoiceLabs:
        """Enter a ``with`` block; the transport is closed on exit."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the transport when the ``with`` block ends."""
        self.close()

    # ── reads (scope voice:read) ─────────────────────────────────────────────

    def list_voices(self) -> VoiceList:
        """Every voice profile on the account: cloned voices and presets.

        The endpoint is not paginated — it returns everything, with a ``total`` that equals the
        number of rows.

        Returns:
            A :class:`~voicelabs_py.VoiceList`.

        Raises:
            VoiceLabsAPIError: The API answered with a problem document.
            VoiceLabsConnectionError: The request never reached the API.
        """
        return VoiceList.from_payload(self._send(self._prepare("GET", "/v1/voices")))

    def list_captures(self, *, limit: int | None = None, offset: int | None = None) -> CaptureList:
        """One page of captures with their transcripts, most-recent first.

        Args:
            limit: Rows per page, 1–200. Defaults to the API's 50. Values above 200 are clamped
                to 200 rather than rejected, because the server would clamp them anyway.
            offset: Rows to skip. Defaults to 0.

        Returns:
            A :class:`~voicelabs_py.CaptureList` echoing back the ``limit`` and ``offset`` the
            SERVER applied — page with those, not with the numbers you asked for.

        Raises:
            VoiceLabsAPIError: The API answered with a problem document.
            VoiceLabsConnectionError: The request never reached the API.
        """
        prepared = self._prepare(
            "GET",
            "/v1/captures",
            query={"limit": clamp_page_size(limit), "offset": offset},
        )
        return CaptureList.from_payload(self._send(prepared))

    def iter_captures(self, *, page_size: int = 200, offset: int = 0) -> Iterator[Capture]:
        """Every capture on the account, paging automatically.

        Pages by ``offset`` using the limit and offset the SERVER echoed rather than the ones
        requested — the API clamps ``limit`` to its own maximum, and a client that assumed its own
        number would skip rows on every page after the first.

        Args:
            page_size: Rows to request per page, clamped to the documented maximum of 200.
            offset: Where to start.

        Yields:
            Each :class:`~voicelabs_py.Capture`, once.
        """
        cursor = offset
        while True:
            page = self.list_captures(limit=page_size, offset=cursor)
            yield from page.data
            if not page.data:
                return
            cursor = page.offset + len(page.data)
            if cursor >= page.total:
                return

    def get_generation(self, generation_id: str) -> Generation:
        """One generation's current state.

        Poll this, or let :meth:`wait_for_generation` do it. Each poll re-mints a fresh
        ``audio_url``; it is short-lived, so follow it promptly rather than storing it.

        Args:
            generation_id: The generation to read.

        Returns:
            A :class:`~voicelabs_py.Generation`.

        Raises:
            NotFoundError: No such generation on this account.
            VoiceLabsAPIError: The API answered with a problem document.
            VoiceLabsConnectionError: The request never reached the API.
        """
        prepared = self._prepare("GET", f"/v1/generations/{_path_segment(generation_id)}")
        return Generation.from_payload(self._send(prepared))

    # ── writes (scope voice:generate) ────────────────────────────────────────

    def create_speech(
        self,
        *,
        text: str,
        voice_id: str | None = None,
        voice_name: str | None = None,
        language: str | None = None,
        idempotency_key: str | None = None,
    ) -> SpeechGeneration:
        """Start generating speech. Returns IMMEDIATELY with a handle — the audio is not ready.

        Follow with :meth:`wait_for_generation`, or use :meth:`generate_speech` to do both.

        Args:
            text: What to say, 1–10000 characters.
            voice_id: A voice profile id. Mutually exclusive with ``voice_name``.
            voice_name: A voice profile name. Mutually exclusive with ``voice_id``.
            language: An ISO language code from :data:`~voicelabs_py.LANGUAGES`. Defaults to the
                profile's own language.
            idempotency_key: A key of your choosing (≤255 chars) that makes this write safe to
                retry. Replaying the SAME key with the SAME body returns the ORIGINAL result
                instead of generating again; replaying it with a DIFFERENT body raises
                :class:`~voicelabs_py.IdempotencyError`. This is the answer to "my connection
                dropped and I do not know whether the audio was made".

        Returns:
            A :class:`~voicelabs_py.SpeechGeneration` handle.

        Raises:
            VoiceLabsConfigError: ``text`` was empty, or both voice arguments were given.
            VoiceLabsAPIError: The API answered with a problem document.
        """
        prepared = self._prepare(
            "POST",
            "/v1/speech",
            body=speech_body(
                text=text, voice_id=voice_id, voice_name=voice_name, language=language
            ),
            idempotency_key=idempotency_key,
        )
        return SpeechGeneration.from_payload(self._send(prepared))

    def create_transcription(
        self,
        *,
        audio: bytes | str | os.PathLike[str] | BinaryIO | None = None,
        audio_base64: str | None = None,
        language: str | None = None,
        idempotency_key: str | None = None,
    ) -> Transcription:
        """Transcribe an audio clip. Returns the transcript synchronously.

        The public surface is JSON with base64 audio, NOT multipart. The SDK encodes for you and
        enforces the contract's 10 MiB base64 ceiling (about 7.5 MiB of raw audio) BEFORE the
        request, so an oversized clip fails instantly rather than after a long upload.

        Args:
            audio: Raw ``bytes``, a filesystem path, or an open binary file object.
            audio_base64: Already-encoded base64 text. Mutually exclusive with ``audio``.
            language: An ISO language code, or ``None`` to let the API detect it.
            idempotency_key: See :meth:`create_speech`.

        Returns:
            A :class:`~voicelabs_py.Transcription`.

        Raises:
            VoiceLabsConfigError: Neither or both audio arguments were given, the file could not
                be read, or the encoded clip exceeds the documented cap.
            VoiceLabsAPIError: The API answered with a problem document.
        """
        prepared = self._prepare(
            "POST",
            "/v1/transcriptions",
            body=transcription_body(audio=audio, audio_base64=audio_base64, language=language),
            idempotency_key=idempotency_key,
        )
        return Transcription.from_payload(self._send(prepared))

    # ── helpers over the six operations ──────────────────────────────────────

    def wait_for_generation(
        self,
        generation: Generation | SpeechGeneration | str,
        *,
        poll_interval: float = 2.0,
        timeout: float = 300.0,
        on_poll: Callable[[Generation], None] | None = None,
    ) -> Generation:
        """Poll a generation until it finishes, then return it.

        The timeout is a REAL deadline, not a poll count. It NEVER returns an unfinished
        generation: silently handing one back would push the failure into the caller's audio
        pipeline instead of surfacing it here.

        Args:
            generation: A generation, a speech handle, or a bare generation id.
            poll_interval: Seconds between polls. Defaults to 2.
            timeout: Seconds to wait before giving up. Defaults to 300.
            on_poll: Called with each intermediate result, for progress reporting.

        Returns:
            The completed :class:`~voicelabs_py.Generation`.

        Raises:
            GenerationFailedError: The generation reached a terminal ``failed`` state.
            GenerationTimeoutError: The deadline passed first. The generation has NOT been
                cancelled and may still complete — the error carries the last-seen generation.
        """
        generation_id = generation_id_of(generation)
        started = time.monotonic()

        while True:
            current = self.get_generation(generation_id)
            elapsed = time.monotonic() - started
            verdict = poll_verdict(current, elapsed, poll_interval, timeout)

            if verdict == "done":
                return current
            if verdict == "failed":
                raise GenerationFailedError(current)
            if on_poll is not None:
                on_poll(current)
            if verdict == "give_up":
                raise GenerationTimeoutError(current, elapsed)
            time.sleep(poll_interval)

    def generate_speech(
        self,
        *,
        text: str,
        voice_id: str | None = None,
        voice_name: str | None = None,
        language: str | None = None,
        idempotency_key: str | None = None,
        poll_interval: float = 2.0,
        timeout: float = 300.0,
        on_poll: Callable[[Generation], None] | None = None,
    ) -> Generation:
        """Generate speech and wait for the audio in one call — the shape most callers want.

        The idempotency key covers only the CREATE. If polling fails afterwards the audio still
        exists; retrying the whole call with the same key returns the original generation rather
        than making — and charging for — a second one.

        Returns:
            The completed :class:`~voicelabs_py.Generation`, ready for :meth:`download_audio`.

        Raises:
            GenerationFailedError: The generation failed.
            GenerationTimeoutError: The generation was still running at the deadline.
            VoiceLabsAPIError: The API answered with a problem document.
        """
        started = self.create_speech(
            text=text,
            voice_id=voice_id,
            voice_name=voice_name,
            language=language,
            idempotency_key=idempotency_key,
        )
        return self.wait_for_generation(
            started, poll_interval=poll_interval, timeout=timeout, on_poll=on_poll
        )

    def download_audio(self, generation: Generation | str) -> AudioFile:
        """Download a completed generation's audio bytes.

        Accepts a :class:`~voicelabs_py.Generation` or the ``audio_url`` string off one. The URL
        is a signed, time-limited CAPABILITY: the signature is the authorization, it carries no
        identity, and it is therefore fetched WITHOUT the API key — on a separate transport, so
        no default header can leak into it.

        Args:
            generation: A completed generation, or its ``audio_url``.

        Returns:
            An :class:`~voicelabs_py.AudioFile` carrying the bytes and the served content type.

        Raises:
            VoiceLabsConfigError: The generation has no ``audio_url`` yet.
            VoiceLabsAPIError: The download was refused — a 401 usually means the signature
                expired, so re-poll the generation for a fresh URL.
        """
        url = audio_url_of(generation)
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as bare:
                response = bare.get(url, headers=bare_headers())
        except httpx.HTTPError as exc:
            raise connection_error(exc, "GET", url) from exc
        return audio_outcome(response)

    def write_audio(self, generation: Generation | str, path: str | os.PathLike[str]) -> Path:
        """Download a generation's audio and write it to ``path``.

        Args:
            generation: A completed generation, or its ``audio_url``.
            path: Where to write. Parent directories are created for you.

        Returns:
            The path written.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.download_audio(generation).content)
        return destination

    # ── transport ────────────────────────────────────────────────────────────

    def _send(self, prepared: PreparedRequest) -> Any:
        """Issue a prepared request, retrying per :mod:`voicelabs_py._retry`, and decode it."""
        attempt = 0
        started = time.monotonic()

        while True:
            error: VoiceLabsError
            try:
                response = self._http.request(
                    prepared.method,
                    prepared.url,
                    params=prepared.params or None,
                    json=prepared.json,
                    headers=prepared.headers or None,
                )
            except httpx.HTTPError as exc:
                error = connection_error(exc, prepared.method, prepared.url)
            else:
                try:
                    return self._outcome(response)
                except VoiceLabsAPIError as api_error:
                    error = api_error

            delay = plan_retry(
                error,
                attempt=attempt,
                max_retries=self.max_retries,
                retry_safe=prepared.retry_safe,
                remaining_budget=self._remaining_budget(started),
            )
            if delay is None:
                raise error
            time.sleep(delay)
            attempt += 1

    def _remaining_budget(self, started: float) -> float | None:
        """Seconds left of the client's timeout, or ``None`` when there is no timeout."""
        if self.timeout is None:
            return None
        return self.timeout - (time.monotonic() - started)


def _path_segment(value: str) -> str:
    """Percent-encode one path segment so an id cannot reshape the URL.

    ``safe=""`` is deliberate: leaving ``/`` unescaped would let an id containing ``../`` be
    normalised away by the URL layer and address a different operation entirely.
    """
    return quote(value, safe="")
