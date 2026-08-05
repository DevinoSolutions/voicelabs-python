"""The asynchronous VoiceLabs API client.

A native :class:`httpx.AsyncClient`, NOT a thread-pool wrapper around the sync client: a wrapper
would burn a worker thread per in-flight generation poll, which is the one workload this API
makes callers do a lot of.

Every judgement call — the ``code``-based error hierarchy, draft-11 rate-limit parsing, the
credential-free capability download, the polling deadline — lives in ``_client.py`` and is shared
verbatim with :class:`~voicelabs_py.VoiceLabs`. Only the I/O call and the ``await`` differ, which
is the only way two clients stay behaviourally identical over time.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator, Callable
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
from .errors import GenerationFailedError, GenerationTimeoutError
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

__all__ = ["AsyncVoiceLabs"]


class AsyncVoiceLabs(BaseVoiceLabs):
    """An asynchronous client for the VoiceLabs public API.

    Takes the same arguments as :class:`~voicelabs_py.VoiceLabs` and returns the same models. Use
    it as ``async with AsyncVoiceLabs(api_key=…) as client:`` so the transport is closed for you;
    otherwise call :meth:`aclose`.

    Args:
        api_key: Your API key. Falls back to the ``VOICELABS_API_KEY`` environment variable.
        base_url: Override the API origin. Defaults to ``https://app.voicelabs.now``.
        timeout: Per-request timeout in seconds. Defaults to 60. Pass ``None`` to disable.
        max_retries: Extra attempts after the first for retryable failures. Defaults to 2.
        auth_style: ``"api_key"`` (default) or ``"bearer"``.
        on_rate_limit: Called after every response that advertised rate-limit headers.

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
        self._http = httpx.AsyncClient(
            timeout=self.timeout,
            headers=self._default_headers(),
            follow_redirects=True,
        )

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def aclose(self) -> None:
        """Close the underlying HTTP transport."""
        await self._http.aclose()

    async def __aenter__(self) -> AsyncVoiceLabs:
        """Enter an ``async with`` block; the transport is closed on exit."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Close the transport when the ``async with`` block ends."""
        await self.aclose()

    # ── reads (scope voice:read) ─────────────────────────────────────────────

    async def list_voices(self) -> VoiceList:
        """Every voice profile on the account: cloned voices and presets.

        Returns:
            A :class:`~voicelabs_py.VoiceList`. The endpoint is not paginated.
        """
        return VoiceList.from_payload(await self._send(self._prepare("GET", "/v1/voices")))

    async def list_captures(
        self, *, limit: int | None = None, offset: int | None = None
    ) -> CaptureList:
        """One page of captures with their transcripts, most-recent first.

        Args:
            limit: Rows per page, 1–200. Values above 200 are clamped to 200.
            offset: Rows to skip. Defaults to 0.

        Returns:
            A :class:`~voicelabs_py.CaptureList` echoing the ``limit`` and ``offset`` the SERVER
            applied.
        """
        prepared = self._prepare(
            "GET",
            "/v1/captures",
            query={"limit": clamp_page_size(limit), "offset": offset},
        )
        return CaptureList.from_payload(await self._send(prepared))

    async def iter_captures(
        self, *, page_size: int = 200, offset: int = 0
    ) -> AsyncIterator[Capture]:
        """Every capture on the account, paging automatically.

        Pages by the limit and offset the SERVER echoed rather than the ones requested — the API
        clamps ``limit`` to its own maximum, and a client that assumed its own number would skip
        rows on every page after the first.

        Args:
            page_size: Rows to request per page, clamped to the documented maximum of 200.
            offset: Where to start.

        Yields:
            Each :class:`~voicelabs_py.Capture`, once.
        """
        cursor = offset
        while True:
            page = await self.list_captures(limit=page_size, offset=cursor)
            for capture in page.data:
                yield capture
            if not page.data:
                return
            cursor = page.offset + len(page.data)
            if cursor >= page.total:
                return

    async def get_generation(self, generation_id: str) -> Generation:
        """One generation's current state.

        Args:
            generation_id: The generation to read.

        Returns:
            A :class:`~voicelabs_py.Generation`. Each poll re-mints a fresh, short-lived
            ``audio_url``.
        """
        prepared = self._prepare("GET", f"/v1/generations/{quote(generation_id, safe='')}")
        return Generation.from_payload(await self._send(prepared))

    # ── writes (scope voice:generate) ────────────────────────────────────────

    async def create_speech(
        self,
        *,
        text: str,
        voice_id: str | None = None,
        voice_name: str | None = None,
        language: str | None = None,
        idempotency_key: str | None = None,
    ) -> SpeechGeneration:
        """Start generating speech. Returns IMMEDIATELY with a handle — the audio is not ready.

        Args:
            text: What to say, 1–10000 characters.
            voice_id: A voice profile id. Mutually exclusive with ``voice_name``.
            voice_name: A voice profile name. Mutually exclusive with ``voice_id``.
            language: An ISO language code. Defaults to the profile's own language.
            idempotency_key: A key (≤255 chars) that makes this write safe to retry.

        Returns:
            A :class:`~voicelabs_py.SpeechGeneration` handle.

        Raises:
            VoiceLabsConfigError: ``text`` was empty, or both voice arguments were given.
        """
        prepared = self._prepare(
            "POST",
            "/v1/speech",
            body=speech_body(
                text=text, voice_id=voice_id, voice_name=voice_name, language=language
            ),
            idempotency_key=idempotency_key,
        )
        return SpeechGeneration.from_payload(await self._send(prepared))

    async def create_transcription(
        self,
        *,
        audio: bytes | str | os.PathLike[str] | BinaryIO | None = None,
        audio_base64: str | None = None,
        language: str | None = None,
        idempotency_key: str | None = None,
    ) -> Transcription:
        """Transcribe an audio clip. Returns the transcript synchronously.

        The public surface is JSON with base64 audio, NOT multipart. The SDK encodes and
        size-checks before the request.

        Args:
            audio: Raw ``bytes``, a filesystem path, or an open binary file object.
            audio_base64: Already-encoded base64 text. Mutually exclusive with ``audio``.
            language: An ISO language code, or ``None`` to let the API detect it.
            idempotency_key: A key (≤255 chars) that makes this write safe to retry.

        Returns:
            A :class:`~voicelabs_py.Transcription`.

        Raises:
            VoiceLabsConfigError: Bad arguments, or the clip exceeds the documented cap.
        """
        prepared = self._prepare(
            "POST",
            "/v1/transcriptions",
            body=transcription_body(audio=audio, audio_base64=audio_base64, language=language),
            idempotency_key=idempotency_key,
        )
        return Transcription.from_payload(await self._send(prepared))

    # ── helpers over the six operations ──────────────────────────────────────

    async def wait_for_generation(
        self,
        generation: Generation | SpeechGeneration | str,
        *,
        poll_interval: float = 2.0,
        timeout: float = 300.0,
        on_poll: Callable[[Generation], None] | None = None,
    ) -> Generation:
        """Poll a generation until it finishes, then return it.

        The timeout is a REAL deadline, not a poll count, and this NEVER returns an unfinished
        generation.

        Args:
            generation: A generation, a speech handle, or a bare generation id.
            poll_interval: Seconds between polls. Defaults to 2.
            timeout: Seconds to wait before giving up. Defaults to 300.
            on_poll: Called with each intermediate result.

        Returns:
            The completed :class:`~voicelabs_py.Generation`.

        Raises:
            GenerationFailedError: The generation reached a terminal ``failed`` state.
            GenerationTimeoutError: The deadline passed first. The generation has NOT been
                cancelled and may still complete.
        """
        generation_id = generation_id_of(generation)
        started = time.monotonic()

        while True:
            current = await self.get_generation(generation_id)
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
            await asyncio.sleep(poll_interval)

    async def generate_speech(
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
        """Generate speech and wait for the audio in one call.

        The idempotency key covers only the CREATE, so retrying the whole call with the same key
        returns the original generation rather than charging for a second one.

        Returns:
            The completed :class:`~voicelabs_py.Generation`.

        Raises:
            GenerationFailedError: The generation failed.
            GenerationTimeoutError: The generation was still running at the deadline.
        """
        started = await self.create_speech(
            text=text,
            voice_id=voice_id,
            voice_name=voice_name,
            language=language,
            idempotency_key=idempotency_key,
        )
        return await self.wait_for_generation(
            started, poll_interval=poll_interval, timeout=timeout, on_poll=on_poll
        )

    async def download_audio(self, generation: Generation | str) -> AudioFile:
        """Download a completed generation's audio bytes.

        The ``audio_url`` is a signed, time-limited CAPABILITY: the signature is the
        authorization, so it is fetched WITHOUT the API key, on a separate transport so no
        default header can leak into it.

        Args:
            generation: A completed generation, or its ``audio_url``.

        Returns:
            An :class:`~voicelabs_py.AudioFile`.

        Raises:
            VoiceLabsConfigError: The generation has no ``audio_url`` yet.
            VoiceLabsAPIError: The download was refused — a 401 usually means the signature
                expired, so re-poll for a fresh URL.
        """
        url = audio_url_of(generation)
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as bare:
                response = await bare.get(url, headers=bare_headers())
        except httpx.HTTPError as exc:
            raise connection_error(exc, "GET", url) from exc
        return audio_outcome(response)

    async def write_audio(self, generation: Generation | str, path: str | os.PathLike[str]) -> Path:
        """Download a generation's audio and write it to ``path``.

        Args:
            generation: A completed generation, or its ``audio_url``.
            path: Where to write. Parent directories are created for you.

        Returns:
            The path written.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        audio = await self.download_audio(generation)
        destination.write_bytes(audio.content)
        return destination

    # ── transport ────────────────────────────────────────────────────────────

    async def _send(self, prepared: PreparedRequest) -> Any:
        """Issue one prepared request and return its decoded body."""
        try:
            response = await self._http.request(
                prepared.method,
                prepared.url,
                params=prepared.params or None,
                json=prepared.json,
                headers=prepared.headers or None,
            )
        except httpx.HTTPError as exc:
            raise connection_error(exc, prepared.method, prepared.url) from exc
        return self._outcome(response)
