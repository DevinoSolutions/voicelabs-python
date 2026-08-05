"""The synchronous VoiceLabs API client.

Hand-written, not generated. Six operations do not need a code generator, and a generated client
could not make the judgement calls that are the whole value here: telling ``quota_exhausted``
apart from ``rate_limit_exceeded`` under a shared 429, fetching the audio capability URL WITHOUT
the API key, or refusing to poll a generation forever.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import quote

import httpx

from ._client import BaseVoiceLabs, PreparedRequest, clamp_page_size, connection_error
from .models import Capture, CaptureList, Generation, VoiceList
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

    # ── transport ────────────────────────────────────────────────────────────

    def _send(self, prepared: PreparedRequest) -> Any:
        """Issue one prepared request and return its decoded body."""
        try:
            response = self._http.request(
                prepared.method,
                prepared.url,
                params=prepared.params or None,
                json=prepared.json,
                headers=prepared.headers or None,
            )
        except httpx.HTTPError as exc:
            raise connection_error(exc, prepared.method, prepared.url) from exc
        return self._outcome(response)


def _path_segment(value: str) -> str:
    """Percent-encode one path segment so an id cannot reshape the URL.

    ``safe=""`` is deliberate: leaving ``/`` unescaped would let an id containing ``../`` be
    normalised away by the URL layer and address a different operation entirely.
    """
    return quote(value, safe="")
