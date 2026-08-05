"""The resource shapes of the VoiceLabs public API.

FIELD NAMES ARE THE WIRE NAMES, snake_case and all. There is no camelCase translation layer in
this SDK, and that is a decision rather than an omission: a mapping layer is a second declaration
of the contract that can drift from the first, and when it drifts the failure is a silently-absent
field rather than a type error. What you read in the OpenAPI document, in a ``curl`` response, and
in these classes is one set of names.

Every model is built through :meth:`from_payload`, which reads the keys it knows and IGNORES the
ones it does not. The API may add fields; a strict constructor would turn an additive server
change into a customer outage. (``additionalProperties: false`` in the contract binds the
*server's* output, not this SDK's forward-compatibility obligation.)
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

__all__ = [
    "AudioFile",
    "Capture",
    "CaptureList",
    "GENERATION_STATUS",
    "Generation",
    "LANGUAGES",
    "MAX_AUDIO_BASE64_CHARS",
    "MAX_CAPTURES_PAGE_SIZE",
    "SpeechGeneration",
    "Transcription",
    "Voice",
    "VoiceList",
]

#: The statuses the API is known to use. NOT exhaustive by contract: ``Generation.status`` is a
#: plain ``str``, never an ``enum.Enum``, because narrowing it would make a future server status
#: raise inside customer code for a response the API considers perfectly valid.
GENERATION_STATUS = SimpleNamespace(
    generating="generating",
    completed="completed",
    failed="failed",
)

#: The ISO language codes ``/v1/speech`` and ``/v1/transcriptions`` accept.
LANGUAGES: tuple[str, ...] = (
    "zh",
    "en",
    "ja",
    "ko",
    "de",
    "fr",
    "ru",
    "pt",
    "es",
    "it",
    "he",
    "ar",
    "da",
    "el",
    "fi",
    "hi",
    "ms",
    "nl",
    "no",
    "pl",
    "sv",
    "sw",
    "tr",
)

#: The contract's ceiling on the base64 TEXT of a transcription upload — roughly 7.5 MiB of raw
#: audio, since base64 costs about a third. Checked client-side so an oversized clip fails before
#: it is uploaded rather than after.
MAX_AUDIO_BASE64_CHARS = 10 * 1024 * 1024

#: The largest page ``GET /v1/captures`` will serve, whatever a caller asks for.
MAX_CAPTURES_PAGE_SIZE = 200


def _str(payload: dict[str, Any], key: str, default: str = "") -> str:
    value = payload.get(key)
    return value if isinstance(value, str) else default


def _opt_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _int(payload: dict[str, Any], key: str, default: int = 0) -> int:
    value = payload.get(key)
    return (
        int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default
    )


def _opt_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _bool(payload: dict[str, Any], key: str) -> bool:
    return bool(payload.get(key))


@dataclass(frozen=True, slots=True)
class Voice:
    """A voice profile on the account: a cloned voice or a preset."""

    id: str
    name: str
    description: str | None
    language: str
    voice_type: str
    default_engine: str | None
    has_personality: bool
    generation_count: int
    sample_count: int
    created_at: str
    updated_at: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Voice:
        """Build a voice from a wire payload, ignoring fields this SDK does not know."""
        return cls(
            id=_str(payload, "id"),
            name=_str(payload, "name"),
            description=_opt_str(payload, "description"),
            language=_str(payload, "language"),
            voice_type=_str(payload, "voice_type"),
            default_engine=_opt_str(payload, "default_engine"),
            has_personality=_bool(payload, "has_personality"),
            generation_count=_int(payload, "generation_count"),
            sample_count=_int(payload, "sample_count"),
            created_at=_str(payload, "created_at"),
            updated_at=_str(payload, "updated_at"),
        )


@dataclass(frozen=True, slots=True)
class VoiceList:
    """Every voice on the account. Unpaginated by design — ``total`` equals ``len(data)``."""

    data: tuple[Voice, ...]
    total: int

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> VoiceList:
        """Build a voice list from a wire payload."""
        rows = payload.get("data")
        voices = (
            tuple(Voice.from_payload(row) for row in rows if isinstance(row, dict))
            if isinstance(rows, list)
            else ()
        )
        return cls(data=voices, total=_int(payload, "total", len(voices)))


@dataclass(frozen=True, slots=True)
class Capture:
    """A captured audio clip and its transcript."""

    id: str
    source: str
    language: str | None
    duration_ms: int | None
    transcript_raw: str
    transcript_refined: str | None
    created_at: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Capture:
        """Build a capture from a wire payload, ignoring fields this SDK does not know."""
        return cls(
            id=_str(payload, "id"),
            source=_str(payload, "source"),
            language=_opt_str(payload, "language"),
            duration_ms=_opt_int(payload, "duration_ms"),
            transcript_raw=_str(payload, "transcript_raw"),
            transcript_refined=_opt_str(payload, "transcript_refined"),
            created_at=_str(payload, "created_at"),
        )


@dataclass(frozen=True, slots=True)
class CaptureList:
    """One page of captures, echoing back the ``limit`` and ``offset`` the SERVER applied.

    The server clamps ``limit`` to its own maximum, so a pager that trusted the number it asked
    for would skip rows on every page after the first. Page with ``offset + len(data)``.
    """

    data: tuple[Capture, ...]
    total: int
    limit: int
    offset: int

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> CaptureList:
        """Build a capture page from a wire payload."""
        rows = payload.get("data")
        captures = (
            tuple(Capture.from_payload(row) for row in rows if isinstance(row, dict))
            if isinstance(rows, list)
            else ()
        )
        return cls(
            data=captures,
            total=_int(payload, "total", len(captures)),
            limit=_int(payload, "limit", len(captures)),
            offset=_int(payload, "offset"),
        )


@dataclass(frozen=True, slots=True)
class Generation:
    """A generation, as returned while polling.

    Attributes:
        id: The generation id, stable across polls.
        status: A plain ``str``, compared against :data:`GENERATION_STATUS`. Deliberately not an
            enum — see the module docstring.
        profile: The voice profile the audio was generated in.
        has_audio: Whether finished audio exists.
        audio_url: A signed, time-limited URL for the finished audio, or ``None`` while it is
            still running. The signature IS the authorization: this URL carries no identity of
            its own and must be fetched WITHOUT the API key. Re-polling mints a fresh URL rather
            than reviving an expired one, so treat it as short-lived and follow it promptly.
        error: The engine's failure reason, when there is one.
    """

    id: str
    status: str
    profile: str
    has_audio: bool
    audio_url: str | None
    error: str | None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Generation:
        """Build a generation from a wire payload, ignoring fields this SDK does not know."""
        return cls(
            id=_str(payload, "id"),
            status=_str(payload, "status"),
            profile=_str(payload, "profile"),
            has_audio=_bool(payload, "has_audio"),
            audio_url=_opt_str(payload, "audio_url"),
            error=_opt_str(payload, "error"),
        )


@dataclass(frozen=True, slots=True)
class SpeechGeneration:
    """The handle ``POST /v1/speech`` returns immediately; the audio is not ready yet."""

    id: str
    status: str
    voice: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> SpeechGeneration:
        """Build a speech handle from a wire payload."""
        return cls(
            id=_str(payload, "id"),
            status=_str(payload, "status"),
            voice=_str(payload, "voice"),
        )


@dataclass(frozen=True, slots=True)
class Transcription:
    """A finished transcription, returned synchronously."""

    id: str
    text: str
    language: str | None
    duration_ms: int | None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Transcription:
        """Build a transcription from a wire payload."""
        return cls(
            id=_str(payload, "id"),
            text=_str(payload, "text"),
            language=_opt_str(payload, "language"),
            duration_ms=_opt_int(payload, "duration_ms"),
        )


@dataclass(frozen=True, slots=True)
class AudioFile:
    """Downloaded audio bytes plus the content type the server served them as.

    The content type is carried so a caller can name the file correctly without guessing at the
    extension from the URL.
    """

    content: bytes
    content_type: str

    def __len__(self) -> int:
        """The number of audio bytes."""
        return len(self.content)
