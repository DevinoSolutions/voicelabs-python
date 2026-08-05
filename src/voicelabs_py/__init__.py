"""Official Python SDK for the VoiceLabs API.

Text to speech and transcription over HTTP, against ``https://app.voicelabs.now/v1``.

    from voicelabs_py import VoiceLabs

    with VoiceLabs(api_key="vl_sandbox_…") as client:
        generation = client.generate_speech(text="Hello from Python.")
        client.write_audio(generation, "hello.wav")
"""

from __future__ import annotations

from ._client import DEFAULT_BASE_URL, ENV_API_KEY
from ._version import __version__
from .client import VoiceLabs
from .errors import (
    ERROR_CODES,
    AuthenticationError,
    FeatureNotEnabledError,
    GenerationFailedError,
    GenerationTimeoutError,
    IdempotencyError,
    InsufficientScopeError,
    InvalidRequestError,
    MethodNotAllowedError,
    NotFoundError,
    ProblemDocument,
    QuotaExhaustedError,
    RateLimitError,
    ServerError,
    VoiceLabsAPIError,
    VoiceLabsConfigError,
    VoiceLabsConnectionError,
    VoiceLabsError,
)
from .models import (
    GENERATION_STATUS,
    LANGUAGES,
    MAX_AUDIO_BASE64_CHARS,
    MAX_CAPTURES_PAGE_SIZE,
    AudioFile,
    Capture,
    CaptureList,
    Generation,
    SpeechGeneration,
    Transcription,
    Voice,
    VoiceList,
)
from .rate_limit import RateLimitInfo

__all__ = [
    "DEFAULT_BASE_URL",
    "ENV_API_KEY",
    "ERROR_CODES",
    "GENERATION_STATUS",
    "LANGUAGES",
    "MAX_AUDIO_BASE64_CHARS",
    "MAX_CAPTURES_PAGE_SIZE",
    "AudioFile",
    "AuthenticationError",
    "Capture",
    "CaptureList",
    "FeatureNotEnabledError",
    "Generation",
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
    "RateLimitInfo",
    "ServerError",
    "SpeechGeneration",
    "Transcription",
    "Voice",
    "VoiceLabs",
    "VoiceLabsAPIError",
    "VoiceLabsConfigError",
    "VoiceLabsConnectionError",
    "VoiceLabsError",
    "VoiceList",
    "__version__",
]
