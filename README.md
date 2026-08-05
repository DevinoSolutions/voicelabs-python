# VoiceLabs Python SDK (voicelabs-py)

The official **Python client** for the [VoiceLabs](https://voicelabs.now) API — text to speech
and transcription over HTTP, against `https://app.voicelabs.now/v1`.

Sync and async, fully typed, hand-written on `httpx` with **one runtime dependency**. It is the
Python twin of the official TypeScript SDK and speaks the same six operations, the same RFC 9457
error codes, and the same draft-11 rate-limit headers.

## Install

```bash
pip install voicelabs-py
```

Requires Python 3.10+. The import name is `voicelabs_py`.

## Authentication

Create a key at [voicelabs.now/connections](https://voicelabs.now/connections) → **API keys**,
choosing its scopes at creation. The secret is shown once; revoking a key takes effect on its
very next request.

Two scopes exist, and a key does exactly what its scopes allow:

| Scope | Grants |
| --- | --- |
| `voice:read` | `list_voices`, `list_captures`, `get_generation` |
| `voice:generate` | `create_speech`, `create_transcription` — these spend the account's metered allowance |

```python
from voicelabs_py import VoiceLabs

client = VoiceLabs(api_key="vl_sandbox_…")
```

The key is read from `VOICELABS_API_KEY` when `api_key` is omitted. An explicit argument always
wins. With neither, the constructor raises `VoiceLabsConfigError` before any request is made.

> Every key issued today begins with `vl_sandbox_`. That is an honest marker, not a placeholder
> awaiting a `vl_live_` twin, and the SDK deliberately does **not** validate the prefix — a
> prefix check would break every caller the day a second prefix ships.

## Text to speech

`POST /v1/speech` is asynchronous: it returns a handle immediately and the audio is not ready
yet. `generate_speech` does the create-then-poll handshake for you.

```python
from voicelabs_py import VoiceLabs

with VoiceLabs() as client:
    generation = client.generate_speech(
        text="Hello from the VoiceLabs Python SDK.",
        voice_name="Narrator",  # or voice_id=…, not both
        language="en",  # optional; defaults to the profile's language
    )
    client.write_audio(generation, "hello.mp3")
```

Doing it by hand, when you want to report progress:

```python
handle = client.create_speech(text="Hello.", voice_name="Narrator")
generation = client.wait_for_generation(
    handle,
    poll_interval=2.0,
    timeout=300.0,
    on_poll=lambda g: print(g.status),
)
audio = client.download_audio(generation)
print(len(audio.content), audio.content_type)  # 40213 audio/mpeg
```

Async is the same call sites with `await`:

```python
import asyncio
from voicelabs_py import AsyncVoiceLabs


async def main():
    async with AsyncVoiceLabs() as client:
        generation = await client.generate_speech(text="Hello.", voice_name="Narrator")
        await client.write_audio(generation, "hello.mp3")


asyncio.run(main())
```

`AsyncVoiceLabs` is a native `httpx.AsyncClient`, not a thread-pool wrapper around the sync
client.

> `generation.audio_url` is a **signed, time-limited capability URL**. The signature *is* the
> authorization, so `download_audio` fetches it with **no API key**, on a separate transport.
> Sending your key to a capability URL would leak a long-lived credential for a request that
> does not need it. Each poll mints a fresh URL rather than reviving an expired one — follow it
> promptly, don't store it.

## Transcription

The public transcription surface is **JSON with base64 audio, not multipart**. Pass raw bytes, a
path, or an open binary file; the SDK encodes and size-checks for you.

```python
transcription = client.create_transcription(audio="meeting.wav", language="en")
print(transcription.text, transcription.duration_ms)
```

```python
with open("clip.ogg", "rb") as handle:
    client.create_transcription(audio=handle)

client.create_transcription(audio=b"\x00\x01…")          # raw bytes
client.create_transcription(audio_base64="UklGRi…")      # already encoded
```

> The contract's cap is **10 MiB of base64 text** — roughly 7.5 MiB of raw audio. The SDK
> enforces it *before* the request and raises `VoiceLabsConfigError` naming the actual size, so
> an oversized clip fails instantly instead of after a long upload.

## Listing

```python
voices = client.list_voices()  # not paginated: everything, plus a total
for voice in voices.data:
    print(voice.id, voice.name, voice.language)

page = client.list_captures(limit=50, offset=0)
for capture in client.iter_captures():  # walks every page for you
    print(capture.id, capture.transcript_refined or capture.transcript_raw)
```

`/v1/captures` is the only paginated operation; `limit` is clamped to the documented maximum of
**200**. `iter_captures` pages by the `limit` and `offset` the *server* echoed, not the ones
requested — the API clamps, and a pager that trusted its own numbers would skip rows.

## Errors

Every non-2xx is an RFC 9457 problem document, and **the SDK branches on its `code`, never on the
HTTP status**. That is not a stylistic preference: this API deliberately returns two different
problems under one status, twice.

| Status | `code` | Meaning | Exception | Retry? |
| --- | --- | --- | --- | --- |
| 429 | `rate_limit_exceeded` | You are calling too fast. Waiting fixes it. | `RateLimitError` | yes, automatically |
| 429 | `quota_exhausted` | The account's audio allowance is spent. | `QuotaExhaustedError` | **never** — futile until the period rolls over |
| 403 | `insufficient_scope` | Valid credential, missing scope. Mint a wider key. | `InsufficientScopeError` | no |
| 403 | `feature_not_enabled` | The plan does not include the capability. | `FeatureNotEnabledError` | no |

A client that branches on `response.status_code` gets both pairs wrong, and the second mistake is
expensive: retrying `quota_exhausted` on a backoff loop burns your rate budget forever without
ever succeeding.

The full hierarchy:

```
VoiceLabsError                       catch-all
├─ VoiceLabsConfigError              missing key, oversized audio, bad arguments (no request made)
├─ VoiceLabsConnectionError          no HTTP response at all; the cause is on __cause__
├─ GenerationFailedError             terminal status == "failed"   (.generation, .reason)
├─ GenerationTimeoutError            the polling deadline passed   (.generation, .waited_seconds)
└─ VoiceLabsAPIError                 any problem document  (.code .status .detail .problem .rate_limit)
   ├─ InvalidRequestError            invalid_request                            400
   ├─ AuthenticationError            unauthorized                               401
   ├─ InsufficientScopeError         insufficient_scope    (.required_scope)    403
   ├─ FeatureNotEnabledError         feature_not_enabled   (.settings_url)      403
   ├─ NotFoundError                  not_found                                  404
   ├─ MethodNotAllowedError          method_not_allowed                         405
   ├─ IdempotencyError               idempotency_key_in_progress | _reused       409 / 422
   ├─ RateLimitError                 rate_limit_exceeded   (.retry_after)       429
   ├─ QuotaExhaustedError            quota_exhausted       (.settings_url)      429
   └─ ServerError                    upstream_error | internal_error |
                                     service_unavailable                        5xx
```

```python
from voicelabs_py import QuotaExhaustedError, RateLimitError, VoiceLabsError

try:
    client.generate_speech(text="…")
except QuotaExhaustedError as exc:
    alert_the_owner(exc.settings_url)  # do NOT retry
except RateLimitError as exc:
    sleep(exc.retry_after or 30)  # this one will succeed later
except VoiceLabsError as exc:
    log(exc)
```

`isinstance` and `.code` always agree. `QuotaExhaustedError` is **not** a subclass of
`RateLimitError`, on purpose. An unrecognised `code` from a newer API version degrades to a plain
`VoiceLabsAPIError` with `.code` set to the raw string rather than crashing or being mistaken for
something familiar. A non-2xx whose body is not a problem document (a proxy's HTML error page, an
empty 502) still raises a typed error, with `.problem is None`.

`VoiceLabsConnectionError` is distinct from every API error because **nothing on the server is
known to have happened** — a write may or may not have executed, which is exactly when an
idempotency key earns its keep.

## Rate limits

The API emits [IETF draft-11](https://datatracker.ietf.org/doc/draft-ietf-httpapi-ratelimit-headers/)
structured fields on both `200` and `429` responses of the API-key lane — a different wire format
from the widely copied `X-RateLimit-*` triple.

```
RateLimit-Policy: "default";q=600;w=3600
RateLimit:        "default";r=599;t=2718
```

```python
client.list_voices()
info = client.rate_limit
print(info.limit, info.window_seconds, info.remaining, info.reset_seconds)
# 600 3600 599 2718

client = VoiceLabs(on_rate_limit=lambda info: gauge.set(info.remaining))
```

> **Absent means unknown, never zero.** `client.rate_limit` is `None` when a response carried no
> budget to report — which is the case on the OAuth lane and for keys with limiting switched off.
> A non-numeric parameter yields `None` rather than a NaN that would poison every comparison
> written against it.

## Idempotency

Both writes accept an `idempotency_key` (≤255 characters).

```python
client.create_speech(text="Hello.", idempotency_key="order-42")
```

Replaying the **same key with the same body** returns the original response instead of doing the
work — and charging for it — a second time. Replaying it with a **different body** raises
`IdempotencyError` with `code == "idempotency_key_reused"`. A replay while the first request is
still in flight raises `IdempotencyError` with `code == "idempotency_key_in_progress"`.

The key also decides retry behaviour:

| Situation | Retried? |
| --- | --- |
| `GET` fails with a connection error | yes |
| `POST` **with** `idempotency_key` fails with a connection error | yes — the server deduplicates it |
| `POST` **without** `idempotency_key` fails with a connection error | **no** — the write may have succeeded, and a blind replay can double-charge |

## Retries

`max_retries=2` by default (three attempts total). Retried: `rate_limit_exceeded`,
`upstream_error`, `internal_error`, `service_unavailable`, and connection failures on
retry-safe requests. Never retried: `quota_exhausted`, any other 4xx, or an unkeyed write.

Delays come from `Retry-After` first, then the draft-11 `t` value, then exponential backoff with
full jitter (base 0.5 s, cap 20 s). A wait that would outlast the client's `timeout` budget
raises instead of silently sleeping past it. `max_retries=0` disables retrying.

## Client options

```python
VoiceLabs(
    api_key=None,  # else $VOICELABS_API_KEY, else VoiceLabsConfigError
    base_url=None,  # else https://app.voicelabs.now (trailing slashes stripped)
    timeout=60.0,  # seconds; None disables
    max_retries=2,  # extra attempts after the first
    auth_style="api_key",  # "api_key" -> x-api-key; "bearer" -> Authorization: Bearer
    on_rate_limit=None,  # called with RateLimitInfo when a response advertises one
)
```

`AsyncVoiceLabs` takes exactly the same arguments and exposes the same methods, with `aclose()`
in place of `close()`. Both support context-manager use.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

ruff check . && ruff format --check . && ty check
pytest tests --ignore=tests/e2e          # offline, respx-mocked
VOICELABS_API_KEY=vl_sandbox_… pytest tests/e2e -v   # hits prod, spends allowance
```

`openapi.json` at the repo root is a committed snapshot of
[`app.voicelabs.now/openapi.json`](https://app.voicelabs.now/openapi.json).
`tests/test_contract_coverage.py` asserts the SDK covers every operation, error code, and enum in
it, so a server change fails CI instead of a customer's build. The e2e suite re-fetches the live
document and diffs it against the snapshot.

The e2e suite skips **loudly** without `VOICELABS_API_KEY` — a skipped e2e is not a green e2e.

## Links

- API reference: <https://app.voicelabs.now/docs/api>
- Developer portal: <https://voicelabs.now/developers>
- API keys: <https://voicelabs.now/connections>

## License

MIT © 2026 Devino Solutions Inc.
