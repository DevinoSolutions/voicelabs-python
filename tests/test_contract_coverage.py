"""The SDK checked against the published contract, in place of code generation.

``openapi.json`` at the repo root is a committed snapshot of
https://app.voicelabs.now/openapi.json. These tests are the drift protection a code generator
would have given: if the server grows an operation, an error code, or a language, unit CI goes
red until somebody decides what the SDK should do about it. The snapshot itself is checked
against production by the e2e suite, so unit CI stays offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voicelabs_py import (
    ERROR_CODES,
    LANGUAGES,
    MAX_CAPTURES_PAGE_SIZE,
    AsyncVoiceLabs,
    VoiceLabs,
)
from voicelabs_py._client import API_KEY_HEADER, bare_headers
from voicelabs_py.errors import _ERROR_CLASSES, VoiceLabsAPIError

CONTRACT_PATH = Path(__file__).resolve().parent.parent / "openapi.json"

#: Every published operation, and the SDK methods that cover it. Adding an operation to the API
#: fails this test until someone decides whether to implement it — which is the point.
OPERATION_MAP: dict[str, tuple[str, ...]] = {
    "listVoices": ("list_voices",),
    "listCaptures": ("list_captures", "iter_captures"),
    "getGeneration": ("get_generation", "wait_for_generation"),
    "createSpeech": ("create_speech", "generate_speech"),
    "createTranscription": ("create_transcription",),
    "getGenerationAudio": ("download_audio", "write_audio"),
}


@pytest.fixture(scope="module")
def contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def operations(contract: dict) -> dict[str, dict]:
    return {
        operation["operationId"]: operation
        for item in contract["paths"].values()
        for method, operation in item.items()
        if method in {"get", "post", "put", "patch", "delete"}
    }


def test_the_vendored_contract_is_the_document_this_sdk_was_written_against(contract):
    assert contract["openapi"] == "3.1.1"
    assert contract["info"]["version"] == "1.0.0"
    assert [server["url"] for server in contract["servers"]] == ["https://app.voicelabs.now"]


def test_the_sdk_implements_every_operation_in_the_published_contract(contract):
    published = set(operations(contract))

    assert published == set(OPERATION_MAP), (
        "The published contract and the SDK's operation map disagree. Re-vendor openapi.json, "
        "then decide whether each new operation gets an SDK method."
    )

    for operation_id, methods in OPERATION_MAP.items():
        for method in methods:
            assert callable(getattr(VoiceLabs, method, None)), f"{operation_id} -> {method}"
            assert callable(getattr(AsyncVoiceLabs, method, None)), f"{operation_id} -> {method}"


def test_every_error_code_in_the_contract_has_a_python_exception_class(contract):
    published = set(contract["components"]["schemas"]["Problem"]["properties"]["code"]["enum"])

    assert published <= set(ERROR_CODES)
    for code in published:
        error_class = _ERROR_CLASSES.get(code)
        assert error_class is not None, code
        assert issubclass(error_class, VoiceLabsAPIError), code


def test_the_two_429_codes_and_the_two_403_codes_map_to_four_distinct_classes(contract):
    published = set(contract["components"]["schemas"]["Problem"]["properties"]["code"]["enum"])
    pairs = {"rate_limit_exceeded", "quota_exhausted", "insufficient_scope", "feature_not_enabled"}

    assert pairs <= published
    classes = {_ERROR_CLASSES[code] for code in pairs}
    assert len(classes) == 4


def test_the_documented_language_enum_matches_the_sdk_constant(contract):
    body = contract["paths"]["/v1/speech"]["post"]["requestBody"]
    published = body["content"]["application/json"]["schema"]["properties"]["language"]["enum"]

    assert list(LANGUAGES) == published
    assert len(LANGUAGES) == 23


def test_the_transcription_language_enum_is_the_same_list(contract):
    body = contract["paths"]["/v1/transcriptions"]["post"]["requestBody"]
    published = body["content"]["application/json"]["schema"]["properties"]["language"]["enum"]

    assert list(LANGUAGES) == published


def test_the_pagination_maximum_matches_the_contract(contract):
    parameters = contract["paths"]["/v1/captures"]["get"]["parameters"]
    limit = next(p for p in parameters if p["name"] == "limit")

    assert limit["schema"]["maximum"] == MAX_CAPTURES_PAGE_SIZE
    assert limit["schema"]["minimum"] == 1
    assert limit["schema"]["default"] == 50


def test_the_voices_operation_is_unpaginated_so_the_sdk_ships_no_paginator_for_it(contract):
    assert contract["paths"]["/v1/voices"]["get"].get("parameters") in (None, [])
    assert not hasattr(VoiceLabs, "iter_voices")


def test_transcription_takes_base64_json_and_not_multipart(contract):
    body = contract["paths"]["/v1/transcriptions"]["post"]["requestBody"]

    assert set(body["content"]) == {"application/json"}
    audio = body["content"]["application/json"]["schema"]["properties"]["audio"]
    assert audio["contentEncoding"] == "base64"


def test_the_audio_operation_takes_no_credential_and_neither_does_the_sdk(contract):
    audio = contract["paths"]["/v1/audio/{generationId}"]["get"]

    # `security: []` — the signature in the URL is the whole authorization.
    assert audio["security"] == []
    headers = {name.lower() for name in bare_headers()}
    assert API_KEY_HEADER not in headers
    assert "authorization" not in headers


def test_every_other_operation_accepts_the_api_key_header_lane(contract):
    for operation_id, operation in operations(contract).items():
        if operation_id == "getGenerationAudio":
            continue
        schemes = {scheme for entry in operation["security"] for scheme in entry}
        assert "apiKeyHeader" in schemes, operation_id


def test_the_api_key_scheme_is_the_header_this_sdk_sends(contract):
    scheme = contract["components"]["securitySchemes"]["apiKeyHeader"]

    assert scheme["type"] == "apiKey"
    assert scheme["in"] == "header"
    assert scheme["name"] == API_KEY_HEADER


def test_both_writes_accept_the_idempotency_key_header(contract):
    for path in ("/v1/speech", "/v1/transcriptions"):
        names = {p["name"] for p in contract["paths"][path]["post"]["parameters"]}
        assert "Idempotency-Key" in names
