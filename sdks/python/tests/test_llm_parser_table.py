"""G3 — the parser fixture table, shared byte-for-byte with the Rust suite.

One fixture directory per (provider, endpoint, shape) case under
`crates/wardex-protocol/tests/fixtures/llm/<case>/`: `request.json`, either
`response.json` or `stream.sse`, and `expect.json` with the assertions every
case must answer. The Rust tests include the same files with
`include_bytes!`, so a fixture edited for one language re-verifies the other.

A new provider or endpoint is one fixture set plus one row here — and the
directory scan below makes an unrowed fixture a FAILURE, so a case cannot be
added to the corpus without the table asserting it.

Not `wardex_sdk.testing.conformance`: that public suite is adapter-scoped
(tree shape, placement, uninstall). This table is the parser's own.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from wardex_sdk import LimitsConfig
from wardex_sdk._protocol import parse_llm_semantics

_REPO = pathlib.Path(__file__).resolve().parents[3]
_FIXTURES = _REPO / "crates" / "wardex-protocol" / "tests" / "fixtures" / "llm"

#: case directory -> (host, path). The dispatch inputs are part of the case.
_CASES: dict[str, tuple[str, str]] = {
    "openai_chat": ("api.openai.com", "/v1/chat/completions"),
    "openai_chat_tools": ("api.openai.com", "/v1/chat/completions"),
    "openai_chat_sse": ("api.openai.com", "/v1/chat/completions"),
    "openai_responses": ("api.openai.com", "/v1/responses"),
    "openai_responses_tools": ("api.openai.com", "/v1/responses"),
    "openai_responses_reasoning": ("api.openai.com", "/v1/responses"),
    "openai_responses_incomplete": ("api.openai.com", "/v1/responses"),
    "openai_responses_sse": ("api.openai.com", "/v1/responses"),
    "openai_responses_sse_unterminated": ("api.openai.com", "/v1/responses"),
    "openai_responses_sse_error": ("api.openai.com", "/v1/responses"),
    "openai_responses_background_queued": ("api.openai.com", "/v1/responses"),
    "openai_embeddings": ("api.openai.com", "/v1/embeddings"),
    "anthropic_messages": ("api.anthropic.com", "/v1/messages"),
    "anthropic_messages_sse": ("api.anthropic.com", "/v1/messages"),
}


def test_every_fixture_directory_has_a_row():
    on_disk = {p.name for p in _FIXTURES.iterdir() if p.is_dir()}
    assert on_disk == set(_CASES), (
        f"fixtures with no table row: {sorted(on_disk - set(_CASES))}; "
        f"rows with no fixture: {sorted(set(_CASES) - on_disk)}"
    )


def _load(case: str):
    d = _FIXTURES / case
    request = (d / "request.json").read_bytes()
    body = d / "response.json"
    response = body.read_bytes() if body.exists() else (d / "stream.sse").read_bytes()
    expect = json.loads((d / "expect.json").read_text())
    return request, response, expect


@pytest.mark.parametrize("case", sorted(_CASES))
def test_fixture_parses_to_its_expectation(case: str):
    host, path = _CASES[case]
    request, response, expect = _load(case)
    sem = parse_llm_semantics(host, path, request, response, LimitsConfig().to_native())
    assert sem is not None, f"{case}: no semantics at all"

    # The identical assertion set for every case:
    assert sem.provider == expect["provider"], case
    assert sem.operation == expect["operation"], case
    assert sem.api_type == expect.get("api_type"), case
    assert sem.request_model == expect.get("request_model"), case
    assert sem.response_model == expect.get("response_model"), case
    assert sem.response_id == expect.get("response_id"), case
    assert sem.input_tokens == expect.get("input_tokens"), case
    assert sem.output_tokens == expect.get("output_tokens"), case
    assert sem.finish_reasons == expect.get("finish_reasons"), case
    if expect["output_messages_present"]:
        assert sem.output_messages, f"{case}: output_messages empty"
    else:
        assert sem.output_messages is None, case

    # U1 as a SET EQUALITY, not membership: the mirrored paths are exactly
    # the provider's scalar leaves (no cap fires at these sizes).
    mirrored = {p: v for p, v in sem.usage_leaves}
    assert mirrored == expect["usage_leaves"], case
    assert sem.usage_dropped_count == 0, case

    # Optional per-case facts, asserted whenever the fixture declares them.
    for attr in (
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "reasoning_output_tokens",
        "response_status",
        "reasoning_level",
        "previous_response_id",
        "stream_terminated",
        "encoding_formats",
        "embedding_dimensions",
        "output_type",
    ):
        if attr in expect:
            got = getattr(sem, attr)
            if isinstance(got, tuple):
                got = list(got)
            assert got == expect[attr], f"{case}: {attr}"
    if "extras" in expect:
        # provider extras spelled with their wire keys (openai.*)
        from wardex_sdk._semantics import provider_extras

        emitted = dict(provider_extras(sem))
        for key, value in expect["extras"].items():
            assert emitted.get(key) == value, f"{case}: {key}"
    if "tool_call_ids" in expect:
        msgs = json.loads(sem.output_messages)
        ids = [
            part["id"] for msg in msgs for part in msg["parts"] if part.get("type") == "tool_call"
        ]
        assert ids == expect["tool_call_ids"], case
    if "output_text" in expect:
        msgs = json.loads(sem.output_messages)
        texts = [
            part["content"] for msg in msgs for part in msg["parts"] if part.get("type") == "text"
        ]
        assert expect["output_text"] in texts, case
    if "error_code" in expect:
        body = json.loads(bytes(sem.decoded_response))
        assert body["error"]["code"] == expect["error_code"], case
