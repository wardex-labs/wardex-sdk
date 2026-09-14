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
    "openai_responses_compact": ("api.openai.com", "/v1/responses/compact"),
    "openai_embeddings": ("api.openai.com", "/v1/embeddings"),
    "anthropic_messages": ("api.anthropic.com", "/v1/messages"),
    "anthropic_messages_sse": ("api.anthropic.com", "/v1/messages"),
}


#: The negative corpus lives one level down and is NOT a `_CASES` row: its
#: cases carry their dispatch inputs and their truth in `truth.json` instead of
#: an `expect.json`, because what they assert is attribution (which provider,
#: and whether the label was a guess), not the full extraction set above.
#: `test_every_negative_directory_has_truth` holds that corpus to its own layout.
_NEGATIVES = _FIXTURES / "negatives"


def test_every_fixture_directory_has_a_row():
    on_disk = {p.name for p in _FIXTURES.iterdir() if p.is_dir() and p != _NEGATIVES}
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


# ---------------------------------------------------------------------------
# Provider attribution: the misattribution rate over positives + negatives
# ---------------------------------------------------------------------------

#: The ceiling every place the SDK infers rather than observes is held to. With a
#: corpus this small it is a statement about ZERO: 0.1 % of 18 cases is less
#: than one case, so a single misattribution fails the gate, and the failure
#: message says "N of M" so nobody reads the percentage as more resolution
#: than the corpus has.
_MISATTRIBUTION_CEILING = 0.001

#: Hosts that ARE the provider's own. A label read off one of these is proven;
#: every other label is a guess and must say so.
_OFFICIAL_HOSTS = frozenset({"api.openai.com", "api.anthropic.com"})


def _negative_cases() -> dict[str, dict]:
    return {
        d.name: json.loads((d / "truth.json").read_text())
        for d in sorted(_NEGATIVES.iterdir())
        if d.is_dir()
    }


def test_every_negative_directory_has_truth():
    """Each negative case mirrors the positive layout (request + response)
    and adds its truth and its provenance, so a case cannot join the corpus
    without saying what the right answer is and where the bytes came from."""
    cases = _negative_cases()
    assert len(cases) >= 3, sorted(cases)
    for case, truth in cases.items():
        d = _NEGATIVES / case
        assert (d / "request.json").exists(), case
        assert (d / "response.json").exists() or (d / "stream.sse").exists(), case
        assert (d / "provenance.md").read_text().strip(), case
        assert {"host", "path", "provider", "inferred"} <= set(truth), case
        assert isinstance(truth["inferred"], bool), case


def _attribution_corpus() -> list[tuple[str, str, str, bytes, bytes, str, bool]]:
    """(case, host, path, request, response, true provider, truly inferred).

    Positives take their truth from `expect.json` and are inferred exactly
    when their host is not the provider's own (all fifteen are official
    today). Negatives take it from `truth.json`.
    """
    corpus = []
    for case, (host, path) in sorted(_CASES.items()):
        request, response, expect = _load(case)
        corpus.append(
            (case, host, path, request, response, expect["provider"], host not in _OFFICIAL_HOSTS)
        )
    for case, truth in _negative_cases().items():
        d = _NEGATIVES / case
        body = d / "response.json"
        response = body.read_bytes() if body.exists() else (d / "stream.sse").read_bytes()
        corpus.append(
            (
                f"negatives/{case}",
                truth["host"],
                truth["path"],
                (d / "request.json").read_bytes(),
                response,
                truth["provider"],
                truth["inferred"],
            )
        )
    return corpus


def test_provider_misattribution_rate_is_under_the_ceiling():
    """A case is MISATTRIBUTED when the provider label differs from the truth,
    or when a label the truth calls inferred reaches Python without
    `provider_inferred` (a guess presented as a fact is the failure this gate
    exists for), or when a proven label is flagged as a guess (a marker on
    every span would make the marker mean nothing)."""
    corpus = _attribution_corpus()
    wrong: list[str] = []
    for case, host, path, request, response, provider, inferred in corpus:
        sem = parse_llm_semantics(host, path, request, response, LimitsConfig().to_native())
        got_provider = sem.provider if sem is not None else None
        got_inferred = bool(sem.provider_inferred) if sem is not None else False
        if got_provider != provider:
            wrong.append(f"{case}: label {got_provider!r}, truth {provider!r}")
        elif got_inferred != inferred:
            wrong.append(f"{case}: inferred={got_inferred}, truth inferred={inferred}")
    rate = len(wrong) / len(corpus)
    assert rate <= _MISATTRIBUTION_CEILING, (
        f"provider misattribution {len(wrong)} of {len(corpus)} "
        f"({rate:.1%}, ceiling {_MISATTRIBUTION_CEILING:.1%} = 0 of {len(corpus)}): {wrong}"
    )
