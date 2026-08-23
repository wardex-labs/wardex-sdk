"""What the byte seam reports when a provider REFUSES the call.

One rule, and every test here is a face of it: a call wardex recognized as an
LLM call is reported as one whatever the provider answered. The response code
decides the span's `status`, never whether the span exists or what it is allowed
to carry.

Before this, the code decided both, and it did so through a predicate whose
every field is response-side. A rate limit therefore produced:

  * under the default `capture_mode="agent"` — no span at all, so the request
    that was throttled is indistinguishable in the data from one never made;
  * under `capture_mode=ALL` — a span with `gen_ai=None` next to a populated
    `gen_ai.input.messages` extra, which is a span contradicting itself, and a
    `semantic_parse_failed` marker claiming wardex could not read a body it had
    read correctly.

The same prompt was exported when the call succeeded and dropped when it
failed. Nothing configured that and nothing documented it; the server's status
line was silently acting as a capture policy. The failures are the calls an
operator goes looking for, so this was the exact inversion of what the SDK is
for.

The gate widening is admitted only for a 4xx/5xx, and `test_a_two_hundred_...`
below is why: both provider gates in the Rust parser are substring matches, so
a 200 from an internal service at an `anthropic`-ish host can parse as a chat
call. That shape is genuinely ambiguous and stays dropped exactly as it is
today; a refusal from a host and path that parse as a provider is not.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from wardex_sdk import _hub
from wardex_sdk._assembly import Limitation, counters
from wardex_sdk._config import BackendConfig, WardexConfig
from wardex_sdk._enums import CaptureMode, OperationName, ProviderName, StatusCode
from wardex_sdk._interceptors._seam import ByteSeamInterceptor
from wardex_sdk._interceptors._trackers import _Http1Tracker

PROMPT = "summarize the incident report"
REQUEST = json.dumps(
    {
        "model": "claude-opus-4-1",
        "max_tokens": 1024,
        "temperature": 0.7,
        "messages": [{"role": "user", "content": PROMPT}],
    }
).encode()

RATE_LIMITED = json.dumps(
    {"type": "error", "error": {"type": "rate_limit_error", "message": "rate limit exceeded"}}
).encode()
COMPLETION = json.dumps(
    {
        "id": "msg_1",
        "model": "claude-opus-4-1",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 30},
        "content": [{"type": "text", "text": "done"}],
    }
).encode()


class _Client:
    def __init__(self, mode: CaptureMode) -> None:
        self.config = WardexConfig(capture_mode=mode, backend=BackendConfig(api_key="k"))
        self.spans: list = []

    def capture_span(self, span) -> None:
        self.spans.append(span)

    def capture_snapshot(self, snapshot) -> None:
        pass

    def close(self, timeout: float = 5.0) -> None:
        pass


class _Seam(ByteSeamInterceptor):
    """A seam with the transport half stubbed, so only the policy is under test."""

    def __init__(self) -> None:
        super().__init__()
        self._tracker = _Http1Tracker()

    def _select_tracker(self, obj):
        return self._tracker

    def _resolve_timing(self, obj, st):
        return 0.0, 0.0, False, ()

    def name(self) -> str:
        return "test_seam"

    def install(self, client, ctx=None) -> None:
        self._client = client

    def uninstall(self) -> None:
        pass


def _drive(
    *,
    mode: CaptureMode = CaptureMode.AGENT,
    status_line: bytes,
    body: bytes,
    host: bytes = b"api.anthropic.com",
    path: bytes = b"/v1/messages",
    request: bytes = REQUEST,
    content_type: bytes = b"application/json",
):
    """Push one real request/response pair through the seam; return the span or None.

    No ambient wardex span is ever entered, which is the condition that matters:
    under the `agent` default that is precisely when the capture policy has to
    decide from the semantics alone.
    """
    _hub.reset_for_test()
    counters.reset()
    client = _Client(mode)
    _hub.set_client(client)
    seam = _Seam()
    seam._client = client
    seam._load_limits(client)

    obj = SimpleNamespace(server_hostname=host.decode())
    seam._on_request_bytes(
        obj,
        b"POST " + path + b" HTTP/1.1\r\nHost: " + host + b"\r\n"
        b"Content-Type: application/json\r\nContent-Length: "
        + str(len(request)).encode()
        + b"\r\n\r\n"
        + request,
    )
    seam._on_response_bytes(
        obj,
        b"HTTP/1.1 " + status_line + b"\r\nContent-Type: " + content_type + b"\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body,
    )
    return client.spans[0] if client.spans else None


def _limitations(span) -> list[str]:
    return [limitation.value for limitation in span.capture_integrity.limitations]


def _extra(span) -> dict:
    return dict(span.extra or ())


# ==========================================================================
# A refused call is still an LLM call
# ==========================================================================


@pytest.mark.parametrize(
    "status_line",
    [
        b"429 Too Many Requests",
        b"401 Unauthorized",
        b"500 Internal Server Error",
        b"529 Overloaded",
    ],
)
def test_a_refused_call_keeps_the_identity_the_request_established(status_line):
    """The identity was never in the response. It was in the body wardex sent.

    Under the default mode this span did not exist, so a throttled deploy and a
    quiet one produced the same data — the failure mode with no evidence of
    itself.
    """
    span = _drive(status_line=status_line, body=RATE_LIMITED)

    assert span is not None, "the call the operator is looking for was dropped"
    assert span.status is StatusCode.ERROR
    assert span.error_type == status_line.split()[0].decode()
    assert span.gen_ai is not None
    assert span.gen_ai.provider is ProviderName.ANTHROPIC
    assert span.gen_ai.operation is OperationName.CHAT
    assert span.gen_ai.request_model == "claude-opus-4-1"
    assert span.gen_ai.temperature == 0.7


def test_the_refused_span_claims_nothing_about_the_response_half():
    """Widening what the request establishes must not back-fill what it cannot.

    `output_type` is the trap: the Rust parser sets it unconditionally, so
    without suppression a refused call ships `output_type="text"` beside no
    tokens and no output messages — asserting it produced text output when the
    provider produced an error.
    """
    span = _drive(status_line=b"429 Too Many Requests", body=RATE_LIMITED)

    assert span.gen_ai.response_model is None
    assert span.gen_ai.input_tokens is None
    assert span.gen_ai.output_tokens is None
    assert span.gen_ai.output_type is None
    assert "gen_ai.output.messages" not in _extra(span)


def test_a_refused_call_carries_what_a_successful_one_carries():
    """The consistency this fix is really about.

    The two spans differ in what the provider returned and in nothing else. A
    field present on success and absent on failure would put the response code
    back in charge of capture content — the implicit rule being removed here,
    reintroduced one layer down and harder to see.
    """
    ok = _drive(status_line=b"200 OK", body=COMPLETION)
    refused = _drive(status_line=b"429 Too Many Requests", body=RATE_LIMITED)

    for span in (ok, refused):
        assert span.gen_ai.provider is ProviderName.ANTHROPIC
        assert span.gen_ai.request_model == "claude-opus-4-1"
        assert PROMPT in _extra(span)["gen_ai.input.messages"]

    assert ok.status is StatusCode.OK
    assert refused.status is StatusCode.ERROR


def test_both_capture_modes_agree_about_a_refused_call():
    """`capture_mode` chooses which traffic is captured, never what a captured
    span is allowed to say. The two modes disagreeing about one transaction's
    CONTENTS is the shape that makes a setting impossible to reason about.
    """
    agent = _drive(mode=CaptureMode.AGENT, status_line=b"429 Too Many Requests", body=RATE_LIMITED)
    every = _drive(mode=CaptureMode.ALL, status_line=b"429 Too Many Requests", body=RATE_LIMITED)

    assert agent.gen_ai.request_model == every.gen_ai.request_model
    assert _limitations(agent) == _limitations(every) == []
    assert _extra(agent) == _extra(every)


# ==========================================================================
# semantic_parse_failed means what it says
# ==========================================================================


def test_an_http_error_is_not_a_parse_failure():
    """The body parsed correctly; it was an error envelope. Saying otherwise
    sent every rate limit and every auth failure out under a marker meaning
    "wardex could not understand this", while the truth was already on the span
    as `status=ERROR` and `error.type`.
    """
    span = _drive(mode=CaptureMode.ALL, status_line=b"429 Too Many Requests", body=RATE_LIMITED)

    assert span is not None, "the span must EXIST, or this asserts nothing"
    assert Limitation.SEMANTIC_PARSE_FAILED not in span.capture_integrity.limitations


def test_semantic_parse_failed_still_has_a_live_path():
    """The marker must not be narrowed into unreachability — a member with no
    reachable slot is a rule that stopped being enforced without anyone
    deciding to stop enforcing it.

    A 200 from an LLM host whose body yields no tokens, no response model and no
    output messages is exactly what it is for. The body below is nonsense ON
    THE CHAT ENDPOINT ITSELF: the old example (`/v1/messages/batches`) stopped
    reaching the parser when the endpoint table replaced substring matching —
    that path was one of the false positives the table exists to remove.
    """
    span = _drive(
        mode=CaptureMode.ALL,
        status_line=b"200 OK",
        body=json.dumps({"id": "b", "type": "unexpected_shape"}).encode(),
        path=b"/v1/messages",
        request=b"{}",
    )

    assert span is not None
    assert Limitation.SEMANTIC_PARSE_FAILED in span.capture_integrity.limitations


# ==========================================================================
# What the widening deliberately does NOT admit
# ==========================================================================


def test_a_two_hundred_from_a_provider_shaped_host_is_still_dropped():
    """Why the widening is gated on the response being an ERROR.

    Both provider gates in the Rust parser are substring matches on host and
    path, so an internal service at `anthropic-proxy.corp/v1/messages` whose
    body carries a `model` field parses as an Anthropic chat call. On a 2xx that
    is genuinely ambiguous, it is dropped today, and admitting it would start
    exporting an unrelated service's request and response bodies as a side
    effect of an attribute fix.
    """
    span = _drive(
        status_line=b"200 OK",
        body=json.dumps({"ok": True}).encode(),
        host=b"anthropic-proxy.corp",
        request=json.dumps({"model": "widget-v2"}).encode(),
    )

    assert span is None


def test_a_refusal_needs_a_model_in_the_request_to_be_identified():
    """Provider and operation alone are not identity. Off the streaming path the
    Rust parser always fills `provider`, so accepting those two would make the
    gate read as `sem is not None` and admit any traffic the parser touched.
    """
    span = _drive(status_line=b"429 Too Many Requests", body=RATE_LIMITED, request=b"{}")

    assert span is None


def test_a_refusal_from_a_host_the_parser_does_not_know_stays_invisible():
    """The honest boundary, asserted rather than described.

    For a self-hosted or proxied endpoint `parse_llm_semantics` returns None
    outright — it classifies on the RESPONSE body, and an error envelope carries
    none of the markers it looks for. So "rate limits are now captured" is true
    of recognized providers and of nothing else, and a release note that omits
    the qualifier is a false claim about the SDK.
    """
    span = _drive(status_line=b"429 Too Many Requests", body=RATE_LIMITED, host=b"llm-gw.internal")

    assert span is None


def test_a_stream_from_an_unknown_provider_still_says_so():
    """The stream markers hang off `streamed`, not off identity. Keying them to
    the identity test instead makes a known provider's usage-less stream quietly
    stop reporting that it was reassembled at all.
    """
    span = _drive(
        mode=CaptureMode.ALL,
        status_line=b"200 OK",
        body=b'event: error\ndata: {"type":"error","error":{"type":"overloaded_error"}}\n\n',
        host=b"llm-gw.internal",
        content_type=b"text/event-stream",
    )

    assert span is not None
    assert span.gen_ai is None
    assert Limitation.SSE_UNKNOWN_PROVIDER in span.capture_integrity.limitations
    assert Limitation.REASSEMBLED_FROM_STREAM not in span.capture_integrity.limitations
