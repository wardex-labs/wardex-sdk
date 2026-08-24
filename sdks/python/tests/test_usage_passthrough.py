"""The open usage model at the seam: the `wardex.usage.*` mirror, its bound,
and its gating.

U1, machine-checked as a SET EQUALITY: the emitted mirror keys are exactly
the provider usage tree's scalar leaf paths — not a membership check, which
would go green while half the leaves silently vanished.
"""

from __future__ import annotations

import json
import pathlib

from wardex_sdk import LimitsConfig
from wardex_sdk._assembly import Limitation, counters
from wardex_sdk._enums import CaptureMode, OperationName
from wardex_sdk._interceptors._ssl import SSLInterceptor
from wardex_sdk._semantics import USAGE_DROPPED_KEY

_REPO = pathlib.Path(__file__).resolve().parents[3]
_FIXTURES = _REPO / "crates" / "wardex-protocol" / "tests" / "fixtures" / "llm"

_MIRROR_PREFIX = "wardex.usage."


def _fixture(case: str, name: str) -> bytes:
    return (_FIXTURES / case / name).read_bytes()


def _scalar_leaf_paths(tree, prefix: str = "") -> set[str]:
    """The independent walker the equality is checked against."""
    out: set[str] = set()
    if isinstance(tree, dict):
        items = tree.items()
    elif isinstance(tree, list):
        items = ((str(i), v) for i, v in enumerate(tree))
    else:
        return out
    for key, value in items:
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, (dict, list)):
            out |= _scalar_leaf_paths(value, path)
        elif value is not None:
            out.add(path)
    return out


def _drive(
    request: bytes,
    response: bytes,
    host: str,
    path: str,
    *,
    limits: LimitsConfig | None = None,
    mode: CaptureMode = CaptureMode.AGENT,
):
    """One HTTP/1 exchange through the real byte seam."""
    from conftest import _FakeSSLSocket

    class _Config:
        debug = False

        def __init__(self) -> None:
            self.limits = limits or LimitsConfig()
            self.capture_mode = mode

    class _Client:
        def __init__(self) -> None:
            self.config = _Config()
            self.spans: list = []

        def capture_span(self, span) -> None:
            self.spans.append(span)

        def capture_deferred(self, job) -> None:
            # Inline: unit doubles may finalize synchronously.
            span = job.ctx.run(job.run)
            if span is not None:
                self.capture_span(span)

    client = _Client()
    itc = SSLInterceptor()
    itc._client = client
    itc._load_limits(client)
    sock = _FakeSSLSocket(None)
    sock.server_hostname = host
    itc._on_request_bytes(
        sock,
        b"POST " + path.encode() + b" HTTP/1.1\r\nHost: " + host.encode() + b"\r\n"
        b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(request)}\r\n\r\n".encode()
        + request,
    )
    itc._on_response_bytes(
        sock,
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
        + f"Content-Length: {len(response)}\r\n\r\n".encode()
        + response,
    )
    return client.spans[-1] if client.spans else None


def _mirror(span) -> dict:
    return {k: v for k, v in span.extra if k.startswith(_MIRROR_PREFIX)}


def test_usage_leaves_ride_as_wardex_usage_extras():
    """T-P5 / U1 — set equality against the fixture's own usage tree, plus
    the individual leaves that motivated the open model (the 1h cache-write
    tier, the web-search charge, the string tiers), and the normalization
    seam: `gen_ai.usage.reasoning.output_tokens` == the thinking leaf.
    """
    span = _drive(
        _fixture("anthropic_messages", "request.json"),
        _fixture("anthropic_messages", "response.json"),
        "api.anthropic.com",
        "/v1/messages",
    )
    assert span is not None
    usage_tree = json.loads(_fixture("anthropic_messages", "response.json"))["usage"]
    emitted = _mirror(span)
    assert set(emitted) == {_MIRROR_PREFIX + p for p in _scalar_leaf_paths(usage_tree)}
    assert emitted["wardex.usage.cache_creation.ephemeral_1h_input_tokens"] == 0
    assert emitted["wardex.usage.server_tool_use.web_search_requests"] == 1
    assert emitted["wardex.usage.service_tier"] == "standard"
    assert emitted["wardex.usage.inference_geo"] == "us"
    # raw leaf spelling preserved even where normalization renames:
    assert emitted["wardex.usage.output_tokens_details.thinking_tokens"] == 120
    assert span.gen_ai.reasoning_output_tokens == 120
    # and the S-3 rescope: the normalized input is the INCLUSIVE total, so it
    # equals leaf(input) + leaf(cache_read) + leaf(cache_creation), not the
    # raw leaf alone.
    assert span.gen_ai.input_tokens == (
        emitted["wardex.usage.input_tokens"]
        + emitted["wardex.usage.cache_read_input_tokens"]
        + emitted["wardex.usage.cache_creation_input_tokens"]
    )


def test_usage_cap_marks_and_counts():
    """T-P6 — the bound is one fact in three carriers: marker 44, the
    dropped-count extra, and the diagnostics bump. The normalized fields are
    untouched (U2) — the cap prunes the mirror, never the semconv totals.
    """
    before = counters.snapshot().get("interceptors.seam.usage_leaves_dropped", 0)
    span = _drive(
        _fixture("anthropic_messages", "request.json"),
        _fixture("anthropic_messages", "response.json"),
        "api.anthropic.com",
        "/v1/messages",
        limits=LimitsConfig(max_extra_keys=2),
    )
    assert span is not None
    assert Limitation.EXTRA_KEYS_DROPPED in span.capture_integrity.limitations
    extras = dict(span.extra)
    # 10 scalar leaves in the fixture, 2 kept -> 8 dropped.
    assert extras[USAGE_DROPPED_KEY] == 8
    assert len(_mirror(span)) == 2
    after = counters.snapshot().get("interceptors.seam.usage_leaves_dropped", 0)
    assert after == before + 1
    # U2: normalized fields extracted separately, cap-immune.
    assert span.gen_ai.input_tokens == 11000
    assert span.gen_ai.output_tokens == 500
    assert span.gen_ai.reasoning_output_tokens == 120


def test_unidentified_span_carries_no_usage_family_artifacts():
    """T-P6a — the gating pin: an UNIDENTIFIED span (usage-shaped body, no
    request identity, no core semantics) ships none of the family — no
    marker 44, no dropped-count key, no counter bump, zero mirror keys —
    even under a cap that would have dropped leaves. A marker explaining a
    key family that is not there cannot exist.
    """
    before = counters.snapshot().get("interceptors.seam.usage_leaves_dropped", 0)
    span = _drive(
        b"{}",
        b'{"usage": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}}',
        "api.openai.com",
        "/v1/chat/completions",
        limits=LimitsConfig(max_extra_keys=2),
        mode=CaptureMode.ALL,
    )
    assert span is not None
    assert span.gen_ai is None
    assert Limitation.EXTRA_KEYS_DROPPED not in span.capture_integrity.limitations
    extras = dict(span.extra)
    assert USAGE_DROPPED_KEY not in extras
    assert not _mirror(span)
    assert counters.snapshot().get("interceptors.seam.usage_leaves_dropped", 0) == before


def test_embeddings_span_is_embeddings_with_usage_once_and_no_output_type():
    """T-P7 — the embeddings span tells the truth now: request model and
    encoding formats extracted, `output_type` ABSENT (vectors are not text),
    the dimension count on its own block, usage exactly once, and the OTLP
    name comes from the operation and the REQUEST model.
    """
    span = _drive(
        _fixture("openai_embeddings", "request.json"),
        _fixture("openai_embeddings", "response.json"),
        "api.openai.com",
        "/v1/embeddings",
    )
    assert span is not None
    assert span.gen_ai.operation == OperationName.EMBEDDINGS
    assert span.gen_ai.request_model == "text-embedding-3-small"
    assert span.gen_ai.encoding_formats == ("float",)
    assert span.gen_ai.input_tokens == 2
    assert span.gen_ai.output_type is None
    assert span.embeddings is not None
    assert span.embeddings.dimension_count == 256
    assert _mirror(span) == {
        "wardex.usage.prompt_tokens": 2,
        "wardex.usage.total_tokens": 2,
    }

    # The encoded OTLP span: named for the operation and model, dimension
    # count on the wire, and gen_ai usage exactly once.
    from wardex_sdk import _wardex_native
    from wardex_sdk._types import Envelope, EnvelopeHeader, SdkInfo

    env = Envelope(
        header=EnvelopeHeader(
            event_id="evt-emb",
            api_key="k",
            sdk=SdkInfo(
                name="wardex.python",
                version="0",
                python_version="3",
                os="mac",
                arch="arm64",
            ),
            sent_at_ns=1,
        ),
        spans=(span,),
    )
    data = _wardex_native.codec.encode_otlp_traces(env)
    decoded = _wardex_native.codec.decode_otlp_traces(data)
    otlp_span = decoded["resource_spans"][0]["scope_spans"][0]["spans"][0]
    assert otlp_span["name"] == "embeddings text-embedding-3-small"
    attrs = otlp_span["attributes"]
    assert attrs["gen_ai.embeddings.dimension.count"] == 256
    assert attrs["gen_ai.usage.input_tokens"] == 2
    assert "gen_ai.output.type" not in attrs
