"""Internal type definitions — spec §2.2.

All types are frozen+slots dataclasses. No external dependencies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from typing import Protocol as TypingProtocol

from ._enums import (
    AgentType,
    CaptureSource,
    Direction,
    Modality,
    OperationName,
    OutputType,
    Protocol,
    ProviderName,
    SpanKind,
    StatusCode,
    ToolExecutionType,
    ToolType,
)

if TYPE_CHECKING:
    # Vocabulary that this module ANNOTATES but does not own. Three fields below
    # were free-form strings until the wire schema closed them; the members live
    # where the vocabulary is defined and enforced, which is one layer up.
    #
    # The import is deliberately type-only, and not to dodge a lint. `assembly`
    # sits ABOVE `_types` — `_assembly/_parentage.py` imports from here — so a
    # runtime import would be a real cycle and, worse, a lower layer reaching
    # upward for a definition. The annotations still say exactly what the fields
    # hold, and the enforcement lives where it can act: `SpanDraft.finish()`
    # refuses a marker that is not a `Limitation`, and it is the only place an
    # `InternalSpan` is constructed.
    from ._assembly._integrity import Limitation
    from ._assembly._parentage import ParentSource
    from ._assembly._vocab import LinkReason

# --- Basic ID types (bytes wrapper) ---


@dataclass(frozen=True, slots=True)
class TraceId:
    value: bytes  # 16 bytes

    def hex(self) -> str:
        return self.value.hex()

    @classmethod
    def generate(cls) -> TraceId:
        return cls(os.urandom(16))


@dataclass(frozen=True, slots=True)
class SpanId:
    value: bytes  # 8 bytes

    def hex(self) -> str:
        return self.value.hex()

    @classmethod
    def generate(cls) -> SpanId:
        return cls(os.urandom(8))


@dataclass(frozen=True, slots=True)
class SpanContext:
    trace_id: TraceId
    span_id: SpanId
    trace_flags: int = 0
    is_remote: bool = False


# --- GenAI attributes (typed; OTel mapping is handled by the Normalizer) ---


@dataclass(frozen=True, slots=True)
class GenAIAttributes:
    operation: OperationName | str
    provider: ProviderName | str | None = None

    request_model: str | None = None  # gen_ai.request.model (renamed from the old 'model')
    response_model: str | None = None  # gen_ai.response.model
    response_id: str | None = None  # gen_ai.response.id (new)

    # Tokens (cache/reasoning are included in input/output_tokens — OTel convention)
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    reasoning_output_tokens: int | None = None  # new (o1/thinking)

    # Request parameters
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None  # new
    top_k: float | None = None  # new (warning: OTel uses double)
    seed: int | None = None  # new
    frequency_penalty: float | None = None  # new
    presence_penalty: float | None = None  # new
    choice_count: int | None = None  # new (gen_ai.request.choice.count)
    stop_sequences: tuple[str, ...] | None = None
    encoding_formats: tuple[str, ...] | None = None  # new (embeddings)
    stream: bool | None = None  # new (gen_ai.request.stream)

    # Response
    finish_reasons: tuple[str, ...] | None = None  # new
    output_type: OutputType | str | None = None  # new (image/json/speech/text)
    time_to_first_chunk_s: float | None = None  # new (warning: OTel uses seconds)

    # System / tool (separated)
    system_instructions: bytes = b""  # new (opt-in, PII, structured JSON bytes)
    tool_definitions_hash: str | None = None  # new (actual definitions live in StateSnapshot)

    prompt_name: str | None = None  # gen_ai.prompt.name (optional)


@dataclass(frozen=True, slots=True)
class AgentAttributes:
    name: str
    id: str | None = None
    description: str | None = None
    version: str | None = None
    agent_type: AgentType = AgentType.PRIMARY
    parent_agent: str | None = None


@dataclass(frozen=True, slots=True)
class ToolAttributes:
    name: str  # gen_ai.tool.name
    call_id: str | None = None  # gen_ai.tool.call.id
    description: str | None = None  # gen_ai.tool.description (new)
    # gen_ai.tool.type (new): function/extension/datastore
    type: ToolType | str | None = None
    # wardex-custom (network/in_process)
    execution_type: ToolExecutionType = ToolExecutionType.NETWORK
    # tool.call.arguments / result use InternalSpan.input_data / output_data


# --- Replay / harness new types ---


@dataclass(frozen=True, slots=True)
class ConversationContext:
    """Identifies a single conversation session (multi-turn, multi-agent).
    Auto-issued + Scope override."""

    conversation_id: str  # gen_ai.conversation.id
    session_id: str | None = None  # wardex-custom (parent session)
    turn_index: int = 0


@dataclass(frozen=True, slots=True)
class CallSite:
    """The host code location that created the span. A clue for Replay's 'where to call again from'.
    Maps to OTel code.filepath/lineno/function. Cost is microseconds, no PII (code path only)."""

    file: str
    line: int
    function: str
    module: str | None = None


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str | None = None
    parameters_schema: bytes = b""  # JSON Schema, preserved as bytes
    version: str | None = None
    type: ToolType | str | None = None
    hash: str | None = None  # per-tool identity (consistency #4)


@dataclass(frozen=True, slots=True)
class ToolDefinitionSet:
    """The tool set at the time of the LLM call. Identity via set_hash → gen_ai.tool.definitions."""

    tools: tuple[ToolDefinition, ...] = ()
    set_hash: str = ""


@dataclass(frozen=True, slots=True)
class RetrievalAttributes:
    data_source_id: str | None = None  # gen_ai.data_source.id
    query_text: str | None = None  # gen_ai.retrieval.query.text (opt-in PII)
    documents: bytes = b""  # gen_ai.retrieval.documents (opt-in, JSON)


@dataclass(frozen=True, slots=True)
class EmbeddingsAttributes:
    dimension_count: int | None = None  # gen_ai.embeddings.dimension.count
    # encoding_formats lives on GenAIAttributes (gen_ai.request.encoding_formats)


@dataclass(frozen=True, slots=True)
class EvaluationAttributes:
    name: str | None = None  # gen_ai.evaluation.name
    explanation: str | None = None  # gen_ai.evaluation.explanation
    score_value: float | None = None  # gen_ai.evaluation.score.value
    score_label: str | None = None  # gen_ai.evaluation.score.label


# --- Forensic new types (v2) ---


@dataclass(frozen=True, slots=True)
class CaptureIntegrity:
    """How completely the data was captured. The primary criterion for replay feasibility.
    If the body wasn't captured / was truncated / dropped_chunk_count > 0, cached replay
    confidence drops."""

    request_headers_captured: bool = False
    request_body_captured: bool = False
    response_headers_captured: bool = False
    response_body_captured: bool = False
    redacted: bool = False
    truncated: bool = False
    dropped_chunk_count: int = 0
    # CLOSED on the wire. This was `tuple[str, ...]`, and the example the old
    # comment gave — "tls_inner_only" — had never actually been emitted by
    # anything: a free-form field invented a marker to describe itself. On the
    # wire it is `repeated Limitation limitation_codes`.
    limitations: tuple[Limitation, ...] = ()


@dataclass(frozen=True, slots=True)
class CorrelationInfo:
    """Basis and confidence of the match between an Adapters span and an Interceptors
    transport (async/concurrent environments).
    If confidence < 1.0, the backend/UI should indicate the match is an estimate."""

    operation_id: str | None = None
    request_id: str | None = None
    attempt_id: str | None = None
    active_span_id_at_capture: SpanId | None = None
    confidence: float = 1.0  # 0.0~1.0
    # CLOSED on the wire, and narrowed to ONE question: how was the parent
    # edge derived. The old comment listed `socket|timing|manual`, none of which
    # were ever produced, next to `adapter_hook`/`adapter_stream`, which were —
    # and those answered a different question ("which source observed this"),
    # already carried by `InternalSpan.capture_sources`. On the wire it is
    # `ParentSource parent_source`.
    strategy: ParentSource | None = None


# --- Span internal events/links ---


@dataclass(frozen=True, slots=True)
class InternalSpanEvent:
    name: str
    timestamp_ns: int
    attributes: tuple[tuple[str, str | int | float | bool], ...] = ()


@dataclass(frozen=True, slots=True)
class InternalSpanLink:
    trace_id: TraceId
    span_id: SpanId
    # CLOSED on the wire. A link is CAUSALITY where the parent edge is
    # CONTAINMENT, and `handoff_from` is what tells a renderer to draw a sibling
    # instead of nesting — so an unrecognized reason degrading to "no reason"
    # silently rebuilds the flame graph the link exists to prevent.
    reason: LinkReason | None = None


# --- Core data types ---


@dataclass(frozen=True, slots=True)
class InternalSpan:
    context: SpanContext
    parent_span_id: SpanId | None
    name: str
    kind: SpanKind
    start_time_ns: int
    end_time_ns: int
    status: StatusCode = StatusCode.UNSET
    status_message: str = ""

    # Typed attributes (instead of string keys)
    gen_ai: GenAIAttributes | None = None
    agent: AgentAttributes | None = None
    tool: ToolAttributes | None = None
    # Enriched by Interceptors (or created by Interceptors alone)
    transport: TransportAttributes | None = None
    retrieval: RetrievalAttributes | None = None  # new (RAG)
    embeddings: EmbeddingsAttributes | None = None  # new
    evaluation: EvaluationAttributes | None = None  # new
    cost_usd: float | None = None

    # New fields aligned with OTel
    conversation: ConversationContext | None = None  # gen_ai.conversation.id (+turn)
    call_site: CallSite | None = None  # code.filepath/lineno/function
    error_type: str | None = None  # error.type (required on failure)
    server_address: str | None = None  # server.address
    server_port: int | None = None  # server.port (required when address is set)
    workflow_name: str | None = None  # gen_ai.workflow.name

    # Forensic layer (v2) — what/how much/how reliably was captured
    capture_sources: tuple[CaptureSource, ...] = ()
    capture_integrity: CaptureIntegrity | None = None
    correlation: CorrelationInfo | None = None

    # Task I/O — unified input/output across all Spans
    # Filled by Adapters: prompts, tool args, user input, etc. (semantic)
    # When Interceptors run alone: raw request/response bytes
    input_data: bytes = b""
    output_data: bytes = b""

    # User-defined attributes (free-form, use sparingly)
    extra: tuple[tuple[str, str | int | float | bool], ...] = ()

    events: tuple[InternalSpanEvent, ...] = ()
    links: tuple[InternalSpanLink, ...] = ()


@dataclass(frozen=True, slots=True)
class TransportTiming:
    tcp_connect_ms: float = 0.0
    tls_handshake_ms: float = 0.0
    ttfb_ms: float = 0.0
    ttft_ms: float = 0.0
    transfer_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class HttpMeta:
    method: str
    url: str
    status_code: int


@dataclass(frozen=True, slots=True)
class GrpcMeta:
    service: str
    method: str
    stream_id: int | None = None
    status_code: int | None = None
    encoding: str | None = None


@dataclass(frozen=True, slots=True)
class WebSocketMeta:
    opcode: int
    direction: str  # "client_to_server" | "server_to_client"


@dataclass(frozen=True, slots=True)
class McpMeta:
    rpc_method: str
    rpc_id: str | None = None


@dataclass(frozen=True, slots=True)
class SseMeta:
    event_type: str | None = None


@dataclass(frozen=True, slots=True)
class A2aMeta:
    task_id: str
    transport: str  # "http" | "grpc"


@dataclass(frozen=True, slots=True)
class TransportAttributes:
    """Transport-layer observation data that interceptors enrich the Span with.
    Covers both network (HTTP, gRPC, etc.) and IPC (stdio, pipe, etc.).
    A Span carrying this attribute is auto-generated even when Interceptors run alone."""

    connection_id: str = ""
    protocol: Protocol = Protocol.HTTP
    direction: Direction = Direction.OUTBOUND

    # Timing — network-only fields (tcp_connect_ms, tls_handshake_ms) are 0 for IPC
    timing: TransportTiming = field(default_factory=TransportTiming)

    request_size: int = 0
    response_size: int = 0

    # Per-protocol metadata (only one is set)
    http: HttpMeta | None = None
    grpc: GrpcMeta | None = None
    websocket: WebSocketMeta | None = None
    mcp: McpMeta | None = None
    sse: SseMeta | None = None
    a2a: A2aMeta | None = None

    # Blob reference (large binaries)
    request_blob_ref: str | None = None
    response_blob_ref: str | None = None

    # Modality + streaming
    request_modality: Modality = Modality.TEXT
    response_modality: Modality = Modality.TEXT
    is_streaming: bool = False
    chunk_index: int = 0
    is_final_chunk: bool = True
    connection_reused: bool = False


@dataclass(frozen=True, slots=True)
class InputRef:
    """A domain-neutral reference to external content that a prompt depended on.
    The body lives in blob_ref (or an external store)."""

    key: str  # opaque identifier (e.g. "POORCODE.md", "retrieved_docs")
    content_hash: str  # "sha256:..."
    blob_ref: str | None = None


@dataclass(frozen=True, slots=True)
class InternalStateSnapshot:
    trace_id: TraceId
    span_id: SpanId
    timestamp_ns: int
    snapshot_type: str = "turn_start"  # "span_start" | "span_end" | "turn_start"
    turn_index: int = 0
    conversation_state: bytes = b""  # accumulated conversation (LogRecord body)
    tool_definitions: ToolDefinitionSet | None = None
    attributes: tuple[tuple[str, str | int | float | bool], ...] = ()  # opaque kv
    input_refs: tuple[InputRef, ...] = ()  # external content references (hashed)


@dataclass(frozen=True, slots=True)
class SdkInfo:
    name: str  # "wardex.python"
    version: str  # SDK version, from package metadata (e.g. "0.1.0b1")
    python_version: str
    os: str
    arch: str
    adapters: tuple[str, ...] = ()
    interceptors: tuple[str, ...] = ()
    otel_semconv_version: str = "1.36.0"  # new — the OTel semconv version reconciled against
    shell: str = ""  # "zsh", "bash" (static host metadata, v2 consistency #1)


@dataclass(frozen=True, slots=True)
class ClientReport:
    timestamp_ns: int
    discarded_events: tuple[tuple[str, int], ...] = ()
    failed_sends: int = 0
    queue_depth: int = 0
    uptime_ms: int = 0


@dataclass(frozen=True, slots=True)
class EnvelopeHeader:
    event_id: str  # UUID
    api_key: str
    sdk: SdkInfo
    sent_at_ns: int


@dataclass(frozen=True, slots=True)
class InternalEnvelope:
    header: EnvelopeHeader
    spans: tuple[InternalSpan, ...] = ()
    state_snapshots: tuple[InternalStateSnapshot, ...] = ()
    client_report: ClientReport | None = None


# --- Protocol parser return types ---


@dataclass(frozen=True, slots=True)
class ParsedMessage:
    protocol: Protocol
    method: str | None = None
    url: str | None = None
    status_code: int | None = None
    headers: tuple[tuple[str, str], ...] = ()
    body: bytes = b""
    header_len: int = 0
    # True when the parser stored fewer body bytes than arrived. `limitations`
    # says why; both travel to CaptureIntegrity so a capped body is visible to
    # the user rather than silently short.
    #
    # The Rust parsers still produce `Vec<&'static str>`, and that is not debt:
    # `wardex-protocol` is the lowest layer and depending on the generated proto
    # types to name a marker would invert the crate graph for one string
    # (`body_cap_exceeded` is the only marker Rust emits today). The strings are
    # resolved to members once, at the boundary that builds this dataclass, and
    # `tests/test_limitation_census.py` scans `crates/` on every run so a Rust
    # marker with no member fails CI rather than reaching the seam.
    truncated: bool = False
    limitations: tuple[Limitation, ...] = ()


# --- Callback protocols (concrete signatures instead of Callable) ---


class BeforeSendCallback(TypingProtocol):
    def __call__(self, envelope: InternalEnvelope) -> InternalEnvelope | None:
        """Returning None cancels the send."""
        ...


class SpanEnrichCallback(TypingProtocol):
    """Used when Interceptors enrich an existing Span with transport attributes."""

    def __call__(self, span: InternalSpan, transport: TransportAttributes) -> InternalSpan: ...


class FlushCallback(TypingProtocol):
    def __call__(self, envelope: InternalEnvelope) -> None: ...
