"""Capture-integrity vocabulary — design §6.5 tier 3, §6.5.1, §I9.

A limitation marker says what wardex *failed* to capture, or captured only by
interpretation. It is the third and last tier of the "concept has no home in
the vocabulary" disposition table: tier 1 is a namespaced ``extra`` key, tier 2
is an ``InternalSpanEvent``, tier 3 is a marker here. Anything that would need
a new span kind is dropped.

The enum is CLOSED, and as of the 2026-07-29 census (design §6.5.1) it is also
**complete**: it is the full vocabulary, not a sample of it. Every limitation
string the SDK can attach to a span today — Python and Rust — has a member
below. Adding a member is a core change on purpose: the dashboard renders these
and step 3b makes them a proto enum whose value names lock the moment they are
declared (``buf`` ``ENUM_VALUE_SAME_NAME``), so a late correction costs a
second deliberate schema break. Emitters must never invent a marker string
inline.

**37 members = 15 declared in step 0 + 21 from the census + 1 from §5.4.**
The census read every assignment and append site that reaches
``CaptureIntegrity.limitations`` and found 25 distinct strings (24 Python, 1
Rust). Four of those merged away — see the ``NOTE (census)`` comments on
``CONNECT_TIMING_UNAVAILABLE``, ``PAYLOAD_COMPRESSED``, ``FRAME_PARSE_FAILED``
and ``CHILD_SPAN_UNCLOSED`` — leaving 21 new members.

**Two vocabularies, not one — do not merge them.** A ``disabled_reason`` is why
a Rust parser latched *itself* off for a whole connection; no span exists to
carry it (nothing was ever parsed), so it stays a ``&'static str`` that
``init(debug=True)`` prints once to stderr. ``headers_exceeded``, ``not_http``
and ``chunk_size_exceeded`` are those, and they are deliberately absent below.
The trap is ``stream_buffer_exceeded``, which is **both**: the HTTP/1 latch
(``crates/wardex-protocol/src/http1.rs``) is a debug reason and stays out,
while the MCP-stdio JSON-RPC instance (``json_rpc.rs``) happens with a pending
request already open, so a span exists and design §4.6 site-3 makes it
reportable — that is why ``STREAM_BUFFER_EXCEEDED`` is a member. Step 3b must
keep the two vocabularies in separate proto enums; they have different
lifetimes and different consumers (wire contract vs stderr).

**Read this before adding a member.** ``tests/test_limitation_census.py``
re-runs the census against the source on every test run and fails if any marker
string, in either language, has no member here. That test is the mechanism;
this docstring is only its description. If it just failed on you, the question
to answer is not "how do I make it pass" but "is my new string a wire-contract
marker (add a member) or a connection-level debug reason (add it to the
exclusion set instead)".

Migration status (design §11): this module is the **python half** of the census
and it was the hard prerequisite for step 3a, which routed all six span-emit
sites through ``SpanDraft.finish()`` — and ``finish()`` raises
``VocabularyError`` on a marker that is not a member here, which the emit site's
``guard()`` swallows. Merged against an incomplete enum, 3a would have silently
deleted every gRPC span, every streaming chat span, every WS span and every
adapter span. That step has landed: **every Python emitter now names a member**,
the seven pre-rename free strings are gone from the tree, and
``tests/test_limitation_census.py`` asserts an EMPTY string census as the
standing rule. One producer is still textual — ``body_cap_exceeded``, built in
``crates/wardex-protocol`` and resolved once at the PyO3 boundary by
``Limitation.from_wire`` — and retyping that is step 3b, which also makes proto
the single source of truth for every language SDK (§6.6).
"""

from __future__ import annotations

from enum import Enum


class Limitation(Enum):
    """CLOSED and complete. The only legal values of a span's limitation markers.

    Values are the wire contract (``CaptureIntegrity.limitations``); they are
    lower_snake to match every other wardex enum and must not be renamed
    casually. Four values below were renamed by the census on the way in — that
    rename is itself a wire change and rides step 3b, the one commit that
    already breaks ``wardex.v1`` on purpose (§6.7). There is no second chance.

    Each member records the condition that emits it and the site that does so,
    as module-qualified functions rather than line numbers. A member marked
    "declared; no emitter" is vocabulary the design named and a later migration
    step wires up — it is not dead code, and it is not evidence that the census
    missed something.
    """

    @classmethod
    def from_wire(cls, value: str) -> Limitation | None:
        """A marker that arrived as a STRING, resolved to its member, or None.

        Exactly one caller is legitimate and it is the PyO3 boundary: the Rust
        protocol parsers still build `Vec<&'static str>` (retyping that is step
        3b, along with `bindings/python/src/lib.rs` and the PII walk that runs
        regexes over the strings), so a marker produced in Rust reaches Python
        as text and has to be resolved once, at the seam that folds it into a
        span. Python-side emitters name the member directly and must not come
        here — a string-to-member lookup used as a general entry point is the
        drift this enum exists to end.

        Returns `None` rather than raising for an unrecognized string, because
        the alternative on that path is deleting the span (`finish()` rejects a
        marker it cannot type) over a marker wardex itself produced. That is
        not a silent hole: `tests/test_limitation_census.py` scans `crates/`
        and `bindings/` on every run and fails the build if the Rust side
        starts emitting a string with no member here, so an unknown value is
        unreachable rather than merely tolerated.

        A dict lookup rather than `cls(value)` in a `try`: `assembly/` is held
        to zero silent swallows (C-S4), and an `except ValueError: return None`
        here would be one — indistinguishable in a diff from a swallow that is
        hiding something.
        """
        return _BY_VALUE.get(value)

    # ------------------------------------------------------------------
    # Parentage: the edge exists but was not derived from a live scope
    # I4 — a guess reports itself. Each of these travels with a
    # CorrelationInfo whose confidence is below 1.0.
    # ------------------------------------------------------------------

    PARENT_UNRESOLVED = "parent_unresolved"
    """A parent was expected and no scope, unit or header supplied one.

    Attached automatically by ``assembly/_parentage.py``'s ``_MARKER`` table for
    ``ParentSource.UNRESOLVED``. Declared; fires the moment ``resolve_parentage``
    acquires its first caller in step 1.
    """

    UNIT_INFERRED_SOLE = "unit_inferred_sole"
    """Exactly one logical unit was live, so it was taken as the parent.

    Attached automatically by ``assembly/_parentage.py``'s ``_MARKER`` table for
    ``ParentSource.UNIT_SOLE``. Declared; fires in step 1.
    """

    CORRELATION_CONFLICT = "correlation_conflict"
    """A framework-id alias and the live context disagreed about the trace.

    Design §5.5. Declared; no emitter — the caller attaches it explicitly via
    ``Parentage.with_limitation``, which arrives with ``assembly/_units.py``.
    """

    # ------------------------------------------------------------------
    # Context propagation: the mechanism that produces parentage is weakened
    # or absent on this runtime/seam. Declared degradation, not a workaround
    # (design §5.3-(v-b), §9).
    # ------------------------------------------------------------------

    CONTEXT_PROPAGATION_DEGRADED = "context_propagation_degraded"
    """The seam propagates context, but not with full fidelity — a thread-pool
    tool handler that copies the context at submit time, say.

    Declared; no emitter until the adapter seams of step 6.
    """

    CONTEXT_PROPAGATION_UNAVAILABLE = "context_propagation_unavailable"
    """The seam cannot propagate context at all on this runtime, so every span
    below it is a root.

    Declared; no emitter until step 6.
    """

    # ------------------------------------------------------------------
    # Unit lifecycle: a logical unit ended for a reason other than the
    # framework finishing it. I10 — eviction emits, it never drops silently.
    # ------------------------------------------------------------------

    UNIT_EVICTED = "unit_evicted"
    """The correlation table hit ``max_units`` and this unit was evicted before
    the framework closed it.

    Points at ``max_units`` specifically. Deliberately NOT merged with
    ``CONNECTION_EVICTED``, which points at ``max_connections``: merging them
    would send a user to turn the wrong knob. Declared; no emitter until
    ``assembly/_units.py``.
    """

    UNIT_INTERRUPTED = "unit_interrupted"
    """The unit was torn down by cancellation or interpreter shutdown rather
    than by a normal end-of-run.

    Declared; no emitter until ``assembly/_units.py``.
    """

    CHILD_SPAN_UNCLOSED = "child_span_unclosed"
    """A child span was force-closed by its parent's teardown or by an eviction
    instead of by its own completion event, so its ``end_time_ns`` is the
    teardown instant and its status is synthesized.

    Emitted from ``adapters/_assembler.py::SessionAssembler._open_tool``
    (open-tool table hit ``max_session_entries``) and ``::_finalize`` (session
    ended with tools still open), both via ``_emit_tool(markers=...)``. Until
    step 3a rewired those two sites it was the free string
    ``"tool_span_unclosed"``.

    NOTE (census): the four-way merge that loses nothing. ``tool_span_unclosed``
    folded into this step-0 member because the marker rides the tool span
    itself, where ``gen_ai.operation.name=execute_tool`` already says the child
    was a tool. This is the single point where the two drifts overlapped —
    vocabulary without an emitter meeting an emitter without vocabulary — and
    without the census it would have frozen into the wire as two names for one
    fact.
    """

    # ------------------------------------------------------------------
    # Adapter lifecycle (I7)
    # ------------------------------------------------------------------

    ADAPTER_UNINSTALLED = "adapter_uninstalled"
    """The span was closed by ``uninstall()`` rather than by the framework.

    Declared; no emitter. Deliberately NOT merged with ``WS_NO_CLOSE`` even
    though both of today's ``ws_no_close`` sites sit inside ``uninstall()``:
    the fact a user reads off ``WS_NO_CLOSE`` is capture completeness (no CLOSE
    frame, so close code and duration are untrustworthy), not lifecycle. The
    two are correct *together*.
    """

    PATCH_SUPERSEDED = "patch_superseded"
    """Something else re-patched a symbol wardex had already patched, so the
    interception wardex installed is no longer the one in effect.

    Declared; no emitter until ``assembly/_patchset.py``.
    """

    # ------------------------------------------------------------------
    # Observation completeness (I8, §8.1 rule W)
    # ------------------------------------------------------------------

    NO_WIRE_EVIDENCE = "no_wire_evidence"
    """The span was assembled entirely from framework callbacks; no interceptor
    ever saw the bytes, so payloads are the framework's account of them.

    Declared; no emitter until the ownership table of §8.2.
    """

    POSSIBLE_DUPLICATE_CHAT = "possible_duplicate_chat"
    """An adapter emitted a ``chat`` span that an installed interceptor may have
    emitted as well. Dual-instrumentation policy, §8.1.

    Declared; no emitter until the ownership table of §8.2.
    """

    STREAM_BUFFER_EXCEEDED = "stream_buffer_exceeded"
    """A stream parser's reassembly buffer hit ``max_stream_buffer_bytes`` and
    the parser latched off mid-stream.

    Declared; no emitter yet. Reportable only on the MCP-stdio JSON-RPC path
    (``crates/wardex-protocol/src/json_rpc.rs``), where a pending request means
    a span already exists to carry it — design §4.6 site-3 wires that up. The
    identically-named HTTP/1 latch in ``http1.rs`` is NOT this: it is a
    connection-level ``disabled_reason``, it has no span, and it stays a debug
    string. Same string, two fates; see the module docstring.
    """

    TOOL_CALL_ID_UNAVAILABLE_IN_PROCESS = "tool_call_id_unavailable_in_process"
    """The framework ran the tool in-process and never exposed a tool-call id,
    so the tool span cannot be joined to the assistant message that requested it.

    Declared; no emitter until step 6.
    """

    SNAPSHOT_TYPE_UNKNOWN = "snapshot_type_unknown"
    """A snapshot arrived with a type outside ``SnapshotType``, so it was
    recorded as ``UNSPECIFIED``.

    Emitted from ``assembly/_snapshot.py::SnapshotDraft.__init__``, when
    ``coerce_snapshot_type`` cannot resolve the value the caller handed
    ``capture_state_snapshot``. Step 3a closed ``SnapshotType`` and built this
    emitter; before it, an unrecognized type was flattened to ``UNSPECIFIED``
    by ``codec.rs``'s ``map_snap`` with nothing recorded anywhere.
    """

    # ------------------------------------------------------------------
    # Identity ambiguity (§5.4) — the census found no emitter for this one
    # ------------------------------------------------------------------

    TOOL_NAME_COLLISION = "tool_name_collision"
    """Two distinct MCP servers exposed tools that sanitize to the same
    CLI-visible name, so the name on this span cannot identify which server ran.

    Design §5.4's V3 correction. Declared; no emitter — the census found none,
    because the detector is built by step 6 with the rest of
    ``adapters/anthropic/_names.py``. It is the 37th member and the one that
    comes from neither step 0 nor the census; §6.5.1 resolves that discrepancy
    explicitly in favor of declaring it now, since 3b locks value names.
    """

    # ------------------------------------------------------------------
    # Transport timing (census) — a duration is missing, not zero
    # ------------------------------------------------------------------

    CONNECT_TIMING_UNAVAILABLE = "connect_timing_unavailable"
    """``tcp_connect_ms`` is 0 because it could not be measured, not because the
    connection was instant.

    Emitted from ``interceptors/_socket.py::RawSocketInterceptor._resolve_timing``
    (always — the raw-socket seam has no connect-time store) and
    ``interceptors/_ssl.py::SSLInterceptor._resolve_timing`` (sync path: the
    shared timing store had no record for this fileno; async path: no stamped
    ``_wardex_timing`` record at all).

    NOTE (census): absorbed the free string ``async_connect_unavailable``
    (``_ssl.py::_resolve_timing``, anyio/httpx path where TLS and TCP are
    separate layers so ``total_ms`` is 0 and connect cannot be derived). What is
    lost is the provenance — sync fileno miss vs anyio layer split. Merged
    anyway: both assert the same fact, ``tcp_connect_ms`` is unknown rather than
    zero, and the user action in both cases is the same (none).
    """

    TTFT_UNAVAILABLE_H2 = "ttft_unavailable_h2"
    """Time-to-first-token is absent on a streamed HTTP/2 response: the seam
    sees DATA frames, not SSE event boundaries.

    Emitted from ``interceptors/_seam.py::ByteSeamInterceptor._emit_span`` when
    a reassembled stream carries core semantics and ``txn.version == "2"``.
    """

    TTFT_IPC_APPROXIMATION = "ttft_ipc_approximation"
    """Time-to-first-token was measured at the IPC boundary, not at the wire, so
    it includes subprocess and pipe latency.

    Emitted from ``adapters/_assembler.py::SessionAssembler._emit_chat`` whenever
    a first-delta timestamp exists for the turn.
    """

    TRANSPORT_TIMING_UNAVAILABLE_SUBPROCESS = "transport_timing_unavailable_subprocess"
    """No transport timing at all: the LLM call happened inside a CLI subprocess
    and wardex observed only the IPC stream.

    Emitted from ``adapters/_assembler.py`` as the module constant
    ``_BASE_LIMITATION``, which rides *every* span the Agent SDK adapter builds
    — chat, tool, subagent and the root ``invoke_agent``.
    """

    # ------------------------------------------------------------------
    # Caps reached (census). These stay four members because they are four
    # different FACTS about the capture, each with a different replay
    # consequence — not, as the first draft of this section claimed, because
    # each names a different `crates/wardex-limits` knob. Two of them
    # (BODY_CAP_EXCEEDED, GRPC_MESSAGE_TRUNCATED) are driven by the same
    # `max_body_bytes`, and a fifth limit — `max_ws_frame_bytes` — surfaces
    # under FRAME_PARSE_FAILED rather than here. Each docstring below names the
    # knob that was MEASURED to drive it; step 3b copies these into the proto
    # enum comments, so a knob named from memory becomes a wire-visible lie.
    # ------------------------------------------------------------------

    BODY_CAP_EXCEEDED = "body_cap_exceeded"
    """An HTTP body was stored up to ``max_body_bytes`` / ``max_opaque_body_bytes``
    and the rest was consumed without being kept.

    The only marker produced in Rust: ``crates/wardex-protocol/src/http1.rs``,
    ``append_capped``. It crosses the PyO3 boundary as
    ``Vec<&'static str>`` (``bindings/python/src/lib.rs``), is read back at
    ``protocol/_http1.py``, and is folded into the span's markers by
    ``interceptors/_seam.py::ByteSeamInterceptor._emit_span``.
    """

    GRPC_MESSAGE_TRUNCATED = "grpc_message_truncated"
    """A gRPC message runs past the end of the captured body: the last
    length-prefixed frame is incomplete, so its payload is not the whole message.

    Emitted from ``semantics/_grpc.py::build_grpc_fields`` when
    ``parse_grpc_frames`` (``crates/wardex-protocol/src/grpc.rs``) reports
    ``truncated`` — which it does when fewer than 5 bytes remain for a prefix or
    the declared length runs past the buffer. That parser takes no ``Limits`` at
    all, so the knob upstream of it is the HTTP body cap: ``application/grpc``
    is classified meaningful by ``cap_for_content_type``, so a gRPC body is
    stored up to ``max_body_bytes``, not the opaque cap. NOT
    ``max_decoded_bytes``, which bounds decompression in ``semantic.rs`` and
    never reaches gRPC framing — the semantic parser is not even called on the
    gRPC branch.

    Kept separate from ``BODY_CAP_EXCEEDED`` despite sharing that knob. "The
    body was stored short" and "a message was cut in half" are different facts
    with different replay consequences, and on HTTP/2 — where gRPC actually
    lives — they do not even co-occur: ``http2.rs`` records its truncation on
    the stream and pushes no marker, so ``body_cap_exceeded`` never appears and
    this is the only marker a user gets.
    """

    WS_PAYLOAD_TRUNCATED = "ws_payload_truncated"
    """The WebSocket content sample hit ``ws_sample_bytes`` in either direction,
    so the body bytes on this span are a prefix of the conversation.

    Emitted from ``interceptors/_trackers.py::_WebSocketTracker._build_txn``,
    from the truncation flags ``_append_sample`` sets when the accumulated
    sample passes the cap — resolved from ``limits_defaults()["ws_sample_bytes"]``
    (and passed as ``sample_cap`` by ``interceptors/_seam.py``). NOT
    ``max_ws_frame_bytes``: that one bounds a single frame's declared length in
    ``crates/wardex-protocol/src/websocket.rs`` and an oversize frame does not
    truncate anything — it disables the parser, which surfaces as
    ``FRAME_PARSE_FAILED``.
    """

    CONNECTION_EVICTED = "connection_evicted"
    """The connection table hit ``max_connections`` and this connection's
    tracker was flushed early, so its span ends at the eviction instant.

    Emitted from ``interceptors/_seam.py::ByteSeamInterceptor._state`` via
    ``_WebSocketTracker.flush(marker)``. Until step 3a rewired that site it was
    the free string ``ws_evicted``.

    NOTE (census): renamed, NOT merged into ``UNIT_EVICTED``. Both say
    "something was evicted", but they name different tables and different
    tunables — ``max_connections`` here, ``max_units`` there — and a merged
    marker would send the user to the wrong knob. The rename drops the ``ws_``
    prefix because the connection table is not WebSocket-specific.
    """

    # ------------------------------------------------------------------
    # Parsing / interpretation (census)
    # ------------------------------------------------------------------

    FRAME_PARSE_FAILED = "frame_parse_failed"
    """The framing layer failed, so the transport fields on this span are
    partial or synthesized.

    Emitted from ``semantics/_grpc.py::build_grpc_fields`` (gRPC frame
    parse raised; the span falls back to plain h2 fields) and
    ``interceptors/_trackers.py::_WebSocketTracker._build_txn`` (either
    direction's frame parser latched off). Until step 3a those two sites emitted
    the free strings ``grpc_parse_failed`` and ``ws_parse_failed``.

    This member carries a LIMIT as well as a bug, which its name does not say:
    a WebSocket frame whose declared payload exceeds ``max_ws_frame_bytes``
    makes ``crates/wardex-protocol/src/websocket.rs`` return ``ParseStep::Error``
    and latch ``disabled``, which the tracker reads back as ``ws_parse_failed``.
    So a user who sees this on a WebSocket span has two candidate causes — a
    desynced parser and a frame above the cap — and only the second has a knob.
    ``WS_PAYLOAD_TRUNCATED`` is NOT that knob; see its docstring.

    NOTE (census): merged, and deliberately NOT merged with
    ``SEMANTIC_PARSE_FAILED``. That one is a different layer — framing
    succeeded and the transport fields are valid — and it leads to a different
    follow-up (parser bug vs unsupported provider).
    """

    SEMANTIC_PARSE_FAILED = "semantic_parse_failed"
    """Framing succeeded but the LLM body parser produced no core semantics and
    no output messages, so the span has transport truth and no gen_ai truth.

    Emitted from ``interceptors/_seam.py::ByteSeamInterceptor._emit_span``.
    """

    PAYLOAD_COMPRESSED = "payload_compressed"
    """The payload was observed compressed and wardex did not decompress it, so
    body bytes on this span are not readable content.

    Emitted from ``semantics/_grpc.py::build_grpc_fields`` (any request or
    response message had its compressed flag set) and
    ``interceptors/_trackers.py::_WebSocketTracker._build_txn``
    (permessage-deflate negotiated). Until step 3a those two sites emitted the
    free strings ``grpc_compressed`` and ``ws_compressed``.

    NOTE (census): merged. What is lost is which protocol it was —
    ``TransportAttributes.protocol`` already carries that, and encoding a
    protocol into a marker duplicates a field.
    """

    TOOL_ARGS_UNPARSED = "tool_args_unparsed"
    """A tool call was found in the response but its arguments were not valid
    JSON, so they are carried as an opaque string.

    Emitted from ``interceptors/_seam.py::ByteSeamInterceptor._emit_span``.
    """

    OUTPUT_MESSAGES_UNMAPPED_PART = "output_messages_unmapped_part"
    """``gen_ai.output.messages`` was reconstructed with at least one part the
    mapper did not recognize, so the *response* replay is incomplete.

    Emitted from ``interceptors/_seam.py::ByteSeamInterceptor._emit_span``.
    Deliberately NOT merged with the input-side marker: in a forensic replay,
    an incomplete response and an incomplete prompt support different
    conclusions.
    """

    INPUT_MESSAGES_UNMAPPED_PART = "input_messages_unmapped_part"
    """``gen_ai.input.messages`` was reconstructed with at least one part the
    mapper did not recognize, so the *prompt* replay is incomplete.

    Emitted from ``interceptors/_seam.py::ByteSeamInterceptor._emit_span``.
    """

    # ------------------------------------------------------------------
    # Streaming (census)
    # ------------------------------------------------------------------

    REASSEMBLED_FROM_STREAM = "reassembled_from_stream"
    """The response body on this span is wardex's reassembly of a token stream,
    not a single response the server sent.

    Emitted from ``interceptors/_seam.py::ByteSeamInterceptor._emit_span`` when
    the semantic parser reports a stream and core semantics were recovered.
    """

    STREAM_USAGE_UNAVAILABLE = "stream_usage_unavailable"
    """A stream was reassembled but carried no usage block, so output token
    counts are absent rather than zero.

    Emitted from ``interceptors/_seam.py::ByteSeamInterceptor._emit_span``.
    """

    SSE_UNKNOWN_PROVIDER = "sse_unknown_provider"
    """An SSE stream was detected and reassembled but matched no known provider
    shape, so nothing was mapped out of it.

    Emitted from ``interceptors/_seam.py::ByteSeamInterceptor._emit_span``.
    """

    # ------------------------------------------------------------------
    # Protocol-specific (census)
    # ------------------------------------------------------------------

    GRPC_WEB_UNSUPPORTED = "grpc_web_unsupported"
    """The content type was ``application/grpc-web``, whose framing wardex does
    not parse; the span was left as plain HTTP/2.

    Emitted from ``interceptors/_seam.py::ByteSeamInterceptor._emit_span``.
    """

    GRPC_STATUS_UNAVAILABLE = "grpc_status_unavailable"
    """No ``grpc-status`` was observed — trailers-only response, or trailers the
    seam never saw — so the span's status is derived from HTTP alone.

    Emitted from ``semantics/_grpc.py::build_grpc_fields``.
    """

    WS_NO_CLOSE = "ws_no_close"
    """The WebSocket span was emitted without ever seeing a CLOSE frame, so its
    close code and duration are not trustworthy.

    Emitted from ``interceptors/_ssl.py::SSLInterceptor.uninstall`` and
    ``interceptors/_socket.py::RawSocketInterceptor.uninstall``, both via
    ``_WebSocketTracker.flush(marker)``. Both sites sit inside ``uninstall()``
    today and so travel alongside ``ADAPTER_UNINSTALLED``; see that member for
    why they stay two markers.
    """

    # ------------------------------------------------------------------
    # Adapter session (census)
    # ------------------------------------------------------------------

    SESSION_ABORTED = "session_aborted"
    """The agent session ended without a clean result: an error was raised, or
    teardown arrived with no result message at all.

    Emitted from ``adapters/_assembler.py::SessionAssembler._finalize``, on all
    three of its terminal branches (result present but errored, no result and an
    error, no result and no error).
    """


_BY_VALUE: dict[str, Limitation] = {m.value: m for m in Limitation}
"""Wire value -> member, for `Limitation.from_wire`. See its docstring for why
this is a table rather than a `try: Limitation(value)`."""
