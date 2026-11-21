"""What a limitation marker or a drop reason means, in words a user can act on.

A span in a backend carries `wardex.limitations` values such as
`unit_inferred_sole`, and `wardex.diagnostics().losses` names reasons such as
`queue_overflow`. Both vocabularies are closed, so each name has exactly one
entry here: what happened, what it means for the data, and what to do. The
drop reasons already carry their two sentences in `REASONS`; the markers carry
theirs in `MARKERS` below, written from each member's emit sites. A test holds
both tables to the enums, so a new member without an entry fails the build.

`explain()` is exported as `wardex.explain`; `markdown()` is what
`python -m wardex_sdk explain --all` prints, the source a published list of
limitations is generated from.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import NamedTuple

from ._drop_reasons import REASONS, DropReason
from ._integrity import Limitation


class MarkerText(NamedTuple):
    """One marker's entry: what happened, what it means for the data, what to do."""

    happened: str
    means: str
    todo: str


_L = Limitation
_NOTHING = "Nothing to configure."
_RESERVED = "This release never attaches this marker; it is reserved for a later one."

#: Every limitation marker's user-facing text. Settings are named as a host
#: writes them, and a test holds every `max_*` name here to `LimitsConfig`.
MARKERS: Mapping[Limitation, MarkerText] = MappingProxyType(
    {
        _L.PARENT_UNRESOLVED: MarkerText(
            "wardex expected this span to have a parent, but no live span, framework id or "
            "trace header named one, so the span starts its own trace.",
            "The span itself is real; its place in the tree is missing, so it can show up as a "
            "separate trace instead of under the run that caused it.",
            "Call init() before the framework starts its run, open the work inside a wardex "
            "span (wardex.span(), @wardex.agent), and pass trace headers across process "
            "boundaries (wardex.get_trace_headers(), wardex.continue_trace()).",
        ),
        _L.UNIT_INFERRED_SOLE: MarkerText(
            "Nothing live said which run this call belongs to, and exactly one run was live, "
            "so wardex attached the call to that run.",
            "The parent is a deduction, recorded with wardex.parent_confidence 0.5: right "
            "whenever only that run could have made the call.",
            "Usually nothing. To make the edge certain, make the call inside the framework's "
            "own run, or inside a wardex span.",
        ),
        _L.CORRELATION_CONFLICT: MarkerText(
            "Two sources disagreed about this span's parent: a framework id pointed into one "
            "trace while the live context pointed elsewhere, or the context still named a run "
            "that had already ended.",
            "wardex kept the live context's answer and marked the disagreement, so the edge "
            "may be wrong.",
            "Look for work that outlives its run (a task started inside a run and still "
            "running after it ended). If there is none, report it with init(debug=True) output.",
        ),
        _L.ALIAS_FORGOTTEN: MarkerText(
            "The framework id this span's parent was looked up by had been dropped from a "
            "bounded table (max_entries_per_unit), so the parent came from a less specific "
            "source.",
            "The span may hang off its run instead of the exact step that caused it.",
            "Raise LimitsConfig(max_entries_per_unit=...) for runs with very many steps.",
        ),
        _L.CONTEXT_PROPAGATION_DEGRADED: MarkerText(
            _RESERVED,
            "It would mean the span's context reached it only in part (a thread pool that "
            "copied the context at submit time), so its parent edge may be incomplete.",
            _NOTHING,
        ),
        _L.CONTEXT_PROPAGATION_UNAVAILABLE: MarkerText(
            _RESERVED,
            "It would mean the runtime cannot carry context to this span at all, so every "
            "span below it starts its own trace.",
            _NOTHING,
        ),
        _L.UNIT_EVICTED: MarkerText(
            "A run or session was still open when wardex's table of them was full "
            "(max_units, or max_sessions for Claude Agent SDK sessions), so wardex closed it "
            "early to make room. On a new root it means the run continues there after its "
            "first part was closed that way.",
            "The span ends at the eviction instant, not when the run ended, and the run's "
            "later work may sit under a second root.",
            "Raise LimitsConfig(max_units=...) or LimitsConfig(max_sessions=...), or end runs "
            "you no longer use.",
        ),
        _L.UNIT_INTERRUPTED: MarkerText(
            "The run was cut off by a shutdown (a signal, wardex.close() or interpreter exit) "
            "before it finished.",
            "The span ends at the shutdown; the run's real outcome was never observed.",
            "Let runs finish, or await them, before the process exits.",
        ),
        _L.CHILD_SPAN_UNCLOSED: MarkerText(
            "This span's own end never came: its parent was torn down or evicted while it was "
            "open, so wardex closed it then.",
            "Its end time is the teardown's and its status was not observed.",
            "Nothing to configure here; the parent's own marker (unit_evicted, "
            "unit_interrupted) says why it ended early.",
        ),
        _L.UNIT_TABLE_FULL: MarkerText(
            "A run had more open children than max_entries_per_unit, so wardex closed this "
            "one, the oldest, early to admit a new one.",
            "Its end time is when it was pushed out, not when the work finished.",
            "Raise LimitsConfig(max_entries_per_unit=...).",
        ),
        _L.SESSION_ENTRY_TABLE_FULL: MarkerText(
            "A Claude Agent SDK session had more open tool calls, sub-agents or model "
            "responses than max_session_entries, so wardex closed the oldest early.",
            "The span ends when it was pushed out, with status UNSET. A call that completes "
            "later ships a second span with the same call id, also marked: leave marked UNSET "
            "spans out of latency totals so the call is not counted twice.",
            "Raise LimitsConfig(max_session_entries=...).",
        ),
        _L.ADAPTER_UNINSTALLED: MarkerText(
            "The span was still open when wardex uninstalled its adapter (wardex.close(), a "
            "second init(), or interpreter exit), so wardex closed it.",
            "Its end time is the uninstall's, not the framework's.",
            "Call wardex.close() after the work it should cover has finished.",
        ),
        _L.PATCH_SUPERSEDED: MarkerText(
            "Another library replaced a function wardex had patched, so calls went through "
            "that library's patch instead of wardex's.",
            "What wardex's patch would have recorded after the replacement is missing.",
            "Call wardex.init() after the other library installs its patches, or report the "
            "conflict.",
        ),
        _L.INSTRUMENTATION_DEGRADED: MarkerText(
            "wardex's own instrumentation failed here: an adapter or a parser raised, a "
            "worker could not start, a session ended before the CLI's last telemetry, or a "
            "Claude Agent SDK session started without wardex's hooks.",
            "Something that belongs on this span, or below it, is missing because of wardex, "
            "not because the work did not happen.",
            "Re-run with init(debug=True) for the traceback and report it. For the Claude "
            "Agent SDK, call wardex.init() before creating the client or importing query.",
        ),
        _L.TRACKING_RESET_AT_FORK: MarkerText(
            "The connection this span used was opened before os.fork(), and the child process "
            "started tracking it from scratch.",
            "Parsing may have started mid-stream, and timing counts from the fork, not from "
            "the connect.",
            "Open connections after forking when exact data matters.",
        ),
        _L.NO_WIRE_EVIDENCE: MarkerText(
            "The span was built from the framework's own account (the prompt and answer of a "
            "codex exec run), not from bytes wardex saw on the network.",
            "Payloads are what the framework reported; instructions it adds around the prompt "
            "are not included.",
            _NOTHING,
        ),
        _L.POSSIBLE_DUPLICATE_CHAT: MarkerText(
            _RESERVED,
            "It would mean an adapter's chat span may also have been recorded by the network "
            "capture.",
            _NOTHING,
        ),
        _L.STREAM_BUFFER_EXCEEDED: MarkerText(
            _RESERVED,
            "It would mean an MCP stdio stream outgrew max_stream_buffer_bytes and its parser "
            "stopped mid-stream.",
            _NOTHING,
        ),
        _L.TOOL_CALL_ID_UNAVAILABLE_IN_PROCESS: MarkerText(
            "The framework ran this tool in-process and did not hand over the tool-call id "
            "(for OpenAI Agents: the call could not be matched to exactly one request).",
            "The tool span cannot be joined by id to the model message that asked for it.",
            _NOTHING,
        ),
        _L.SNAPSHOT_TYPE_UNKNOWN: MarkerText(
            "capture_state_snapshot() was given a type that is not a SnapshotType, so the "
            "snapshot was recorded as UNSPECIFIED.",
            "The snapshot is kept; its type is not.",
            "Pass a wardex.SnapshotType member.",
        ),
        _L.TOOL_NAME_COLLISION: MarkerText(
            "Two MCP servers expose tools whose names become the same name the Claude CLI "
            "sees, so this span's name cannot say which server ran the call.",
            "The call may be recorded twice, or this span may be its only record.",
            "Give the servers' tools distinct names.",
        ),
        _L.LINK_AMBIGUOUS: MarkerText(
            "LangGraph named this step's trigger in a form with more than one reading (a node "
            "name containing '+'), so wardex did not add the link.",
            "The causal link to the nodes that triggered this step is missing.",
            "Use node names without '+'.",
        ),
        _L.CONNECT_TIMING_UNAVAILABLE: MarkerText(
            "wardex saw no timed TCP connect for this connection (none was observed, or an "
            "asynchronous connect could not be timed on its own), so tcp_connect_ms is unset.",
            "The connect time is unknown, not zero.",
            _NOTHING,
        ),
        _L.TTFT_UNAVAILABLE_H2: MarkerText(
            "The streamed response came over HTTP/2, where wardex sees data frames but not "
            "the stream's event boundaries, so time-to-first-token is absent.",
            "Time-to-first-token is unknown, not zero.",
            _NOTHING,
        ),
        _L.TTFT_IPC_APPROXIMATION: MarkerText(
            "Time-to-first-token was measured where the agent CLI's output reached this "
            "process, not on the network.",
            "It includes the subprocess's and the pipe's latency.",
            _NOTHING,
        ),
        _L.TRANSPORT_TIMING_UNAVAILABLE_SUBPROCESS: MarkerText(
            "The model call ran inside an agent CLI subprocess; wardex saw only its output "
            "stream, not the network.",
            "The span has no connection or network timing.",
            _NOTHING,
        ),
        _L.PEER_UNRESOLVED: MarkerText(
            "wardex could not read the peer's host and port (a unix socket, or a TLS layer "
            "with no socket address), so server.address is the TLS server name or 'unknown' "
            "and server.port is 0.",
            "The call was captured; only its address is a placeholder.",
            _NOTHING,
        ),
        _L.BODY_CAP_EXCEEDED: MarkerText(
            "An HTTP body was longer than wardex keeps of one, so it kept the first "
            "max_body_bytes (max_opaque_body_bytes for a content type wardex does not parse, "
            "and for a JSON request outside any span until its reply showed a model call), or "
            "what the shared max_in_flight_body_bytes had room for, and let the rest pass "
            "unrecorded.",
            "The payload is a prefix; anything parsed from the rest of it is missing.",
            "Raise LimitsConfig(max_body_bytes=...), LimitsConfig(max_opaque_body_bytes=...), "
            "or LimitsConfig(max_in_flight_body_bytes=...) when many bodies arrive at once.",
        ),
        _L.GRPC_MESSAGE_TRUNCATED: MarkerText(
            "The last gRPC message in the captured body was incomplete, usually because the "
            "body was cut at max_body_bytes.",
            "That message's payload is not whole.",
            "Raise LimitsConfig(max_body_bytes=...).",
        ),
        _L.WS_PAYLOAD_TRUNCATED: MarkerText(
            "The WebSocket content sample reached ws_sample_bytes.",
            "The payload on this span is the start of the conversation, not all of it.",
            "Raise LimitsConfig(ws_sample_bytes=...).",
        ),
        _L.CONNECTION_EVICTED: MarkerText(
            "wardex's connection table reached max_connections, so tracking of this "
            "connection ended early.",
            "The span ends at that moment; a WebSocket session goes on unwatched, with no "
            "length and no sizes.",
            "Raise LimitsConfig(max_connections=...).",
        ),
        _L.H2_REQUEST_EVICTED: MarkerText(
            "The HTTP/2 stream table reached max_streams and dropped this stream's request "
            "half before its response finished.",
            "Method, path and request body are missing (shown as '? /'); status and response "
            "were observed.",
            "Raise LimitsConfig(max_streams=...).",
        ),
        _L.PARSE_SKIPPED_AT_SHUTDOWN: MarkerText(
            "The process was ending and its shutdown budget ran out before wardex parsed this "
            "response's model-call fields.",
            "Transport, timing and status were measured; the gen_ai fields are absent.",
            "Give shutdown more time: wardex.close(timeout=...) or "
            "BatchingConfig(shutdown_timeout=...).",
        ),
        _L.OTLP_ATTRIBUTE_TRUNCATED: MarkerText(
            "An attribute value was longer than max_otlp_attribute_bytes on export, so the "
            "backend shows its start only. When the span also carries "
            "wardex.payload_dropped.reason = max_otlp_request_bytes, its whole payload was "
            "removed so the span fits in one request.",
            "What the backend shows is a prefix of what wardex captured, or no payload at all.",
            "Raise LimitsConfig(max_otlp_attribute_bytes=...), or "
            "LimitsConfig(max_otlp_request_bytes=...) for a dropped payload, if the backend "
            "accepts more.",
        ),
        _L.EXTRA_KEYS_DROPPED: MarkerText(
            "A provider's usage object had more fields than max_extra_keys, so some "
            "wardex.usage.* fields were left out; wardex.usage_leaves.dropped_count says how "
            "many.",
            "Some provider-specific usage counters are missing.",
            "Raise LimitsConfig(max_extra_keys=...).",
        ),
        _L.PARSE_BACKLOG_FULL: MarkerText(
            "Too many responses were waiting to be parsed (max_parse_backlog, "
            "max_parse_backlog_bytes), so this one shipped without being parsed.",
            "Transport, timing and status were measured; the gen_ai fields are absent, and "
            "the raw bodies may be withheld.",
            "Raise LimitsConfig(max_parse_backlog=...) or "
            "LimitsConfig(max_parse_backlog_bytes=...).",
        ),
        _L.FRAME_PARSE_FAILED: MarkerText(
            "wardex could not read this exchange's framing: gRPC or WebSocket frames, or an "
            "HTTP/1 response whose stream ended before its declared length.",
            "The transport fields are partial: the body is what arrived, its size may be "
            "unset, and a 2xx status is UNSET. On a WebSocket, a frame over "
            "max_ws_frame_bytes also causes it.",
            "On HTTP, look at the server or network that ended the response early. On a "
            "WebSocket with large frames, raise LimitsConfig(max_ws_frame_bytes=...).",
        ),
        _L.SEMANTIC_PARSE_FAILED: MarkerText(
            "The body was read, but wardex could not understand it as a model call or as an "
            "MCP tool result it could vouch for.",
            "The span has transport data and no gen_ai fields.",
            "If the provider should be supported, report it with init(debug=True) output.",
        ),
        _L.PAYLOAD_COMPRESSED: MarkerText(
            "The payload was compressed (gRPC message compression, or WebSocket "
            "permessage-deflate) and wardex does not decompress it.",
            "The body bytes on this span are not readable content.",
            _NOTHING,
        ),
        _L.TOOL_ARGS_UNPARSED: MarkerText(
            "A tool call's arguments were not valid JSON.",
            "They are kept as one opaque string.",
            _NOTHING,
        ),
        _L.OUTPUT_MESSAGES_UNMAPPED_PART: MarkerText(
            "The model's response held a part wardex does not map, so gen_ai.output.messages "
            "leaves it out.",
            "The response, as recorded, is incomplete.",
            "Upgrade wardex-sdk, and report it if it persists.",
        ),
        _L.INPUT_MESSAGES_UNMAPPED_PART: MarkerText(
            "The prompt held a part wardex does not map, so gen_ai.input.messages leaves it out.",
            "The prompt, as recorded, is incomplete.",
            "Upgrade wardex-sdk, and report it if it persists.",
        ),
        _L.PROVIDER_INFERRED: MarkerText(
            "gen_ai.provider.name was inferred from a host name that merely contains the "
            "provider's name (a proxy, a mock, an Azure deployment), or from the shape of the "
            "body or API, not read off the provider's own host.",
            "The provider label may be wrong.",
            _NOTHING,
        ),
        _L.REASSEMBLED_FROM_STREAM: MarkerText(
            "The response was streamed, and the body on this span is wardex's reassembly of "
            "the stream's events.",
            "It is not one response the server sent as such.",
            _NOTHING,
        ),
        _L.STREAM_USAGE_UNAVAILABLE: MarkerText(
            "The streamed response carried no usage block.",
            "Token counts are absent, not zero.",
            "If the provider can include usage in a stream, ask it to.",
        ),
        _L.SSE_UNKNOWN_PROVIDER: MarkerText(
            "An event stream was reassembled but matched no provider format wardex knows.",
            "No model-call fields were read from it.",
            "If the provider should be supported, report it.",
        ),
        _L.SSE_EVENT_UNPARSED: MarkerText(
            "An event in the model's stream was not valid JSON (cut short, or not JSON at "
            "all), so wardex skipped it.",
            "Whatever that event carried may be missing from the span.",
            "If the span also carries body_cap_exceeded, raise LimitsConfig(max_body_bytes=...).",
        ),
        _L.GRPC_WEB_UNSUPPORTED: MarkerText(
            "The exchange used gRPC-Web, whose framing wardex does not parse.",
            "It is recorded as plain HTTP/2.",
            _NOTHING,
        ),
        _L.GRPC_STATUS_UNAVAILABLE: MarkerText(
            "No grpc-status was seen (a trailers-only response, or trailers wardex did not "
            "observe).",
            "The span's status comes from HTTP alone.",
            _NOTHING,
        ),
        _L.WS_NO_CLOSE: MarkerText(
            "The WebSocket span was emitted without a CLOSE frame: the socket closed, or "
            "wardex.close() stopped watching it.",
            "There is no close code.",
            _NOTHING,
        ),
        _L.WS_LLM_SEMANTICS_UNREAD: MarkerText(
            "This WebSocket carried model calls (the OpenAI Responses API over WebSocket), "
            "and wardex does not parse those events.",
            "The span has transport counts and payload samples, but no model, tokens or messages.",
            "Use the framework's HTTP transport for full spans; for openai-agents that is "
            "use_responses_websocket=False, the default.",
        ),
        _L.SESSION_ABORTED: MarkerText(
            "The agent session ended without a clean result: it raised, or ended with no "
            "result message.",
            "The run did not complete normally.",
            "Look at the run's own error; there is nothing to configure in wardex.",
        ),
        _L.OTEL_BRIDGE_NO_DATA: MarkerText(
            "The OTel bridge configured the agent CLI's telemetry, confirmed the setting "
            "reached the subprocess, and still received nothing before the session ended.",
            "The tree is what wardex records without the bridge.",
            "Check the CLI's version, and any policy that strips a subprocess's environment "
            "or blocks loopback connections.",
        ),
        _L.OTEL_BRIDGE_SCHEMA_UNKNOWN: MarkerText(
            "Bridge telemetry arrived for this session but held nothing the bridge recognizes.",
            "The tree is what wardex records without the bridge.",
            "Upgrade wardex-sdk; the CLI's telemetry format may have changed.",
        ),
        _L.OTEL_BRIDGE_JOIN_TOLERANT: MarkerText(
            "This chat span took its timing from a CLI request that matched it only within "
            "a one-second tolerance.",
            "Read the timing as an estimate: it may be a neighbouring request's.",
            _NOTHING,
        ),
        _L.SUBPROCESS_MODEL_CALLS_UNOBSERVED: MarkerText(
            "The agent CLI (codex exec) reported one usage total per turn and no model calls, "
            "so no chat spans were recorded; the run's span carries the total.",
            "Per-call model, usage and timing are missing.",
            "Pass AdaptersConfig(codex_exec=CodexExecConfig(otel_bridge=True)) to record each "
            "call from the CLI's own telemetry.",
        ),
        _L.PII_FILTER_FAILED: MarkerText(
            "PII masking failed on this span or snapshot, so wardex removed its content "
            "rather than ship it unmasked.",
            "Only ids, kind and times remain, named [PII_FILTER_ERROR].",
            "This is a wardex bug; report it.",
        ),
    }
)

_NATURE = {
    "loss": "What it counted is gone.",
    "partial": "What it counted shipped, missing part of itself.",
    "none": "Nothing was lost: the count records a choice made on purpose.",
}

_MARKER_OF = {m.value: m for m in Limitation}
_REASON_OF = {m.value: m for m in DropReason}


def _marker_paragraph(member: Limitation) -> str:
    text = MARKERS[member]
    return (
        f"{member.value} (a limitation marker on a span, in wardex.limitations): "
        f"{text.happened} {text.means} {text.todo}"
    )


def _reason_paragraph(member: DropReason) -> str:
    info = REASONS[member]
    return (
        f"{member.value} (a loss reason in wardex.diagnostics().losses, counted per "
        f"{info.unit.value} by default): {info.meaning} {_NATURE[info.loss]} {info.next_step}"
    )


def explain(name: str) -> str:
    """What the limitation marker or drop reason `name` means, and what to do.

    `name` is a marker as a backend shows it (`unit_inferred_sole`), a loss
    reason (`queue_overflow`), or a loss's wire key (`queue_overflow:span`).
    A name that is both a marker and a reason gets both paragraphs. An unknown
    name gets a sentence saying so, never an exception: this runs in the
    process wardex observes.
    """
    key = str(name).strip().lower()
    key = key.partition(":")[0] if key not in _MARKER_OF else key
    found = []
    if key in _MARKER_OF:
        found.append(_marker_paragraph(_MARKER_OF[key]))
    if key in _REASON_OF:
        found.append(_reason_paragraph(_REASON_OF[key]))
    if not found:
        return (
            f"{name!r} is not a wardex limitation marker or loss reason. "
            "`python -m wardex_sdk explain --all` lists every one."
        )
    return "\n\n".join(found)


def markdown() -> str:
    """Every marker and every loss reason, as one Markdown document."""
    lines = [
        "# wardex limitation markers and loss reasons",
        "",
        "Generated by `python -m wardex_sdk explain --all`. A limitation marker is a value "
        "of a span's `wardex.limitations` attribute: the span shipped, and this is what it "
        "is missing. A loss reason is a key of `wardex.diagnostics().losses`: something "
        "wardex did not record, counted.",
        "",
        "## Limitation markers",
        "",
    ]
    for member in Limitation:
        text = MARKERS[member]
        lines += [
            f"### `{member.value}`",
            "",
            f"- **What happened:** {text.happened}",
            f"- **What it means for the data:** {text.means}",
            f"- **What to do:** {text.todo}",
            "",
        ]
    lines += ["## Loss reasons", ""]
    for reason in DropReason:
        info = REASONS[reason]
        lines += [
            f"### `{reason.value}`",
            "",
            f"- **What happened:** {info.meaning}",
            f"- **What it means for the data:** {_NATURE[info.loss]} Counted per "
            f"`{info.unit.value}` by default.",
            f"- **What to do:** {info.next_step}",
            "",
        ]
    return "\n".join(lines).rstrip("\n") + "\n"
