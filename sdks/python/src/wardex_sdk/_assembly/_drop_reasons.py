"""The drop ledger's closed vocabulary: why something was lost, what was counted.

Kept apart from the ledger's mechanism (`_ledger`) because it is data, and the
largest part of it: one row per reason, with the unit it counts by default,
whether it is a loss at all, and the two sentences a reader is shown. Callers
import all of it from `_ledger`, which re-exports it; this module is where a
new reason is added, with its row, before anything records it.

The wire spelling is `"<reason>:<unit>"` from the members' `.value`s. Never
format a member with an f-string directly: a `str` mixin enum renders as
`DropReason.X` there on newer Pythons, not as its value.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Literal


class DropUnit(str, Enum):
    """What a count counts. The unit is what makes one reason serve several
    tables: `table_full:session` and `table_full:stream` call for different
    knobs."""

    SPAN = "span"
    SNAPSHOT = "snapshot"
    REQUEST = "request"
    CONNECTION = "connection"
    STREAM = "stream"
    SESSION = "session"
    RUN = "run"
    MESSAGE = "message"
    BRIDGE_SPAN = "bridge_span"
    BRIDGE_SESSION = "bridge_session"
    ENTRY = "entry"
    LINK = "link"
    HOOK_EVENT = "hook_event"
    PAYLOAD = "payload"
    MARKER = "marker"
    ATTRIBUTE = "attribute"
    PROCESS = "process"
    SEAM = "seam"
    ADAPTER = "adapter"
    FAILURE = "failure"
    REPORT = "report"


class DropReason(str, Enum):
    """Why something was lost. Closed: a new reason is added here, with its
    `REASONS` row, before anything records it. The wire value is `.value`."""

    NETWORK_ERROR = "network_error"
    SEND_ERROR = "send_error"
    RATELIMIT_BACKOFF = "ratelimit_backoff"
    SEND_UNCONFIRMED = "send_unconfirmed"
    REQUEST_TOO_LARGE = "request_too_large"
    BUDGET_EXHAUSTED = "budget_exhausted"
    PARTIAL_SUCCESS = "partial_success"
    UNMARSHALLABLE = "unmarshallable"
    TRANSPORT_UNSUPPORTED = "transport_unsupported"
    QUEUE_OVERFLOW = "queue_overflow"
    CACHE_OVERFLOW = "cache_overflow"
    BEFORE_SEND = "before_send"
    BEFORE_SEND_ERROR = "before_send_error"
    AFTER_CLOSE = "after_close"
    SHUTDOWN_TAIL = "shutdown_tail"
    BACKPRESSURE = "backpressure"
    INTERNAL_SDK_ERROR = "internal_sdk_error"
    DIAGNOSTIC_UNWRITABLE = "diagnostic_unwritable"
    LEDGER_OVERFLOW = "ledger_overflow"
    MODE_REFUSED = "mode_refused"
    EXCLUDED_BY_POLICY = "excluded_by_policy"
    PARSER_DISABLED = "parser_disabled"
    RESPONSE_UNOBSERVED = "response_unobserved"
    REQUEST_UNOBSERVED = "request_unobserved"
    CONNECTION_EVICTED = "connection_evicted"
    IN_FLIGHT_AT_CLOSE = "in_flight_at_close"
    UNCLASSIFIED_CONNECTION = "unclassified_connection"
    TABLE_FULL = "table_full"
    LATE_TO_CLOSED_UNIT = "late_to_closed_unit"
    MARKER_LOST = "marker_lost"
    LINK_LOST = "link_lost"
    ATTRIBUTE_DROPPED = "attribute_dropped"
    INSTRUMENTATION_UNAVAILABLE = "instrumentation_unavailable"
    PATCH_SUPERSEDED = "patch_superseded"
    CHILD_UNINITIALIZED = "child_uninitialized"
    ATTRIBUTE_CAP = "attribute_cap"
    BODY_CAP = "body_cap"
    WS_SAMPLE_CAP = "ws_sample_cap"
    WS_FRAME_CAP = "ws_frame_cap"
    DECODED_CAP = "decoded_cap"
    BRIDGE_LATE = "bridge_late"
    BRIDGE_UNROUTABLE = "bridge_unroutable"
    BRIDGE_REQUEST_REJECTED = "bridge_request_rejected"
    BRIDGE_UNRECOGNIZED = "bridge_unrecognized"
    BRIDGE_INCOMPLETE = "bridge_incomplete"
    BRIDGE_RETIRED = "bridge_retired"
    BRIDGE_MERGE_SKIPPED = "bridge_merge_skipped"
    BRIDGE_UNAVAILABLE = "bridge_unavailable"
    FRAMEWORK_TRACING_DISABLED = "framework_tracing_disabled"
    FRAMEWORK_EVENT_UNMATCHED = "framework_event_unmatched"
    FRAMEWORK_SPAN_KIND_UNMAPPED = "framework_span_kind_unmapped"
    HOOKS_ABSENT = "hooks_absent"
    STREAM_LINE_UNRECOGNIZED = "stream_line_unrecognized"
    PAYLOAD_UNSERIALIZABLE = "payload_unserializable"
    USAGE_WITHHELD = "usage_withheld"
    LAUNCH_UNRECOGNIZED = "launch_unrecognized"


#: "loss": the thing is gone. "partial": it shipped, missing part of itself.
#: "none": nothing was lost; the count says who decided it should not ship.
Loss = Literal["loss", "partial", "none"]


@dataclass(frozen=True)
class ReasonInfo:
    """One reason's row: the unit it counts by default, whether it is a loss,
    and the two sentences a reader is shown — what happened, and what to do."""

    unit: DropUnit
    loss: Loss
    meaning: str
    next_step: str


_R = DropReason
_U = DropUnit

#: Every reason's data, in one place. Diagnostics, strict mode and the close
#: summary print these sentences as they are.
REASONS: Mapping[DropReason, ReasonInfo] = MappingProxyType(
    {
        _R.NETWORK_ERROR: ReasonInfo(
            _U.SPAN,
            "loss",
            "The batch never reached the receiver: the connection was refused, the name did "
            "not resolve, or connecting timed out.",
            "Check that the receiver's address is reachable from this process.",
        ),
        _R.SEND_ERROR: ReasonInfo(
            _U.SPAN,
            "loss",
            "The receiver refused the batch for good (a 4xx, or a 5xx not worth retrying), or "
            "the transport raised while sending it.",
            "Check the project key and the receiver's status; a transport that raises is a bug.",
        ),
        _R.RATELIMIT_BACKOFF: ReasonInfo(
            _U.SPAN,
            "loss",
            "The receiver asked wardex to back off (HTTP 429, or a wardex receiver's 503) and "
            "the batch was not sent.",
            "Export less often (raise batching.flush_interval) or ask for a higher limit.",
        ),
        _R.SEND_UNCONFIRMED: ReasonInfo(
            _U.SPAN,
            "loss",
            "Whether the batch arrived is unknown: no answer came back (a read timeout, a "
            "connection cut before the answer, a 502/504, or a budget that ran out mid-request).",
            "Give flush()/close() a larger timeout, or check the network path to the receiver.",
        ),
        _R.REQUEST_TOO_LARGE: ReasonInfo(
            _U.SPAN,
            "loss",
            "Too large to send: a request over the receiver's size limit, or a single span "
            "over max_otlp_request_bytes even without its payload.",
            "Raise max_otlp_request_bytes if the receiver accepts more.",
        ),
        _R.BUDGET_EXHAUSTED: ReasonInfo(
            _U.REQUEST,
            "loss",
            "A batch split into several requests ran out of time before the rest were sent.",
            "Give flush()/close() a larger timeout, or lower limits.max_buffer_spans.",
        ),
        _R.PARTIAL_SUCCESS: ReasonInfo(
            _U.SPAN,
            "loss",
            "The OTLP backend accepted the request but rejected some of its spans.",
            "Read the backend's own logs for why those spans were rejected.",
        ),
        _R.UNMARSHALLABLE: ReasonInfo(
            _U.SPAN,
            "loss",
            "The span held a value of the wrong type and could not be encoded.",
            "Re-run with debug=True to see which span and value.",
        ),
        _R.TRANSPORT_UNSUPPORTED: ReasonInfo(
            _U.SNAPSHOT,
            "loss",
            "The configured transport cannot carry this kind of item (OTLP carries no state "
            "snapshots).",
            "Send to a wardex receiver to keep state snapshots.",
        ),
        _R.QUEUE_OVERFLOW: ReasonInfo(
            _U.SPAN,
            "loss",
            "The buffer was full, so the oldest items were evicted before they were exported.",
            "Raise limits.max_buffer_spans / limits.max_buffer_bytes, or export more often.",
        ),
        _R.CACHE_OVERFLOW: ReasonInfo(
            _U.REQUEST,
            "loss",
            "The queue holding batches for a retry was full, so a held batch was dropped.",
            "Raise the retry queue's bound, or restore the receiver sooner.",
        ),
        _R.BEFORE_SEND: ReasonInfo(
            _U.SPAN,
            "none",
            "before_send_envelope returned None: the host dropped the batch on purpose.",
            "Nothing to do unless the filter drops more than intended.",
        ),
        _R.BEFORE_SEND_ERROR: ReasonInfo(
            _U.SPAN,
            "loss",
            "before_send_envelope raised, so the batch was dropped unsent.",
            "Fix the hook; re-run with debug=True for its traceback.",
        ),
        _R.AFTER_CLOSE: ReasonInfo(
            _U.SPAN,
            "loss",
            "Captured after close(), when nothing would ever export it.",
            "Call close() after the work it should cover, or init() again before more work.",
        ),
        _R.SHUTDOWN_TAIL: ReasonInfo(
            _U.SPAN,
            "loss",
            "Still buffered when close() or a shutdown signal ran out of time to send it.",
            "Give close() a larger budget (batching.shutdown_timeout), or flush() before exit.",
        ),
        _R.BACKPRESSURE: ReasonInfo(
            _U.SPAN,
            "partial",
            "The span shipped without its parsed gen_ai fields: the parse queue was full.",
            "Raise the parse queue's bound, or reduce the burst of model calls.",
        ),
        _R.INTERNAL_SDK_ERROR: ReasonInfo(
            _U.FAILURE,
            "loss",
            "wardex raised inside itself and swallowed it; what that step handled may be lost.",
            "Re-run with debug=True for the traceback, and report it.",
        ),
        _R.DIAGNOSTIC_UNWRITABLE: ReasonInfo(
            _U.FAILURE,
            "loss",
            "A wardex warning could not be written: stderr or a host log handler failed.",
            "Check stderr and any handler attached to the wardex_sdk logger.",
        ),
        _R.LEDGER_OVERFLOW: ReasonInfo(
            _U.REPORT,
            "loss",
            "More drop reports awaited confirmation than the ledger holds; the oldest was dropped.",
            "Restore the receiver; repeated send_unconfirmed counts say why reports piled up.",
        ),
        _R.MODE_REFUSED: ReasonInfo(
            _U.REQUEST,
            "loss",
            "capture_mode=AGENT declined an exchange outside any span that it did not "
            "recognize as a model call.",
            "Open a span around the work, or use capture_mode=ALL.",
        ),
        _R.EXCLUDED_BY_POLICY: ReasonInfo(
            _U.REQUEST,
            "none",
            "Left out by design (a framework's own telemetry upload).",
            "Nothing to do.",
        ),
        _R.PARSER_DISABLED: ReasonInfo(
            _U.CONNECTION,
            "loss",
            "A protocol parser stopped reading a connection or stream; what followed on it "
            "was not captured.",
            "Re-run with debug=True to see which connection and why.",
        ),
        _R.RESPONSE_UNOBSERVED: ReasonInfo(
            _U.REQUEST,
            "partial",
            "The request was seen but no response was (a timeout, a cancellation, a reset).",
            "Nothing to configure; the request half is what the network gave.",
        ),
        _R.REQUEST_UNOBSERVED: ReasonInfo(
            _U.REQUEST,
            "partial",
            "A response was seen whose request was not (a connection opened before init()).",
            "Call init() before opening connections.",
        ),
        _R.CONNECTION_EVICTED: ReasonInfo(
            _U.REQUEST,
            "loss",
            "The connection table was full, so an exchange in progress was evicted.",
            "Raise limits.max_connections.",
        ),
        _R.IN_FLIGHT_AT_CLOSE: ReasonInfo(
            _U.REQUEST,
            "loss",
            "An exchange was still in progress when close() or a re-init uninstalled capture.",
            "Close after in-flight requests finish.",
        ),
        _R.UNCLASSIFIED_CONNECTION: ReasonInfo(
            _U.CONNECTION,
            "loss",
            "The first bytes seen on a connection were a response, so it was never tracked.",
            "Call init() before opening connections.",
        ),
        _R.TABLE_FULL: ReasonInfo(
            _U.ENTRY,
            "loss",
            "A bounded table was full and evicted an entry; the unit says which table.",
            "Raise the bound for that table.",
        ),
        _R.LATE_TO_CLOSED_UNIT: ReasonInfo(
            _U.PAYLOAD,
            "loss",
            "Data arrived for a unit that had already closed or been evicted.",
            "Raise limits.max_units if units are evicted; otherwise close later.",
        ),
        _R.MARKER_LOST: ReasonInfo(
            _U.MARKER,
            "loss",
            "A limitation marker could not be attached, so a span looks more complete than it is.",
            "Report it; this is wardex's own bookkeeping failing.",
        ),
        _R.LINK_LOST: ReasonInfo(
            _U.LINK,
            "loss",
            "A causal edge (a trigger, a handoff, a traceparent) could not be kept.",
            "Re-run with debug=True to see which edge.",
        ),
        _R.ATTRIBUTE_DROPPED: ReasonInfo(
            _U.ATTRIBUTE,
            "loss",
            "An attribute set under a reserved name was replaced or left out.",
            "Rename the attribute.",
        ),
        _R.INSTRUMENTATION_UNAVAILABLE: ReasonInfo(
            _U.SEAM,
            "loss",
            "An adapter, seam or patch is not installed, so what it would capture is absent.",
            "Read wardex.diagnostics() for which one and why.",
        ),
        _R.PATCH_SUPERSEDED: ReasonInfo(
            _U.SEAM,
            "loss",
            "Another library replaced a wardex patch, and calls went around it.",
            "Call init() after that library patches, or report the conflict.",
        ),
        _R.CHILD_UNINITIALIZED: ReasonInfo(
            _U.PROCESS,
            "loss",
            "A spawned child process did not receive the configuration, so it records nothing.",
            "Call init() in the child, or make the configuration picklable.",
        ),
        _R.ATTRIBUTE_CAP: ReasonInfo(
            _U.SPAN,
            "partial",
            "An attribute was cut at max_otlp_attribute_bytes.",
            "Raise max_otlp_attribute_bytes if the backend accepts more.",
        ),
        _R.BODY_CAP: ReasonInfo(
            _U.SPAN,
            "partial",
            "A body was cut at its capture limit.",
            "Raise the body capture limit.",
        ),
        _R.WS_SAMPLE_CAP: ReasonInfo(
            _U.SPAN,
            "partial",
            "A WebSocket message was cut at the sample limit.",
            "Raise the WebSocket sample limit.",
        ),
        _R.WS_FRAME_CAP: ReasonInfo(
            _U.SPAN,
            "partial",
            "A WebSocket frame over the frame limit stopped the parser for that direction.",
            "Raise the WebSocket frame limit.",
        ),
        _R.DECODED_CAP: ReasonInfo(
            _U.SPAN,
            "partial",
            "A compressed body was cut at the decoded-size limit.",
            "Raise the decoded body limit.",
        ),
        _R.BRIDGE_LATE: ReasonInfo(
            _U.BRIDGE_SPAN,
            "loss",
            "An OTel bridge batch arrived after its run had already been assembled.",
            "Nothing to configure; report it if it recurs.",
        ),
        _R.BRIDGE_UNROUTABLE: ReasonInfo(
            _U.BRIDGE_SPAN,
            "loss",
            "An OTel bridge span could not be matched to any run.",
            "Report it with the debug=True output.",
        ),
        _R.BRIDGE_REQUEST_REJECTED: ReasonInfo(
            _U.REQUEST,
            "loss",
            "The OTel bridge receiver refused or could not decode a POST.",
            "Re-run with debug=True to see why.",
        ),
        _R.BRIDGE_UNRECOGNIZED: ReasonInfo(
            _U.BRIDGE_SPAN,
            "loss",
            "An OTel bridge span had an unknown name or no join key, so it could not be used.",
            "Upgrade wardex-sdk; the agent's telemetry may have changed.",
        ),
        _R.BRIDGE_INCOMPLETE: ReasonInfo(
            _U.SESSION,
            "partial",
            "The OTel bridge's per-call data did not add up to the run's own totals.",
            "Read the run's totals as authoritative.",
        ),
        _R.BRIDGE_RETIRED: ReasonInfo(
            _U.SESSION,
            "loss",
            "A session's OTel bridge slot was dropped when its root was evicted.",
            "Raise limits.max_units.",
        ),
        _R.BRIDGE_MERGE_SKIPPED: ReasonInfo(
            _U.SPAN,
            "partial",
            "A chat span shipped without its OTel bridge data: the merge queue was full.",
            "Raise the bridge's pending bound.",
        ),
        _R.BRIDGE_UNAVAILABLE: ReasonInfo(
            _U.ADAPTER,
            "loss",
            "The OTel bridge receiver could not start.",
            "Re-run with debug=True to see why.",
        ),
        _R.FRAMEWORK_TRACING_DISABLED: ReasonInfo(
            _U.RUN,
            "loss",
            "The framework's own tracing was off, so its structure was not recorded.",
            "Leave the framework's tracing on.",
        ),
        _R.FRAMEWORK_EVENT_UNMATCHED: ReasonInfo(
            _U.SPAN,
            "loss",
            "A framework event, hook or span found no partner to attach to.",
            "Report it with the debug=True output.",
        ),
        _R.FRAMEWORK_SPAN_KIND_UNMAPPED: ReasonInfo(
            _U.SPAN,
            "loss",
            "The framework produced a span kind wardex does not map.",
            "Upgrade wardex-sdk.",
        ),
        _R.HOOKS_ABSENT: ReasonInfo(
            _U.SESSION,
            "loss",
            "The adapter's hooks never reached the session, so what they observe is absent.",
            "Call init() before importing the framework's entry point.",
        ),
        _R.STREAM_LINE_UNRECOGNIZED: ReasonInfo(
            _U.MESSAGE,
            "loss",
            "A CLI agent's output line could not be read.",
            "Upgrade wardex-sdk; the CLI's output format may have changed.",
        ),
        _R.PAYLOAD_UNSERIALIZABLE: ReasonInfo(
            _U.PAYLOAD,
            "partial",
            "A payload could not be serialized and shipped without that part.",
            "Report it with the debug=True output.",
        ),
        _R.USAGE_WITHHELD: ReasonInfo(
            _U.SPAN,
            "partial",
            "Token usage totals were withheld because their parts did not reconcile.",
            "Read the per-call usage, which is still recorded.",
        ),
        _R.LAUNCH_UNRECOGNIZED: ReasonInfo(
            _U.RUN,
            "loss",
            "A CLI agent was launched in a way the adapter does not recognize.",
            "Launch it with an argument list rather than a shell string.",
        ),
    }
)
