"""Capture-integrity vocabulary — design §6.5 tier 3, §I9.

A limitation marker says what wardex *failed* to capture, or captured only by
interpretation. It is the third and last tier of the "concept has no home in
the vocabulary" disposition table: tier 1 is a namespaced ``extra`` key, tier 2
is an ``InternalSpanEvent``, tier 3 is a marker here. Anything that would need
a new span kind is dropped.

The enum is CLOSED. Adding a member is a core change on purpose — the
dashboard renders these, so the set is a product surface, not a scratchpad.
Emitters must never invent a marker string inline.

Step 0 scope: this module holds the enum and nothing else. ``IntegrityBuilder``
(design §3.2) arrives with the rest of the assembly core, and the free-form
marker strings the byte seam and the Rust parsers emit today
(``grpc_parse_failed``, ``body_cap_exceeded``, ``ttft_unavailable_h2``, …) are
folded into this enum in migration step 3, together with the proto enum that
becomes its single source of truth for every language SDK (design §6.6).
Until then this enum and those strings coexist and neither is authoritative
over the other.
"""

from __future__ import annotations

from enum import Enum


class Limitation(Enum):
    """CLOSED. The only legal values of a span's limitation markers.

    Values are the wire contract (``CaptureIntegrity.limitations``); they are
    lower_snake to match every other wardex enum and must not be renamed
    casually.
    """

    # --- Parentage: the edge exists but was not derived from a live scope ---
    # I4 — a guess reports itself. Each of these travels with a
    # CorrelationInfo whose confidence is below 1.0.
    PARENT_UNRESOLVED = "parent_unresolved"
    UNIT_INFERRED_SOLE = "unit_inferred_sole"
    CORRELATION_CONFLICT = "correlation_conflict"

    # --- Context propagation: the mechanism that produces parentage is
    # weakened or absent on this runtime/seam. Declared degradation, not a
    # workaround (design §5.3-(v-b), §9).
    CONTEXT_PROPAGATION_DEGRADED = "context_propagation_degraded"
    CONTEXT_PROPAGATION_UNAVAILABLE = "context_propagation_unavailable"

    # --- Unit lifecycle: a logical unit ended for a reason other than the
    # framework finishing it. I10 — eviction emits, it never drops silently.
    UNIT_EVICTED = "unit_evicted"
    UNIT_INTERRUPTED = "unit_interrupted"
    CHILD_SPAN_UNCLOSED = "child_span_unclosed"

    # --- Adapter lifecycle (I7) ---
    ADAPTER_UNINSTALLED = "adapter_uninstalled"
    PATCH_SUPERSEDED = "patch_superseded"

    # --- Observation completeness (I8, §8.1 rule W) ---
    NO_WIRE_EVIDENCE = "no_wire_evidence"
    POSSIBLE_DUPLICATE_CHAT = "possible_duplicate_chat"
    STREAM_BUFFER_EXCEEDED = "stream_buffer_exceeded"
    TOOL_CALL_ID_UNAVAILABLE_IN_PROCESS = "tool_call_id_unavailable_in_process"
    SNAPSHOT_TYPE_UNKNOWN = "snapshot_type_unknown"
