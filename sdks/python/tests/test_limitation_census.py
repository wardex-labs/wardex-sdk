"""Limitation vocabulary census — design §6.5.1, and the reason the enum is complete.

`_assembly/_integrity.Limitation` is a CLOSED vocabulary, and every span-emit
site runs under `SpanDraft.finish()`, which raises `VocabularyError` on a
marker that is not a member of it — an exception `SpanSink.guard()` swallows.
A marker string that exists in the emitters and not in the enum therefore does
not produce a warning, a partial span or a log line: it deletes the whole span,
and the only trace left is a counter. Routed against the 15-member enum that
predated this census, that would have silently deleted every gRPC span, every
streaming chat span, every WebSocket span and every Agent-SDK adapter span.

So this file is not a unit test of the enum. It is the census itself, re-run
from source on every test run, and it is the mechanism that keeps the two
halves from drifting apart again. Two drifts produced the situation it guards:

  vocabulary without an emitter — the 15 members declared before the census, of
  which 8 have since acquired an emit site (`PARENT_UNRESOLVED` was the first,
  on the orphan-snapshot path in `__init__.py`) while the other 7 still reach no
  span. `_EMITTED_MEMBERS` below is the current split, asserted not described;

  an emitter without vocabulary — the 25 free strings the live code actually
  attaches to spans, none of which was a member before the census.

`_CENSUS_PY`, `_CENSUS_RUST` and `_MEMBER_SITES` below are the recorded output
of the 2026-07-29 census. The scanner rebuilds them from the real source —
Python by AST, Rust by text — and the tests demand exact equality in BOTH
directions, marker BY SITE and not merely by name. A new marker fails because
it is not in the frozen mapping; a marker that vanishes fails because the
mapping still expects it; a NEW SITE for an already-censused marker fails too,
which matters because seven of the censused strings are pre-rename aliases that
`SpanDraft.finish()` would reject *today*, so a site added to one of them after
the freeze is a span-deleting change no name-level check would notice.

HOW A REWIRED SITE MOVES THROUGH THIS FILE. Replacing a free string with a
`Limitation` member happens at the same slot. The scanner resolves `Limitation.X`
references as well as string literals, so a rewired site does not disappear
from the scan — it moves. The commit that rewires a site therefore moves its
entry from `_CENSUS_PY` to `_MEMBER_SITES`, and that move is the evidence the
site actually changed hands. Nothing in this file needs its bounds lowered to
let that happen; if a test here tells you to lower a number, that is a bug in
the test and not an instruction.

Scope note: everything here reads *source text*. Nothing imports the Rust
extension, so a stale `_wardex_native` wheel cannot make these tests lie, and
`uv sync --reinstall-package wardex-sdk` is not needed to run them.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

from wardex_sdk._assembly import Limitation

_REPO = pathlib.Path(__file__).resolve().parents[3]
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "wardex_sdk"
_RUST_ROOTS = (_REPO / "crates", _REPO / "bindings")


# ==========================================================================
# The census, frozen
# ==========================================================================

_CENSUS_PY: dict[str, frozenset[str]] = {}
"""Every limitation string Python can attach to a span today: **none**.

Emptying this table is what it was for. It once held 24 free strings reaching
`CaptureIntegrity.limitations` from six modules, and seven of them — the four
renames plus the three merges — were not `Limitation` values at all, so routing
their sites through `SpanDraft.finish()` without rewiring them would have
deleted the spans and left a counter. Every one of those sites now names a
member and appears in `_MEMBER_SITES` below; the move IS the record of it.

It stays here, empty, rather than being deleted. An empty expectation is a live
assertion: `test_python_census_matches_source` now says *no Python site may
emit a free-string marker again*, which is stronger than anything the populated
table said. Deleting it would retire that rule silently.
"""

_CENSUS_RUST: dict[str, frozenset[str]] = {
    "body_cap_exceeded": frozenset({"crates/wardex-protocol/src/http1.rs"}),
}
"""Every limitation string Rust can attach to a span today (1), by site.

`crates/wardex-protocol` builds markers into `Vec<&'static str>`, they cross the
PyO3 boundary in `bindings/python/src/lib.rs`, are read back in
`_protocol/_http1.py`, and are merged into the span's markers by the byte seam.
They are invisible to any Python-only scan, which is why the Rust half of this
file exists.
"""

_CENSUS_RUST_ENUM: dict[str, frozenset[str]] = {
    "body_cap_exceeded": frozenset({"crates/wardex-codec/src/otlp/map.rs"}),
    "otlp_attribute_truncated": frozenset({"crates/wardex-codec/src/otlp/map.rs"}),
    "vocabulary_unmapped": frozenset(
        {
            "bindings/python/src/codec.rs",
            "crates/wardex-codec/src/vocab.rs",
        }
    ),
}
"""Every reference to a `Limitation` value by its GENERATED RUST NAME, by file.

The blind spot the table above cannot see. `_CENSUS_RUST` scans for marker
STRINGS, which is the whole channel while every Rust marker is a
`&'static str` — and it stopped being the whole channel when the OTLP surface
started attaching markers by proto number. Writing the literal there would have
been visible to the string scan and would also have been a second declaration
of a vocabulary `common.proto` owns (§6.6), free to drift from it; reading the
number keeps one declaration and costs this scanner instead.

REFERENCES, not emit sites: the scan is textual, so a `#[cfg(test)]` module
counts the same as production code, and two of the entries here are exactly
that (`map.rs` asserting the projection, `vocab.rs` pinning the meta value's
number). That is the right disposition for the property being defended — no
member may be reachable from Rust invisibly — and `_RUST_ENUM_EMITTERS` below
is the smaller, hand-declared set that actually reaches a span.

`vocabulary_unmapped` is a META value and not a member of the Python enum at
all; it appears here because the scan finds it, and `_members_with_an_emitter`
filters it out rather than pretending it is vocabulary.
"""

_RUST_ENUM_EMITTERS: frozenset[str] = frozenset({"otlp_attribute_truncated"})
"""Of the references above, the ones that ATTACH a marker to a span.

Hand-declared, and the one place in this file that is. Telling a production
attachment from a test assertion needs the Rust module tree, which no regex
has; what the scanner can do — and what `test_every_rust_enum_emitter_is_seen`
makes it do — is refuse an entry here that it did not find in the source at
all. So the risk this leaves is a marker declared as an emitter that is not
one, which costs a docstring that over-explains; not a marker that reaches a
span with nothing recording it, which is the failure this file exists for.
"""

_META_VALUES: frozenset[str] = frozenset({"vocabulary_unmapped"})
"""Values that live in the schema's `Limitation` enum but not in the vocabulary.

Exactly one today, and it earns the exception rather than being tidied away: it
is a statement ABOUT the vocabulary — "this marker was not one of the words" —
so it is attached to spans like any other value and yet has no Python member to
name, which is what would make every join in this file trip over it.

Listed BY NAME rather than inferred from "no member has this value", because
the two look the same and mean opposite things: a value with no member is
either this deliberate case or a typo in one of the frozen tables above, and
only the second one must fail loudly. `test_the_meta_values_have_no_member`
keeps the list from quietly absorbing the first kind.
"""

_MEMBER_SITES: dict[str, frozenset[str]] = {
    # --- _assembly/ itself ---
    # `_parentage.py` attaches these two from its `_MARKER` table, which fires
    # for the source; `_units.py` attaches them again from `resolve()`, which is
    # the only place that CHOOSES a heuristic source in the first place. Two
    # sites for one fact is not duplication here: one is the mechanism, the
    # other is the decision.
    # The Anthropic adapter was a third site for both and is NOT any more, and
    # the removal is the point rather than tidying. It used to walk its own
    # three-tier ladder — live scope, then sole live session, then nothing — and
    # stamp these by hand at the end of it. Its tool wrapper now opens through
    # `AdapterContext`, which declares the fallback and lets the table above
    # attach the marker, so there is no site left in an adapter that could
    # stamp one of these onto an edge it decided itself.
    # `testing/conformance.py` is a third site for both and attaches NEITHER: it
    # is the conformance suite's `_EDGE_MARKERS`, the four members whose
    # presence on any span of a healthy run means the parent edge is not what it
    # looks like. It is named so the scanner can see it — see the note on
    # ADAPTER_UNINSTALLED below.
    "PARENT_UNRESOLVED": frozenset(
        {
            "_assembly/_parentage.py",
            "_assembly/_units.py",
            "testing/conformance.py",
        }
    ),
    "UNIT_INFERRED_SOLE": frozenset(
        {
            "_assembly/_parentage.py",
            "_assembly/_units.py",
            "testing/conformance.py",
        }
    ),
    # The two §5.4 markers, both on the in-process tool span: the handler is
    # never told the tool_use_id, and the two observers' key spaces can be split
    # or ambiguous in two narrow, detectable configurations.
    "TOOL_CALL_ID_UNAVAILABLE_IN_PROCESS": frozenset({"_adapters/_anthropic_agent_sdk.py"}),
    "TOOL_NAME_COLLISION": frozenset({"_adapters/_anthropic_agent_sdk.py"}),
    "SNAPSHOT_TYPE_UNKNOWN": frozenset({"_assembly/_snapshot.py"}),
    "PATCH_SUPERSEDED": frozenset({"_assembly/_patchset.py"}),
    # `resolve()` records an alias and a live context disagreeing about
    # the trace, `pin_driver()` records a pin declared for a task other than the
    # one calling it, and `open()`/`resolve()` record the scope a CLOSED pin
    # left standing. All were declared vocabulary with no emitter until the unit
    # registry landed.
    # The refusal sites no longer spell the member at the slot: the WORD is
    # chosen in `UnitRegistry.refused_ambient_marker` (this member for a pin or
    # lifetime strand, INSTRUMENTATION_DEGRADED for one the registry's own
    # eviction left), whose two literals sit in a marker-ish assignment this
    # scanner reads — still `_assembly/_units.py`, so the file set holds. The
    # adapter surface's `Fallback.SOLE_LIVE_RUN` path — a site declaring a
    # fallback takes a parent_unit, which is exactly what stops `open()` from
    # seeing the poisoned ambient — asks that same helper rather than spelling
    # a member of its own, so `_adapters/_context.py` left this set: its call
    # is the recorded `Call:refused_ambient_marker` hole in `_UNRESOLVED_PY`.
    "CORRELATION_CONFLICT": frozenset(
        {
            "_assembly/_units.py",
            # A second `system/init` naming a different run on a transport key
            # this table still holds live: two agent runs sharing one root.
            "_adapters/_assembler.py",
            # `_EDGE_MARKERS` again — see PARENT_UNRESOLVED above.
            "testing/conformance.py",
        }
    ),
    # --- transport timing ---
    "CONNECT_TIMING_UNAVAILABLE": frozenset({"_interceptors/_socket.py", "_interceptors/_ssl.py"}),
    "TTFT_UNAVAILABLE_H2": frozenset({"_interceptors/_seam.py"}),
    "TTFT_IPC_APPROXIMATION": frozenset({"_adapters/_assembler.py"}),
    "TRANSPORT_TIMING_UNAVAILABLE_SUBPROCESS": frozenset({"_adapters/_assembler.py"}),
    # --- caps ---
    "GRPC_MESSAGE_TRUNCATED": frozenset({"_semantics/_grpc.py"}),
    "WS_PAYLOAD_TRUNCATED": frozenset({"_interceptors/_seam.py", "_interceptors/_trackers.py"}),
    "CONNECTION_EVICTED": frozenset({"_interceptors/_seam.py"}),
    # --- parsing / interpretation ---
    "FRAME_PARSE_FAILED": frozenset(
        {"_interceptors/_seam.py", "_interceptors/_trackers.py", "_semantics/_grpc.py"}
    ),
    "SEMANTIC_PARSE_FAILED": frozenset({"_interceptors/_seam.py"}),
    "PAYLOAD_COMPRESSED": frozenset({"_interceptors/_trackers.py", "_semantics/_grpc.py"}),
    "TOOL_ARGS_UNPARSED": frozenset({"_interceptors/_seam.py"}),
    "OUTPUT_MESSAGES_UNMAPPED_PART": frozenset({"_interceptors/_seam.py"}),
    "INPUT_MESSAGES_UNMAPPED_PART": frozenset({"_interceptors/_seam.py"}),
    # --- streaming ---
    "REASSEMBLED_FROM_STREAM": frozenset({"_interceptors/_seam.py"}),
    "STREAM_USAGE_UNAVAILABLE": frozenset({"_interceptors/_seam.py"}),
    "SSE_UNKNOWN_PROVIDER": frozenset({"_interceptors/_seam.py"}),
    # --- protocol-specific ---
    "GRPC_WEB_UNSUPPORTED": frozenset({"_interceptors/_seam.py"}),
    "GRPC_STATUS_UNAVAILABLE": frozenset({"_semantics/_grpc.py"}),
    # One site, not two: both byte seams flushed their open WS sessions with the
    # same six lines, and the copy is what let one of them keep a stale
    # installed-flag gate on the uninstall the other had outgrown.
    "WS_NO_CLOSE": frozenset({"_interceptors/_seam.py"}),
    # --- unit / adapter lifecycle ---
    "CHILD_SPAN_UNCLOSED": frozenset({"_adapters/_assembler.py", "_assembly/_units.py"}),
    # The registry's own breadth bound: `UnitRegistry.open` (child table) and
    # `Unit.open_span` (open-draft table). The assembler's per-session tables
    # are the same shape under a DIFFERENT knob (`max_session_entries`) and
    # carry SESSION_ENTRY_TABLE_FULL below — see both members' docstrings for
    # why a generalization in the core limits table is still a separate field.
    "UNIT_TABLE_FULL": frozenset({"_assembly/_units.py"}),
    # The adapter's per-session bound, every site in one file: the open-tool
    # eviction, the sub-agent eviction, and the completion half that reports
    # the same eviction from the other end.
    "SESSION_ENTRY_TABLE_FULL": frozenset({"_adapters/_assembler.py"}),
    # Two emitters, one per bound that can evict a session: the registry closes
    # the oldest ROOT unit at `max_units`, and the
    # assembler closes the oldest SESSION at `max_sessions`. Both EMIT the root
    # span carrying this marker; the code they replace dropped the session and
    # its root with no marker and no test, which is the silent drop I10 forbids.
    "UNIT_EVICTED": frozenset({"_adapters/_assembler.py", "_assembly/_units.py"}),
    "SESSION_ABORTED": frozenset({"_adapters/_assembler.py"}),
    # The OTel bridge's two fail-open verdicts, both attached to the session
    # ROOT at finalize by `_merge_bridge`: NO_DATA when the read-back-confirmed
    # injection produced zero spans, SCHEMA_UNKNOWN when data arrived and
    # classified as nothing (or an undecodable POST was attributed to the sole
    # live bridge session). One emitting file, by design — the receiver and
    # the classifier report through counters and hand the marker decision to
    # the one place that holds the root draft.
    "OTEL_BRIDGE_NO_DATA": frozenset({"_adapters/_assembler.py"}),
    "OTEL_BRIDGE_SCHEMA_UNKNOWN": frozenset({"_adapters/_assembler.py"}),
    # The two shutdown markers, and the split between them is which shutdown
    # actually happened rather than which code path ran. The adapter's
    # `uninstall()` names ADAPTER_UNINSTALLED, and it is what an ordinary exit
    # reaches, because atexit tears the adapter down. `_runtime.py` names
    # UNIT_INTERRUPTED from the signal handler, on the one disposition where
    # the process dies inside the handler and atexit provably never runs.
    #
    # Both members existed here as declarations with no emitter for as long as
    # `close_all` had no production caller — the state this table is designed to
    # make visible rather than comfortable.
    #
    # `testing/conformance.py` is the third site for both and is the one that
    # ATTACHES NEITHER. This table records where a member is USED, and the
    # conformance suite uses them as expectations: it drives each shutdown path
    # and asserts that the run span which shipped says why it is short. Read as
    # a list of emitters it would be wrong; read as what it is — every Python
    # site that names a marker — a checker that names the two shutdown markers
    # is exactly the site that must not be allowed to drift away from the
    # teardowns above it.
    "ADAPTER_UNINSTALLED": frozenset(
        {
            "_adapters/_anthropic_agent_sdk.py",
            "_adapters/_langgraph.py",
            "testing/conformance.py",
        }
    ),
    "UNIT_INTERRUPTED": frozenset({"_runtime.py", "testing/conformance.py"}),
    # Two sites, and they are the two halves of one fact: where wardex failed,
    # and where the consequence lands. `_adapters/_context.py` knows it failed —
    # `_abandon` marks a unit whose open or description died, `_run` marks one
    # whose activation died, `_degrade` marks the enclosing unit when the span
    # itself will not ship. `_assembly/_parentage.py::resolve_observed` marks a
    # span BELOW such a failure: a byte seam's transaction that reached the wire
    # inside a block whose run entry never opened. Neither can see the other's
    # span — that is the point of the second site, not an oversight — because
    # the one that failed does not exist and the one that survived is three
    # layers away with nothing in common but the task.
    "INSTRUMENTATION_DEGRADED": frozenset(
        {
            "_adapters/_context.py",
            "_assembly/_parentage.py",
            # `refused_ambient_marker`: an evict-origin stranded scope is
            # wardex's own bound at work, so the refusal says so instead of
            # CORRELATION_CONFLICT — the repair is `max_units`, not the
            # adapter's pin or lifetime discipline.
            "_assembly/_units.py",
            # `_EDGE_MARKERS` again — see PARENT_UNRESOLVED above.
            "testing/conformance.py",
        }
    ),
}
"""Every place a `Limitation` MEMBER (rather than a free string) reaches a
marker slot, by site — the other half of the census, and the half that grew.

21 entries moved into this table out of `_CENSUS_PY`, one per rewired site,
plus `SNAPSHOT_TYPE_UNKNOWN` — the one member that gained a NEW emitter rather
than a renamed one, because closing `SnapshotType` created the condition it
reports.

The site sets are the same files the strings were emitted from, with three
exceptions that are the census's merges landing:

  * `FRAME_PARSE_FAILED` and `PAYLOAD_COMPRESSED` each list more than one file,
    because `grpc_parse_failed` (the gRPC field builder) and `ws_parse_failed`
    (`_trackers.py`) were one fact, and so were `grpc_compressed` and
    `ws_compressed`.
  * `CONNECT_TIMING_UNAVAILABLE` absorbed `async_connect_unavailable`, which
    shared `_ssl.py` with it, so the file set is unchanged.

Extracting the gRPC semantics moved four sites without changing a line of their
logic: `build_grpc_fields` left `_interceptors/_seam.py` for `_semantics/_grpc.py`, so
`GRPC_MESSAGE_TRUNCATED`, `GRPC_STATUS_UNAVAILABLE` and `PAYLOAD_COMPRESSED`
moved with it. `FRAME_PARSE_FAILED` GAINED that file rather than moving,
because the seam still names the member — `if Limitation.FRAME_PARSE_FAILED not
in grpc_markers` is how the span's label stays consistent with the marker the
builder returned, and R7 sees it.

These are SOURCE sites — the place a member NAME appears in a marker slot — and
that is not the same as the set of markers that reach a span, in either
direction. `capture_state_snapshot` ships `PARENT_UNRESOLVED` onto real records
and does not appear here, because `__init__.py` names no member: the reference
is `_MARKER`'s and `SnapshotDraft`'s, one call away. `UNIT_INFERRED_SOLE` is the
mirror image — three source sites for two decisions, because `_units.py` and the
adapter each name it at the tier that CHOSE `ParentSource.UNIT_SOLE` and
`_parentage.py`'s table would have attached it there regardless.

`BODY_CAP_EXCEEDED` is absent for a different reason: it is produced in Rust and
crosses the PyO3 boundary as a string, so `_CENSUS_RUST` is where it is
recorded. `_protocol/_http1.py` resolves it with `Limitation.from_wire` — the one
string-to-member crossing left — and `_interceptors/_seam.py` copies the
resulting member onto the span. Neither spells a member out, so neither appears
here, and correctly so.
"""

_DISABLED_REASONS: frozenset[str] = frozenset(
    {
        "headers_exceeded",
        "not_http",
        "chunk_size_exceeded",
        "stream_buffer_exceeded",
    }
)
"""`disabled_reason` is a SEPARATE, OPEN vocabulary and must never become enum
members.

These say why a Rust parser latched itself off for an entire connection. No span
exists to carry them — the latch fires precisely because no message was ever
parsed — so `init(debug=True)` prints them once per connection to stderr instead.
Design §6.5.1 keeps them out of the `Limitation` proto enum: the two
vocabularies have different lifetimes and different consumers, one a wire
contract and one a debug string.
"""

_TWO_FATES = "stream_buffer_exceeded"
"""The one string that is both, and the trap §6.5.1 names explicitly.

The HTTP/1 latch (`crates/wardex-protocol/src/http1.rs`) is a `disabled_reason`
and stays out of the enum. The MCP-stdio JSON-RPC instance (`json_rpc.rs`)
happens with a pending request already open, so a span exists and design §4.6
site-3 makes it reportable as `Limitation.STREAM_BUFFER_EXCEEDED`. The member is
therefore correct AND the string must not be reachable from the span-marker
scanner today, because nothing reports it yet. Both halves are asserted below;
either one alone would let the trap close.
"""

_NOT_ADOPTED: frozenset[str] = frozenset(
    {
        # removed 2026-07-01 when full-part mapping landed; the only trace left
        # is an absence assertion in test_output_messages.py
        "output_messages_tool_calls_only",
        # planning documents only — zero emitters in sdks/python/src/, and
        # test_tool_calls_extraction.py records that the branch never fires
        "tool_calls_parse_failed",
        # never emitted at all: an illustrative example in a `_types.py` comment,
        # a ghost the free-form field invented to describe itself
        "tls_inner_only",
        # §13-R5 split this off as a FUTURE detector (a unit with token usage and
        # no child chat span). It is not an adopted marker and this test is where
        # that decision is preserved — adding it as a member is a product
        # decision, not a vocabulary cleanup
        "chat_span_unobserved",
    }
)

_RENAMES: dict[str, Limitation] = {
    "ws_evicted": Limitation.CONNECTION_EVICTED,
    "grpc_compressed": Limitation.PAYLOAD_COMPRESSED,
    "grpc_parse_failed": Limitation.FRAME_PARSE_FAILED,
    "tool_span_unclosed": Limitation.CHILD_SPAN_UNCLOSED,
}
"""The four value-name changes §6.5.1 mandates.

Each is a wire-value change in its own right. They rode the one deliberate
`wardex.v1` break that put these vocabularies on the wire (§6.7), because
`buf`'s `ENUM_VALUE_SAME_NAME` locks a value name the instant it is declared —
so a rename that does not ride such a break never happens at all.
"""

_MERGES: dict[str, Limitation] = {
    # provenance lost: sync fileno miss vs anyio layer split. Same fact
    # (tcp_connect_ms unknown, not zero), same user action (none).
    "async_connect_unavailable": Limitation.CONNECT_TIMING_UNAVAILABLE,
    # provenance lost: which protocol. TransportAttributes.protocol already
    # carries it; encoding a protocol into a marker duplicates a field.
    "ws_compressed": Limitation.PAYLOAD_COMPRESSED,
    # nothing lost: both are framing-layer failures.
    "ws_parse_failed": Limitation.FRAME_PARSE_FAILED,
}
"""Emitted strings folded into a member that has another name.

The merge rule is one line: *if the action a user takes after reading two
markers differs, they stay two members; if the only difference is provenance,
they merge.* The first draft of this comment stated the rule as "a marker that
names a unique tunable is never merged", which is a proxy for the real rule and
a false one — measurement (see `test_merges_lose_only_provenance`) found that
`body_cap_exceeded` and `grpc_message_truncated` both bottom out in
`max_body_bytes`, and that `ws_payload_truncated` is driven by `ws_sample_bytes`
while `max_ws_frame_bytes` surfaces under `frame_parse_failed` instead. The
knobs do not partition the markers; the user's next action does.

`tool_span_unclosed` is in `_RENAMES` rather than here on purpose: it is the one
place where the two drifts overlapped — an emitted string meeting an already
declared member for the same fact — and it is the case the census exists to
catch. Without it the wire would have carried two names for one condition.
"""

_ALIASES: dict[str, Limitation] = {**_RENAMES, **_MERGES}

_EMITTED_MEMBERS: frozenset[str] = frozenset(
    {
        # the 21 the census added
        "CONNECT_TIMING_UNAVAILABLE",
        "TTFT_UNAVAILABLE_H2",
        "TTFT_IPC_APPROXIMATION",
        "TRANSPORT_TIMING_UNAVAILABLE_SUBPROCESS",
        "BODY_CAP_EXCEEDED",
        "GRPC_MESSAGE_TRUNCATED",
        "WS_PAYLOAD_TRUNCATED",
        "CONNECTION_EVICTED",
        "FRAME_PARSE_FAILED",
        "SEMANTIC_PARSE_FAILED",
        "PAYLOAD_COMPRESSED",
        "TOOL_ARGS_UNPARSED",
        "OUTPUT_MESSAGES_UNMAPPED_PART",
        "INPUT_MESSAGES_UNMAPPED_PART",
        "REASSEMBLED_FROM_STREAM",
        "STREAM_USAGE_UNAVAILABLE",
        "SSE_UNKNOWN_PROVIDER",
        "GRPC_WEB_UNSUPPORTED",
        "GRPC_STATUS_UNAVAILABLE",
        "WS_NO_CLOSE",
        "SESSION_ABORTED",
        # pre-census declared, and the census found an emitter for it
        "CHILD_SPAN_UNCLOSED",
        # pre-census declared, whose emitter was BUILT rather than renamed:
        # closing `SnapshotType` created the condition, and `SnapshotDraft`
        # attaches this when a caller hands `capture_state_snapshot` a type
        # outside the enum. The one legitimate way this set grows — a member
        # moving from "declared" to "emitted" — as opposed to rewiring a site,
        # which never changes it.
        "SNAPSHOT_TYPE_UNKNOWN",
        # pre-census declared, whose emitter was BUILT by `_assembly/_patchset.py`:
        # the SDK's one patch mechanism, whose identity-checked restore is the
        # first code able to observe that something else re-patched a symbol
        # wardex had patched. The second member to move from "declared" to
        # "emitted" by gaining a NEW emitter rather than a renamed one.
        "PATCH_SUPERSEDED",
        # pre-census declared, whose first emitter is the _parentage.py _MARKER table.
        # PARENT_UNRESOLVED reached real records first, on the orphan-snapshot
        # path `capture_state_snapshot` opened. Both now have callers that name
        # them at the slot as well: the unit registry's `resolve()` and the
        # Anthropic adapter's in-process tool path each PICK
        # `ParentSource.UNRESOLVED` or `ParentSource.UNIT_SOLE` and say so on
        # the span, rather than leaving the table to speak for them
        "PARENT_UNRESOLVED",
        "UNIT_INFERRED_SOLE",
        # pre-census declared, whose emitters the unit registry BUILT — the
        # third and fourth members
        # to move from "declared" to "emitted" by gaining a NEW emitter rather
        # than a renamed one. `UNIT_EVICTED` fires when the unit registry closes
        # the oldest root at `max_units`; `CORRELATION_CONFLICT` fires four
        # ways, all of them "two answers to one parent question, and the
        # disagreement is on the wire rather than in a counter": an alias and
        # the live context land in different traces, a pin is declared for a
        # task other than the caller, `open()`/`resolve()` refuse the scope a
        # CLOSED pin left standing, and — the one emit path outside the
        # registry — the adapter marks the tool span whose edge a stale pin
        # decided, with the word `refused_ambient_marker` chooses for it.
        "UNIT_EVICTED",
        "CORRELATION_CONFLICT",
        # pre-census declared and §5.4's, whose emitters were BUILT in the
        # Anthropic adapter — the fifth and sixth members to move from "declared" to
        # "emitted". Both ride the in-process tool span: the handler is never
        # given a `tool_use_id`, and the two observers' key spaces can be split
        # (an unresolved server token) or ambiguous (`NO_PREFIX` plus two servers
        # exporting one bare name). `TOOL_NAME_COLLISION` is also the member the
        # 37-value vocabulary was extended for, so this is the moment §5.4's V3
        # correction stops being a declaration.
        "TOOL_CALL_ID_UNAVAILABLE_IN_PROCESS",
        "TOOL_NAME_COLLISION",
        # pre-census declared, and the seventh and eighth to move from
        # "declared" to "emitted" — but by a route neither of the others took.
        # No emitter was built for these: `UnitRegistry.close_all` could always
        # attach them and had no production caller, so they sat in the
        # vocabulary describing a shutdown that never wrote anything down. What
        # moved is not the marker but the teardown — the adapter's `uninstall`
        # and the SIG_DFL branch of the signal handler now finalize live
        # sessions instead of dropping them. A span that used to simply not
        # exist now exists and says why it is short.
        "ADAPTER_UNINSTALLED",
        "UNIT_INTERRUPTED",
        # The ninth, and the only member declared and emitted one commit apart —
        # deliberately, because the commit that declared it said so in its own
        # docstring rather than leaving the gap to be discovered here. Its
        # emitter is `_adapters/_context.py`, which is the only module that
        # learns wardex's own work failed: a unit whose open or description
        # died, one whose activation died, and the enclosing unit when the span
        # itself will not ship. The one member whose subject is wardex.
        "INSTRUMENTATION_DEGRADED",
        # The tenth, and the first reached through neither of the two channels
        # this file was built around: not a Python site naming a member and not
        # a Rust marker string, but a proto NUMBER pushed onto the OTLP surface
        # by `crates/wardex-codec/src/otlp/map.rs`. `_CENSUS_RUST_ENUM` is the
        # scan that makes it visible; before it, this member could have reached
        # a user's wire with nothing here recording that it existed.
        "OTLP_ATTRIBUTE_TRUNCATED",
        # The eleventh: minted WITH its two emit sites (the registry's two
        # `max_entries_per_unit` eviction points), which both carried
        # CHILD_SPAN_UNCLOSED before — a marker swap, not a new capability, so
        # the emitted set gains a name without any site gaining a marker.
        "UNIT_TABLE_FULL",
        # The twelfth and thirteenth: the OTel bridge's fail-open pair, minted
        # WITH their emitter (`_merge_bridge`, the finalize-time merge) in the
        # same PR — the census rule that a new limitation cannot land as
        # vocabulary-without-an-emitter, applied at authoring time.
        "OTEL_BRIDGE_NO_DATA",
        "OTEL_BRIDGE_SCHEMA_UNKNOWN",
        # The fourteenth: the adapter's per-session bound, minted WITH its
        # sites in the same PR for the same reason. Two of those sites are a
        # marker swap the way UNIT_TABLE_FULL was (the open-tool eviction gave
        # up CHILD_SPAN_UNCLOSED) and two are new capability — a sub-agent
        # eviction that used to drop its span entirely, and the completion half
        # that reports the same eviction from the other end.
        "SESSION_ENTRY_TABLE_FULL",
    }
)
"""Which MEMBERS have an emit site today, derived independently below.

Invariant under rewiring: turning `"ws_compressed"` into
`Limitation.PAYLOAD_COMPRESSED` changes how a member is reached, never whether
it is reached. So this set is a decision record that survives a rewiring, where
a count of census entries would not.
"""


# ==========================================================================
# Python scanner
# ==========================================================================
#
# Discovery is by dataflow shape, not by a list of known files: a list would
# have to be extended by the same person who forgot to extend the enum. A name
# is "marker-ish" if it contains `marker` or `limitation` (case-insensitively),
# which is the convention every site already follows, and a value is a candidate
# marker when it reaches a marker-ish slot by one of the shapes below.
#
#   R1  a keyword argument with a marker-ish name       CaptureIntegrity(limitations=...)
#   R2  an assignment to a marker-ish name              _BASE_LIMITATION = "..."
#   R3  .append/.extend/.add/.insert on a marker-ish    markers.append("...")
#   R4  an argument bound to a marker-ish PARAMETER —   tracker.flush("...")
#       derived from the parameter names, so new         _emit_tool(markers=("...",))
#       helpers are covered without an edit here
#   R5  a binary op with a marker-ish operand           limitations + ("...",)
#   R6  a function whose result is unpacked into a      connect, hs, reused, limitations
#       marker-ish name — registered as marker-             = self._resolve_timing(...)
#       producing, then its `return` tuples are read
#       at exactly that index (reading the whole tuple
#       would drag in span names and extra keys)
#   R7  a membership test against a marker-ish name     "..." in txn.ws_markers
#   R8  a name whose scope binds it exactly once —      reason = "..."
#       resolved through that binding, so a literal      markers.append(reason)
#       does not become invisible by being given a name
#   R9  ANY function handed a marker-ish value becomes  _note(markers, "...")
#       a marker sink, and every argument at its call
#       sites is read — the callee's parameter names
#       cannot be trusted to be marker-ish
#
# Both passes are whole-program: `flush` is defined in `_trackers.py` and called
# from three other modules, and `_resolve_timing` is registered in `_seam.py`
# but defined in `_ssl.py` and `_socket.py`. A per-file scanner misses four of
# the 24 markers and still looks like it works.
#
# R8 and R9 exist because the rules above them are all SYNTACTIC: they see a
# literal only where it sits, spelled out, in a marker-ish slot. Six one-line
# edits to `_WebSocketTracker._build_txn` — a local variable, a helper, a module
# constant, a funnel list, a tuple, a conditional — each put a real string on a
# real span while leaving every rule R1..R7 with nothing to look at. R8 resolves
# four of those; R9 resolves the helper; and the funnel list is caught by
# `_UNRESOLVED_PY`, which is the backstop for everything no rule can name.
#
# ANY marker-ish slot the scanner cannot resolve to a value is recorded in
# `unresolved` and frozen, because "I found no marker here" and "I could not
# look" are the same green build otherwise — and it is the second one that
# deletes spans.

_MARKERISH = re.compile(r"marker|limitation", re.IGNORECASE)
_ATTR_NAME_BUILTINS = frozenset({"getattr", "setattr", "hasattr", "delattr"})
_MEMBER_NAMES = frozenset(m.name for m in Limitation)
_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.Module)


def _markerish(name: str | None) -> bool:
    return name is not None and _MARKERISH.search(name) is not None


def _defaults(a: ast.arguments):  # noqa: ANN202
    """Each parameter paired with its default expression, or None.

    Positional defaults align from the RIGHT — `ast.arguments.defaults` covers
    the last N of `posonlyargs + args` — while `kw_defaults` is 1:1 with
    `kwonlyargs` and holds None for the ones without. Getting the alignment
    wrong would attribute a default to the wrong parameter, which is worse than
    not reading defaults at all.
    """
    positional = [*a.posonlyargs, *a.args]
    padding: list[ast.expr | None] = [None] * (len(positional) - len(a.defaults))
    yield from zip(positional, [*padding, *a.defaults], strict=True)
    yield from zip(a.kwonlyargs, a.kw_defaults, strict=True)


def _bound_name(node: ast.expr) -> str | None:
    """The identifier a target/operand binds, for `x` and for `self.x` alike."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _callee(node: ast.Call) -> str | None:
    """The bare name of a call's callee: `f()` and `obj.f()` both yield `f`."""
    return _bound_name(node.func)


class _Slot:
    """What a single marker-ish slot resolved to.

    `inert` is the difference between "this slot holds a 5" and "this slot holds
    something I could not follow". Only the second is a hole.
    """

    __slots__ = ("strings", "members", "inert")

    def __init__(self) -> None:
        self.strings: set[str] = set()
        self.members: set[str] = set()
        self.inert = False

    def __bool__(self) -> bool:
        return bool(self.strings or self.members)


def _bindings(scope: ast.AST) -> dict[str, list[ast.expr | None]]:
    """Name -> every value bound to it directly in `scope`'s own body (R8).

    A `None` entry means "bound by something that is not a simple assignment" —
    a loop target, a parameter, an `except ... as`, an augmented assignment. Such
    a name is deliberately left unresolvable rather than guessed at: R8's whole
    licence is that a name bound exactly once, by an expression the scanner can
    read, is that expression.
    """
    out: dict[str, list[ast.expr | None]] = {}

    def add(name: str, value: ast.expr | None) -> None:
        out.setdefault(name, []).append(value)

    def poison(target: ast.expr) -> None:
        for sub in ast.walk(target):
            if isinstance(sub, ast.Name):
                add(sub.id, None)

    stack: list[ast.AST] = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        if isinstance(node, _SCOPE_NODES) and node is not scope:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                add(node.name, None)
            continue  # a nested scope's own bindings are not this scope's
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    add(target.id, node.value)
                else:
                    poison(target)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            add(node.target.id, node.value)
        elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            add(node.target.id, node.value)
        elif isinstance(node, ast.AugAssign):
            poison(node.target)
        elif isinstance(node, ast.For | ast.AsyncFor):
            poison(node.target)
        elif isinstance(node, ast.comprehension):
            poison(node.target)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            poison(node.optional_vars)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            add(node.name, None)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            for alias in node.names:
                add((alias.asname or alias.name).split(".")[0], None)
        stack.extend(ast.iter_child_nodes(node))

    if isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
        a = scope.args
        for p in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]:
            if p is not None:
                add(p.arg, None)
    return out


def _is_member_ref(node: ast.expr) -> str | None:
    """`Limitation.WS_NO_CLOSE` -> `"WS_NO_CLOSE"`, for anything else None.

    Rewiring a site is exactly the edit that replaces a literal with one of
    these at the same slot, so the scanner has to see both or a rewired site
    looks like a deleted one — which is how a vacuity bound ends up lowered.
    """
    if isinstance(node, ast.Attribute) and node.attr in _MEMBER_NAMES:
        if _bound_name(node.value) == "Limitation":
            return node.attr
    return None


def _resolve(
    node: ast.expr | None,
    slot: _Slot,
    assembled: set[str],
    chain: list[dict[str, list[ast.expr | None]]],
    depth: int = 0,
) -> None:
    """Every marker value reachable in `node`, minus positions that are not values.

    Prunes the attribute-name argument of `getattr`/`setattr`/`hasattr`: the
    byte seam writes `getattr(txn, "limitations", ())` inside an assignment to
    `limitations`, and counting that literal would invent a marker named
    "limitations".

    Records rather than collects anything ASSEMBLED. The census asserts every
    marker is a plain literal — zero are built by f-string, `%`, `.format` or
    `join` — and that assumption is load-bearing: a computed marker cannot be
    enumerated by any scanner, so it must fail the build rather than be missed.
    """
    if node is None:
        return
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            slot.strings.add(node.value)
        else:
            slot.inert = True
        return
    if isinstance(node, ast.JoinedStr):
        assembled.add("f-string")
        return
    member = _is_member_ref(node)
    if member is not None:
        slot.members.add(member)
        return
    if isinstance(node, ast.Name):  # R8
        if depth >= 4:
            return  # cyclic or deep aliasing: report it as unresolved, do not guess
        for scope_bindings in chain:
            if node.id not in scope_bindings:
                continue
            values = scope_bindings[node.id]
            if len(values) == 1 and values[0] is not None:
                _resolve(values[0], slot, assembled, chain, depth + 1)
            return  # the innermost binding scope wins, resolved or not
        return
    if isinstance(node, ast.Call):
        name = _callee(node)
        if name in _ATTR_NAME_BUILTINS:
            for i, arg in enumerate(node.args):
                if i != 1:
                    _resolve(arg, slot, assembled, chain, depth)
            return
        if name in {"format", "join"}:
            assembled.add(f"str.{name}")
            return
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        assembled.add("%-format")
    for child in ast.iter_child_nodes(node):
        _resolve(child, slot, assembled, chain, depth)  # type: ignore[arg-type]


def _shape(node: ast.expr) -> str:
    """A compact, diff-stable description of an unresolvable slot.

    Names the KIND of hole rather than the exact source, so an unrelated edit in
    the same expression does not churn `_UNRESOLVED_PY`, while a hole of a new
    kind — or in a new file — still fails.
    """
    if isinstance(node, ast.Name):
        return f"Name:{node.id}"
    if isinstance(node, ast.Attribute):
        return f"Attribute:{node.attr}"
    if isinstance(node, ast.Call):
        return f"Call:{_callee(node) or '?'}"
    return type(node).__name__


def _trees() -> dict[str, ast.Module]:
    return {
        path.relative_to(_SRC).as_posix(): ast.parse(
            path.read_text(encoding="utf-8"), filename=str(path)
        )
        for path in sorted(_SRC.rglob("*.py"))
    }


class _PythonCensus:
    """The scan result, kept whole so the tests can assert on its parts.

    `markers` and `members` are the answer; `marker_functions`, `producers`,
    `sites` and `unresolved` are the scanner showing its work, so that a scan
    which quietly stopped finding anything fails a test instead of passing one.
    """

    def __init__(self, trees: dict[str, ast.Module]) -> None:
        self.markers: dict[str, set[str]] = {}
        self.members: dict[str, set[str]] = {}
        self.assembled: set[str] = set()
        self.marker_functions: set[str] = set()
        self.producers: set[tuple[str, int | None]] = set()
        self.unresolved: set[tuple[str, str]] = set()
        self.sites: int = 0
        # callee name -> (marker-ish parameter names, marker-ish positional indices)
        self._slots: dict[str, tuple[set[str], set[int]]] = {}
        self._take_all: set[str] = set()  # R9 sinks: read EVERY argument
        self._defs: dict[str, list[tuple[str, ast.AST]]] = {}
        self._scan(trees)

    def _take(
        self, rel: str, node: ast.expr | None, chain: list[dict[str, list[ast.expr | None]]]
    ) -> None:
        slot = _Slot()
        _resolve(node, slot, self.assembled, chain)
        if slot:
            self.sites += 1
            for marker in slot.strings:
                self.markers.setdefault(marker, set()).add(rel)
            for member in slot.members:
                self.members.setdefault(member, set()).add(rel)
        elif node is not None and not slot.inert:
            self.unresolved.add((rel, _shape(node)))

    # -- pass 1: whole-program facts -------------------------------------

    def _collect_defs(self, trees: dict[str, ast.Module]) -> None:
        for rel, tree in trees.items():
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                self._defs.setdefault(node.name, []).append((rel, node))
                a = node.args
                positional = [*a.posonlyargs, *a.args]
                every = [*positional, *a.kwonlyargs, a.vararg, a.kwarg]
                names = {p.arg for p in every if p is not None and _markerish(p.arg)}
                if not names:
                    continue
                self.marker_functions.add(node.name)  # R4
                slot_names, slot_indices = self._slots.setdefault(node.name, (set(), set()))
                slot_names |= names
                # `self` is not an argument at the call site of `obj.f(...)`.
                offset = 1 if positional and positional[0].arg in {"self", "cls"} else 0
                for i, p in enumerate(positional):
                    if _markerish(p.arg):
                        slot_indices.add(i - offset)
                # A marker-ish parameter's DEFAULT is a marker slot too, and the
                # one slot the call-site scan can never reach: a default fires
                # precisely when no call site supplies an argument. A string
                # written there — `def add_limitation(self, marker=
                # "some_marker")` — reaches a span every time the funnel is
                # called bare, and the whole rest of this scanner is blind to it.
                for param, default in _defaults(a):
                    if default is not None and _markerish(param.arg):
                        self._take(rel, default, [])

    def _collect_sinks(self, trees: dict[str, ast.Module]) -> None:
        """R9 — a function handed a marker container is a marker sink.

        `_note(markers, "...")` declares neither parameter marker-ish, so R4
        never looks at it; what gives it away is that a marker-ish value goes
        IN. Such a callee has every argument read at every call site, because
        nothing about its signature says which one carries the string.
        """
        for tree in trees.values():
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                callee = _callee(node)
                if callee is None or callee not in self._defs:
                    continue
                arguments = [*node.args, *(kw.value for kw in node.keywords)]
                if any(_markerish(_bound_name(arg)) for arg in arguments):
                    self.marker_functions.add(callee)
                    self._take_all.add(callee)

    def _collect_producers(self, trees: dict[str, ast.Module]) -> None:
        for tree in trees.values():
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
                    continue
                callee = _callee(node.value)
                if callee is None:
                    continue
                for target in node.targets:  # R6
                    if isinstance(target, ast.Tuple):
                        for i, el in enumerate(target.elts):
                            if _markerish(_bound_name(el)):
                                self.producers.add((callee, i))
                    elif _markerish(_bound_name(target)) and callee in self._defs:
                        self.producers.add((callee, None))

    # -- pass 2: collect, with a live scope chain ------------------------

    def _visit(
        self, rel: str, scope: ast.AST, chain: list[dict[str, list[ast.expr | None]]]
    ) -> None:
        chain = [_bindings(scope), *chain]
        stack: list[ast.AST] = list(ast.iter_child_nodes(scope))
        while stack:
            node = stack.pop()
            if isinstance(node, _SCOPE_NODES):
                self._visit(rel, node, chain)
                continue
            self._check(rel, node, chain)
            stack.extend(ast.iter_child_nodes(node))

    def _check(
        self, rel: str, node: ast.AST, chain: list[dict[str, list[ast.expr | None]]]
    ) -> None:
        if isinstance(node, ast.Call):
            for kw in node.keywords:  # R1
                if _markerish(kw.arg):
                    self._take(rel, kw.value, chain)
            callee = _callee(node)
            if callee in self.marker_functions:  # R4 / R9
                names, indices = self._slots.get(callee, (set(), set()))
                # A splat destroys the positional correspondence, so fall back
                # to reading everything rather than reading the wrong argument.
                splatted = any(isinstance(a, ast.Starred) for a in node.args) or any(
                    kw.arg is None for kw in node.keywords
                )
                take_all = splatted or callee in self._take_all
                for i, arg in enumerate(node.args):
                    if take_all or i in indices:
                        self._take(rel, arg, chain)
                for kw in node.keywords:
                    if take_all or kw.arg in names:
                        self._take(rel, kw.value, chain)
            func = node.func  # R3
            if (
                isinstance(func, ast.Attribute)
                and func.attr in {"append", "extend", "add", "insert"}
                and _markerish(_bound_name(func.value))
            ):
                for arg in node.args:
                    self._take(rel, arg, chain)
        if isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign):  # R2
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Tuple) and _markerish(_bound_name(target)):
                    self._take(rel, node.value, chain)
        if isinstance(node, ast.BinOp):  # R5
            if _markerish(_bound_name(node.left)):
                self._take(rel, node.right, chain)
            if _markerish(_bound_name(node.right)):
                self._take(rel, node.left, chain)
        if isinstance(node, ast.Compare):  # R7
            for op, comparator in zip(node.ops, node.comparators, strict=True):
                if isinstance(op, ast.In | ast.NotIn) and _markerish(_bound_name(comparator)):
                    self._take(rel, node.left, chain)

    def _scan(self, trees: dict[str, ast.Module]) -> None:
        self._collect_defs(trees)
        self._collect_sinks(trees)
        self._collect_producers(trees)

        for rel, tree in trees.items():
            self._visit(rel, tree, [])

        # pass 3 — returns of registered marker-producing functions (R6)
        for callee, index in self.producers:
            for rel, definition in self._defs.get(callee, []):
                chain = [_bindings(definition)]
                for node in ast.walk(definition):
                    if not isinstance(node, ast.Return) or node.value is None:
                        continue
                    if index is None:
                        self._take(rel, node.value, chain)
                    elif isinstance(node.value, ast.Tuple) and index < len(node.value.elts):
                        self._take(rel, node.value.elts[index], chain)


_UNRESOLVED_PY: frozenset[tuple[str, str]] = frozenset(
    {
        # `flush` is two functions: `_WebSocketTracker.flush(marker)` and the
        # public `Client.flush(timeout)`. The scanner keys helpers by bare
        # name, so a bare-Name argument to the public one lands here.
        # `__init__.py` used to appear for exactly that and no longer does:
        # the public `wardex.flush` now maps its None default to a sentinel in
        # a conditional expression, which the scanner reads. `_client.py` left
        # the list the same way (a local derived from a deadline).
        ("_types.py", "Tuple"),
        # `_build_tool(sess, tool, end_ns, status, markers, error_type)` declares
        # a marker-ish parameter, so R4 registers it; R9 then makes it read-all
        # because a marker container goes in. Its other arguments land here.
        # None of them can hold a marker string — they are a session, a tool
        # record, a timestamp, a `StatusCode` and an `error.type`. `Name:status`
        # is where `Name:failed` used to sit: the bool became the field it was
        # encoding, so the third outcome (UNSET, what a bound owes a call it
        # stopped watching) is expressible.
        ("_adapters/_assembler.py", "Name:end_ns"),
        ("_adapters/_assembler.py", "Name:error_type"),
        ("_adapters/_assembler.py", "Name:status"),
        ("_adapters/_assembler.py", "Name:marker"),
        ("_adapters/_assembler.py", "Name:markers"),
        ("_adapters/_assembler.py", "Name:sess"),
        ("_adapters/_assembler.py", "Name:tool"),
        # `_emit_tool(..., markers: tuple[Limitation, ...] = ())` — the DEFAULT,
        # newly visible now that marker-ish parameters have theirs read. The
        # tuple is empty and so demonstrably holds no marker, but slots are
        # classified by shape rather than by contents, and a container that
        # counts itself empty today would count itself empty after someone put
        # a member in it. Recorded rather than special-cased, which is the
        # conservative direction: a spurious hole is noise, a missing one is a
        # marker nothing in this file can see.
        ("_adapters/_assembler.py", "Tuple"),
        # `_PendingSpan.deferred_markers: tuple[Limitation, ...] = ()` — the
        # same shape as the `_emit_tool` default above, one file over: the
        # dataclass field's empty-tuple default. The members that actually
        # flow into the field are spelled at the pend sites in
        # `_adapters/_assembler.py`, where `_MEMBER_SITES` records them.
        ("_adapters/_session_state.py", "Tuple"),
        # every one below is a marker CONTAINER being passed along, or a
        # marker-typed PARAMETER being forwarded, not a marker.
        # `Name:inherited` is `SpanDraft.__init__` copying the `Limitation`
        # MEMBERS `_parentage.py` already put on the EDGE onto the span that
        # edge produced — every one of them censused at the `_MARKER` table
        # where it was decided. It moved here out of the unit registry, which
        # is where it used to live as a helper two of the six parentage sites
        # remembered to call.
        ("_assembly/_builder.py", "Attribute:markers"),
        ("_assembly/_builder.py", "Call:tuple"),
        ("_assembly/_builder.py", "List"),
        ("_assembly/_builder.py", "Name:inherited"),
        ("_assembly/_builder.py", "Name:marker"),
        ("_assembly/_parentage.py", "BinOp"),
        ("_assembly/_parentage.py", "Call:_markers_for"),
        ("_assembly/_parentage.py", "Name:marker"),
        ("_assembly/_parentage.py", "Tuple"),
        ("_assembly/_snapshot.py", "Call:list"),
        ("_assembly/_snapshot.py", "List"),
        ("_assembly/_snapshot.py", "Name:marker"),
        # Three shapes in the unit registry, none of which can introduce a
        # value. `Name:marker` is `Unit.note`,
        # a one-line forward onto the unit's own draft. `Name:reason` is
        # `close_all(reason=...)`, whose member is the CALLER's (an adapter
        # uninstall passes `ADAPTER_UNINSTALLED`, a cancelled process passes
        # `UNIT_INTERRUPTED`) and therefore unreadable from here by
        # construction. The markers this module DECIDES — `UNIT_EVICTED`,
        # `UNIT_TABLE_FULL`, `CHILD_SPAN_UNCLOSED` and the three that
        # `resolve()` stamps — are spelled out as literals at their slots and
        # appear in `_MEMBER_SITES`, which is what keeps this set one of
        # pipes rather than a hiding place. `Call:refused_ambient_marker` is
        # the refusal slot in `open()`/`resolve()`: the member is DECIDED in
        # `refused_ambient_marker` (evict-origin strand vs pin/lifetime
        # strand), whose two literals sit in marker-ish assignments the
        # scanner reads and `_MEMBER_SITES` records — the call site is a
        # recorded hole, not a hiding place.
        ("_assembly/_units.py", "Call:refused_ambient_marker"),
        ("_assembly/_units.py", "Name:marker"),
        ("_assembly/_units.py", "Name:reason"),
        # The three forwards that carry a shutdown marker down to an adapter:
        # `AnthropicAgentSdkAdapter.close_units`, `LangGraphAdapter.close_units`
        # and `close_units_all`. All three are one-line passes with no value of
        # their own. The members they carry are spelled as literals at the sites
        # that DECIDE them — each adapter's `uninstall` and the signal handler —
        # and every one of those is in `_MEMBER_SITES`.
        ("_adapters/_anthropic_agent_sdk.py", "Name:marker"),
        ("_adapters/_langgraph.py", "Name:marker"),
        ("_adapters/_registry.py", "Name:marker"),
        # The adapter contract's own two forwards. `Name:marker` is the `marker`
        # parameter of `Scope.note` / `RunHandle.note` / `Attachment.note` and
        # `AdapterContext.close_all`, each a one-line pass onto the registry.
        # `Attribute:name` is `owner=self.name` riding along in that same
        # `close_all` call — R9 makes every argument of a marker-taking callee
        # read-all, and an adapter's own name is not a marker.
        # `Call:refused_ambient_marker` is the SOLE_LIVE_RUN fallback asking
        # the registry which word the stranded scope earns — the same slot the
        # registry's own refusals use, recorded in `_assembly/_units.py` above;
        # the two literals it can yield are censused where they are decided.
        ("_adapters/_context.py", "Attribute:name"),
        ("_adapters/_context.py", "Call:refused_ambient_marker"),
        ("_adapters/_context.py", "Name:marker"),
        ("_interceptors/_seam.py", "Name:marker"),
        ("_interceptors/_seam.py", "Tuple"),
        ("_interceptors/_socket.py", "Tuple"),
        ("_interceptors/_ssl.py", "Tuple"),
        ("_interceptors/_trackers.py", "Attribute:_req_limitations"),
        ("_interceptors/_trackers.py", "Attribute:limitations"),
        ("_interceptors/_trackers.py", "Call:_merge_markers"),
        ("_interceptors/_trackers.py", "Call:list"),
        ("_interceptors/_trackers.py", "Call:tuple"),
        ("_interceptors/_trackers.py", "Tuple"),
        # The conformance suite READS markers off spans that have already
        # shipped; it never builds a draft and never reaches a sink, so neither
        # of these slots can put a value on a span. `harness.py`'s
        # `Attribute:limitations` is `CaptureIntegrity.limitations` being copied
        # into the `SpanNode` view a check asserts on. `conformance.py`'s
        # `Name:marker` is two slots of one shape: `_assert_shipped(live, root,
        # marker)` asking whether the member the shutdown checks passed to
        # `close_units_all` came back on the wire, and the loop over
        # `_EDGE_MARKERS` asking whether any of the four members that would mean
        # the parent edge is not what it looks like is present. Every member
        # either slot can carry is spelled out as a literal in that same file,
        # which is why it appears six times in `_MEMBER_SITES`.
        ("testing/conformance.py", "Name:marker"),
        ("testing/harness.py", "Attribute:limitations"),
        # `_resolve_markers(raw)` is where a Rust-produced marker STRING becomes
        # a member, and it is now the ONLY such crossing (it moved here
        # from the byte seam, which is why `_interceptors/_seam.py::Name:member`
        # is no longer a hole). It cannot introduce a value: `from_wire` returns
        # a member or None, and the Rust half of this census bounds which
        # members it can return.
        ("_protocol/_http1.py", "Call:_resolve_markers"),
        # `build_grpc_fields` returns its `limitations` accumulator, which is a
        # parameter rebound three times — R8 will not guess at a name bound more
        # than once, and R6 reads the tuple slot it lands in. The hole is the
        # container, not a value: every member that reaches it is spelled out at
        # the rebinding a few lines above and is censused there. This entry
        # moved from `_interceptors/_seam.py` when the function moved; the
        # expression is byte-identical.
        ("_semantics/_grpc.py", "Name:limitations"),
    }
)
"""Every marker-ish slot the scanner could NOT resolve to a value, frozen.

This is the backstop, and it is the half of the census that has no other
witness. Rules R1..R9 report what they found; this reports where they looked and
came back empty, which is the same green build unless it is written down. The
concrete case: `notes = []` / `notes.append("...")` / `markers.extend(notes)`
puts a real string on a real span and defeats every rule in the file — R8
resolves `notes` to `[]` and finds nothing. It cannot be enumerated, so it is
recorded as a hole instead, and a NEW hole fails.

Every entry here was read and classified as a marker container being moved
around — `tuple(markers)`, `limitations + (...)`, the `_MARKER` table's own
plumbing. None of them can introduce a marker string. That is the claim this
freeze pins: not that the scanner is complete, but that its incompleteness is
exactly this list.
"""


# ==========================================================================
# Rust scanner
# ==========================================================================
#
# Rust markers never appear in Python source, so they are read from the crates
# as text. Two shapes produce a span marker (`vec.push("...")` and
# `vec = vec!["..."]`), and the structural guard that keeps the text scan from
# going blind is keyed on the marker vector's TYPE, not on its name: a scan for
# the identifier `limitations` cannot be the backstop for regexes that already
# assume the identifier `limitations`. `Vec<&'static str>` is the type of the
# channel, and every declaration of it in `crates/` and `bindings/` must be a
# known one.
#
# The receiver names the mutation regex looks for are DERIVED from those
# declarations rather than hard-coded, so a marker vector named anything else is
# scanned the moment it is declared. And a mutation whose argument is not a
# string literal — `push(SOME_CONST)` — is recorded as unresolved rather than
# passed over, for the same reason the Python half records its holes.

#: The BORROWED spelling, which every marker channel uses today. Distinctive
#: enough to be the discovery key on its own: `Vec<&'static str>` in this
#: codebase is a marker channel and nothing else, so any NAME declared with it
#: is scanned, including one nobody anticipated.
_RUST_VEC_TYPE = r"Vec\s*<\s*&\s*(?:'static\s+)?str\s*>"
#: The OWNED spelling, which none uses YET — and that "yet" is the hole. A
#: marker vector that ever needs to build a string at runtime becomes
#: `Vec<String>`; keyed on the borrowed form alone this scanner would not see
#: the declaration, the name would not enter `receivers`, and the mutation regex
#: would never look at the vector. The anti-blindness backstop below reads the
#: same declarations, so it would go quiet in the same instant — the exact
#: coupling that guard's docstring says it was designed to avoid.
#:
#: Gated on a marker-ish NAME, and that is a measured decision rather than a
#: cautious one. Accepting every `Vec<String>` drags in seven declarations that
#: are not marker channels at all — `data_lines` in the SSE parser, `into_vec`
#: in the semantic parser, `key`, `items`, `unmapped` — and each would have to
#: be recorded in `_RUST_VEC_DECLARATIONS` as though it carried markers. That
#: does not merely add noise: the table's worth is that an entry MEANS
#: something, and one full of unrelated vectors teaches the next reader to wave
#: the new entry through, which is how a real channel gets recorded and ignored.
#:
#: The residual is stated rather than papered over: an owned marker vector whose
#: name contains neither "marker" nor "limitation" is still invisible. The name
#: rule is a pattern and not the identifier `limitations`, so it is not the
#: circularity the backstop forbids — but it is not the type rule's coverage
#: either, and a channel named for what it carries is the only thing keeping it.
_RUST_OWNED_VEC_TYPE = r"Vec\s*<\s*String\s*>"

#: A marker vector filled by stringifying members of the CLOSED Python enum.
#: Readable, unlike an opaque constant: the vocabulary bounds it at the source.
_RUST_FROM_CLOSED_ENUM = re.compile(r"\benum_str\s*\(")

#: A `Limitation` value named through the generated Rust enum. Readable for the
#: same reason the closed-enum case above is: the schema bounds it at the
#: source, so the scan only has to say WHICH value and where.
#:
#: CamelCase only, deliberately. prost generates CamelCase variants, so an
#: all-caps `Limitation::SOME_VALUE` is prose quoting the proto or the Python
#: enum — it compiles nowhere and attaches nothing, and counting it would put
#: a doc comment in the census as though it were a channel.
_RUST_ENUM_REF = re.compile(r"\bLimitation::([A-Z][A-Za-z0-9]*)\b")
_RUST_BINDING_DECL = re.compile(r"\b([A-Za-z_]\w*)\s*:\s*" + _RUST_VEC_TYPE)
_RUST_FN_DECL = re.compile(r"\bfn\s+([A-Za-z_]\w*)\s*\([^)]*\)\s*->\s*" + _RUST_VEC_TYPE)
_RUST_OWNED_BINDING_DECL = re.compile(r"\b([A-Za-z_]\w*)\s*:\s*" + _RUST_OWNED_VEC_TYPE)
_RUST_OWNED_FN_DECL = re.compile(
    r"\bfn\s+([A-Za-z_]\w*)\s*\([^)]*\)\s*->\s*" + _RUST_OWNED_VEC_TYPE
)


def _wardex_value(variant: str) -> str:
    """`OtlpAttributeTruncated` -> `otlp_attribute_truncated`.

    prost derives the Rust variant from the proto value name by stripping the
    enum's prefix and CamelCasing what is left, so undoing that is the whole
    conversion — and `crates/wardex-codec/src/vocab.rs` derives the wardex value
    from the same proto name by the same convention. Two derivations of one
    naming rule, which is what lets this scanner name a marker the Rust code
    never spells.
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", variant).lower()


def _rust_declared_names(text: str) -> set[str]:
    """Every marker-vector name declared in one Rust source file.

    Two rules, deliberately asymmetric — see `_RUST_OWNED_VEC_TYPE` for why the
    owned form is gated on the name and the borrowed form is not.
    """
    names = {m.group(1) for m in _RUST_BINDING_DECL.finditer(text)}
    names |= {m.group(1) for m in _RUST_FN_DECL.finditer(text)}
    for pattern in (_RUST_OWNED_BINDING_DECL, _RUST_OWNED_FN_DECL):
        names |= {m.group(1) for m in pattern.finditer(text) if _markerish(m.group(1))}
    return names


_RUST_MUTATORS = r"(?:push|insert|extend|extend_from_slice|append)"
_RUST_DISABLED = re.compile(
    r"(?:Disabled|Fail)\s*\(\s*\"([^\"]*)\"" r"|disabled_reason\s*=\s*Some\s*\(\s*\"([^\"]*)\""
)
_RUST_STR = re.compile(r'"((?:[^"\\]|\\.)*)"')

_RUST_VEC_DECLARATIONS: dict[str, str] = {
    "crates/wardex-protocol/src/http1.rs": "ParsedHttp.limitations — the marker vector itself",
    "bindings/python/src/lib.rs": "the PyO3 getter that hands it to _protocol/_http1.py",
    "crates/wardex-codec/src/otlp/map.rs": (
        "the OTLP projection of a span's markers onto an attribute, not a place "
        "any marker is minted. It moved out of the PyO3 binding when the OTLP "
        "semantic mapping did, and it reads NUMBERS off the wire schema now "
        "rather than stringifying a Python enum — so the scan has no literal to "
        "find here either way. Recorded rather than renamed out of the scan: "
        "limitation strings really do flow through it on their way to the wire, "
        "which is what this table is for."
    ),
}
"""Every declaration of a `Vec<&str>` in `crates/` and `bindings/`, by file.

Equality is asserted. A new one is not necessarily a marker channel — but it has
the exact shape of one, and the cost of being wrong is a marker the Rust scan
never sees and `SpanDraft.finish()` silently deletes. If yours carries something
else, add it
here with a comment saying what; if it carries markers, the derived receiver
name means the scan already covers it and the census will tell you what it found.
"""


def _is_cargo_build_output(path: pathlib.Path) -> bool:
    """True only for a real Cargo `target/`, i.e. one beside a `Cargo.toml`.

    `p.parts` containing "target" is not that test: `target` is an ordinary
    module name in a protocol crate (a parse target, a routing target), and a
    marker pushed from `src/target/late.rs` would be dropped in silence by the
    coarser check.
    """
    for parent in path.parents:
        if parent.name == "target" and (parent.parent / "Cargo.toml").is_file():
            return True
    return False


def _rust_paths() -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    """(scanned, excluded) — the exclusion is reported, not silent."""
    scanned: list[pathlib.Path] = []
    excluded: list[pathlib.Path] = []
    for root in _RUST_ROOTS:
        for path in sorted(root.rglob("*.rs")):
            if not path.is_file():
                continue
            (excluded if _is_cargo_build_output(path) else scanned).append(path)
    return scanned, excluded


class _RustCensus:
    def __init__(self) -> None:
        self.markers: dict[str, set[str]] = {}
        self.enum_markers: dict[str, set[str]] = {}
        self.disabled_reasons: dict[str, set[str]] = {}
        self.declarations: dict[str, set[str]] = {}
        self.receivers: set[str] = set()
        self.unresolved: set[tuple[str, str]] = set()
        scanned, excluded = _rust_paths()
        self.files = scanned
        self.excluded = frozenset(p.relative_to(_REPO).as_posix() for p in excluded)
        texts = {p.relative_to(_REPO).as_posix(): p.read_text(encoding="utf-8") for p in scanned}

        # pass 0 — every value named through the generated enum, string or not
        for rel, text in texts.items():
            for match in _RUST_ENUM_REF.finditer(text):
                value = _wardex_value(match.group(1))
                self.enum_markers.setdefault(value, set()).add(rel)

        # pass 1 — where is a marker vector declared, and what is it called?
        for rel, text in texts.items():
            names = _rust_declared_names(text)
            if names:
                self.declarations[rel] = names
                self.receivers |= names

        # pass 2 — every mutation of a vector by one of those names
        alternation = "|".join(sorted(re.escape(n) for n in self.receivers)) or r"(?!)"
        mutate = re.compile(
            r"(?:[A-Za-z_]\w*\s*\.\s*)*\b(?:"
            + alternation
            + r")\s*\.\s*"
            + _RUST_MUTATORS
            + r"\s*\("
        )
        vec_init = re.compile(r"\b(?:" + alternation + r")\s*(?::[^=;]*)?=\s*vec!\s*\[")
        for rel, text in texts.items():
            for pattern in (mutate, vec_init):
                for match in pattern.finditer(text):
                    end = text.find(";", match.end())
                    segment = text[match.end() : end if end != -1 else match.end() + 400]
                    if _RUST_FROM_CLOSED_ENUM.search(segment):
                        # A value READ off the closed `Limitation` enum, not a
                        # marker this file names. `enum_str` stringifies a Python
                        # enum member, so every value is one of the declared
                        # members by construction and there is nothing here the
                        # census could learn from or the wire could be surprised
                        # by. Kept narrow on purpose: `push(SOME_CONST)` is still
                        # a hole, because a constant is a name this scan cannot
                        # follow to a value.
                        continue
                    literals = _RUST_STR.findall(segment)
                    if literals:
                        for literal in literals:
                            self.markers.setdefault(literal, set()).add(rel)
                    else:
                        self.unresolved.add((rel, " ".join(segment.split())[:60]))
            for match in _RUST_DISABLED.finditer(text):
                reason = match.group(1) or match.group(2)
                self.disabled_reasons.setdefault(reason, set()).add(rel)


@pytest.fixture(scope="module")
def py_census() -> _PythonCensus:
    return _PythonCensus(_trees())


@pytest.fixture(scope="module")
def rust_census() -> _RustCensus:
    return _RustCensus()


# ==========================================================================
# The enum itself
# ==========================================================================


_VOCABULARY: dict[str, str] = {
    # --- declared before the census (15), values unchanged by it ---
    "PARENT_UNRESOLVED": "parent_unresolved",
    "UNIT_INFERRED_SOLE": "unit_inferred_sole",
    "CORRELATION_CONFLICT": "correlation_conflict",
    "CONTEXT_PROPAGATION_DEGRADED": "context_propagation_degraded",
    "CONTEXT_PROPAGATION_UNAVAILABLE": "context_propagation_unavailable",
    "UNIT_EVICTED": "unit_evicted",
    "UNIT_INTERRUPTED": "unit_interrupted",
    "CHILD_SPAN_UNCLOSED": "child_span_unclosed",  # absorbed tool_span_unclosed
    "ADAPTER_UNINSTALLED": "adapter_uninstalled",
    "PATCH_SUPERSEDED": "patch_superseded",
    "NO_WIRE_EVIDENCE": "no_wire_evidence",
    "POSSIBLE_DUPLICATE_CHAT": "possible_duplicate_chat",
    "STREAM_BUFFER_EXCEEDED": "stream_buffer_exceeded",
    "TOOL_CALL_ID_UNAVAILABLE_IN_PROCESS": "tool_call_id_unavailable_in_process",
    "SNAPSHOT_TYPE_UNKNOWN": "snapshot_type_unknown",
    # --- census, new (21) ---
    "CONNECT_TIMING_UNAVAILABLE": "connect_timing_unavailable",
    "TTFT_UNAVAILABLE_H2": "ttft_unavailable_h2",
    "TTFT_IPC_APPROXIMATION": "ttft_ipc_approximation",
    "TRANSPORT_TIMING_UNAVAILABLE_SUBPROCESS": "transport_timing_unavailable_subprocess",
    "BODY_CAP_EXCEEDED": "body_cap_exceeded",
    "GRPC_MESSAGE_TRUNCATED": "grpc_message_truncated",
    "WS_PAYLOAD_TRUNCATED": "ws_payload_truncated",
    "CONNECTION_EVICTED": "connection_evicted",  # renamed from ws_evicted
    "FRAME_PARSE_FAILED": "frame_parse_failed",  # renamed from grpc_parse_failed
    "SEMANTIC_PARSE_FAILED": "semantic_parse_failed",
    "PAYLOAD_COMPRESSED": "payload_compressed",  # renamed from grpc_compressed
    "TOOL_ARGS_UNPARSED": "tool_args_unparsed",
    "OUTPUT_MESSAGES_UNMAPPED_PART": "output_messages_unmapped_part",
    "INPUT_MESSAGES_UNMAPPED_PART": "input_messages_unmapped_part",
    "REASSEMBLED_FROM_STREAM": "reassembled_from_stream",
    "STREAM_USAGE_UNAVAILABLE": "stream_usage_unavailable",
    "SSE_UNKNOWN_PROVIDER": "sse_unknown_provider",
    "GRPC_WEB_UNSUPPORTED": "grpc_web_unsupported",
    "GRPC_STATUS_UNAVAILABLE": "grpc_status_unavailable",
    "WS_NO_CLOSE": "ws_no_close",
    "SESSION_ABORTED": "session_aborted",
    # --- §5.4 V3, no emitter today (1) ---
    "TOOL_NAME_COLLISION": "tool_name_collision",
    # --- the one member that describes wardex rather than the observation (1) ---
    "INSTRUMENTATION_DEGRADED": "instrumentation_degraded",
    # --- added after the census, by the OTLP size guard (1) ---
    "OTLP_ATTRIBUTE_TRUNCATED": "otlp_attribute_truncated",
    # --- added after the census, by the registry breadth bound (1) ---
    "UNIT_TABLE_FULL": "unit_table_full",
    # --- added after the census, by the adapter's per-session bound (1): the
    #     same shape one layer out, kept apart because raising the per-unit
    #     knob leaves this one exactly where it was ---
    "SESSION_ENTRY_TABLE_FULL": "session_entry_table_full",
    # --- added after the census, by the Agent SDK OTel bridge (2): its two
    #     fail-open outcomes, kept apart because the reader's next action
    #     differs (nothing arrived vs data arrived and meant nothing) ---
    "OTEL_BRIDGE_NO_DATA": "otel_bridge_no_data",
    "OTEL_BRIDGE_SCHEMA_UNKNOWN": "otel_bridge_schema_unknown",
}


def test_the_vocabulary_is_exactly_these_forty_three() -> None:
    """15 declared before the census + 21 from it + 1 from §5.4 + 1 for wardex
    itself + 1 for the OTLP size guard + 1 for the registry breadth bound
    + 2 for the OTel bridge's fail-open pair + 1 for the adapter's per-session
    bound, name by name.

    A count alone is not enough: a RENAME keeps the count and is the single most
    expensive mistake available here. These are proto enum values in
    `common.proto`, so `buf`'s `ENUM_VALUE_SAME_NAME` (FILE category) locks
    every name and `ENUM_VALUE_NO_DELETE` locks every number. A name that is
    wrong costs a SECOND deliberate schema break to correct, and §6.7's argument
    for why the first one was free — no backend, no deployed envelope bytes —
    expires the day the first ingest endpoint stores anything.

    So this table is a decision record, not a duplicate of the enum. Changing it
    is meant to be as annoying as changing the wire, because it IS changing the
    wire.
    """
    assert {m.name: m.value for m in Limitation} == _VOCABULARY


def test_values_are_unique_and_lower_snake() -> None:
    values = [m.value for m in Limitation]
    assert len(set(values)) == len(values)
    for value in values:
        assert re.fullmatch(r"[a-z][a-z0-9]*(_[a-z0-9]+)*", value), value


def _member_docs() -> dict[str, str]:
    """Each member's attached docstring, read from `_integrity.py` by AST.

    NOT via `Limitation.X.__doc__`: an enum member with no docstring of its own
    inherits the CLASS docstring, so the obvious version of this check passes
    for a member that documents nothing. Reading the source is the only way to
    tell "documented" from "inherited".
    """
    source = _SRC / "_assembly" / "_integrity.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "Limitation")
    docs: dict[str, str] = {}
    body = cls.body
    for i, node in enumerate(body):
        if not (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        following = body[i + 1] if i + 1 < len(body) else None
        if (
            isinstance(following, ast.Expr)
            and isinstance(following.value, ast.Constant)
            and isinstance(following.value.value, str)
        ):
            docs[name] = following.value.value
    return docs


def _members_with_an_emitter() -> set[str]:
    """The members the census reaches, however they are spelled at the site.

    Derived from the frozen tables rather than counted, so moving a site from a
    free string to a member does not change the answer — which member has an
    emitter is invariant under a rewiring, and a bare count is not.

    `_RUST_ENUM_EMITTERS` joins on VALUE like the string tables do, and brings
    one thing the string tables cannot: it is drawn from `_CENSUS_RUST_ENUM`,
    which legitimately contains `vocabulary_unmapped` — a META value that IS
    attached to spans (`bindings/python/src/codec.rs`) and has no member to
    name, because it is a statement about the vocabulary rather than a word in
    it. So the meta values are subtracted BY NAME.

    By name, and not by "whatever `by_value` has no key for", which is the
    difference between a filter and a hole: every other string here is a member
    value by construction, so `by_value[s]` raising is the guard that catches a
    typo recorded in one of the frozen tables. A blanket `if s in by_value`
    would turn that raise into a silent skip and let the census sign off on
    full provenance for a value no member has.
    """
    by_value = {m.value: m for m in Limitation}
    strings = (set(_CENSUS_PY) | set(_CENSUS_RUST) | set(_RUST_ENUM_EMITTERS)) - _META_VALUES
    return {(_ALIASES.get(s) or by_value[s]).name for s in strings} | set(_MEMBER_SITES)


def test_every_member_carries_its_own_provenance() -> None:
    """A member with no docstring of its own cannot be told apart from a debug
    state without re-deriving the census — which is the failure this whole file
    exists to prevent. So the requirement is mechanical, not a review habit.

    Members with an emitter must name the file that emits them; members without
    one must say so, because "no emitter" and "emitter I failed to find" look
    identical in a diff and mean opposite things.
    """
    docs = _member_docs()
    assert set(docs) == {m.name for m in Limitation}

    emitted = _members_with_an_emitter()
    assert emitted == _EMITTED_MEMBERS, (
        "the set of members with an emit site changed.\n"
        f"  gained: {sorted(emitted - _EMITTED_MEMBERS)}\n"
        f"  lost:   {sorted(_EMITTED_MEMBERS - emitted)}\n"
        "This set is invariant under a rewiring — it changes HOW a "
        "member is reached, not WHETHER. If you moved an entry from _CENSUS_PY "
        "to _MEMBER_SITES and this fired, the two entries do not name the same "
        "member and one of them is wrong."
    )

    for name, doc in docs.items():
        assert len(doc.strip()) > 40, name
        if name in emitted:
            assert ".py" in doc or ".rs" in doc, f"{name} does not say where it is emitted"
        else:
            assert "Declared" in doc, f"{name} does not say it has no emitter yet"


def test_legacy_strings_are_named_by_the_member_that_absorbed_them() -> None:
    """The renames and merges are recorded twice — in the enum's prose and in
    `_ALIASES` — and this is what keeps the two copies honest.

    Without it, a reader who finds `grpc_compressed` in the emitters and greps
    the enum for it comes up empty and concludes the census missed one.
    """
    docs = _member_docs()
    for old, member in _ALIASES.items():
        assert old in docs[member.name], f"{member.name} does not mention {old}"


# ==========================================================================
# The census, re-run from source
# ==========================================================================


def _diff_sites(
    discovered: dict[str, set[str]], frozen: dict[str, frozenset[str]]
) -> dict[str, tuple[list[str], list[str]]]:
    """marker -> (sites gained, sites lost), for every marker where they differ."""
    out: dict[str, tuple[list[str], list[str]]] = {}
    for marker in sorted(set(discovered) | set(frozen)):
        now = set(discovered.get(marker, set()))
        then = set(frozen.get(marker, frozenset()))
        if now != then:
            out[marker] = (sorted(now - then), sorted(then - now))
    return out


def test_python_census_matches_source(py_census: _PythonCensus) -> None:
    """Equality by SITE, not by name.

    A marker that gains a site is the case a name-level check waves through, and
    for the seven pre-rename aliases it is a span-deleting change: those strings
    are not `Limitation` values, so `SpanDraft.finish()` raises on
    them and `guard()` eats the span. The site map is what makes "one more place
    emits `grpc_parse_failed`" a build failure instead of a diff nobody reads.
    """
    drift = _diff_sites(py_census.markers, _CENSUS_PY)
    assert not drift, (
        f"Python limitation census drifted: {drift} (marker -> (sites gained, sites lost)).\n"
        "A marker with a NEW site: add a Limitation member if it is new vocabulary; "
        "if it is a connection-level parser disable reason with no span to carry "
        "it, it does not belong in `limitations` at all — see _integrity.py's "
        "module docstring.\n"
        "A marker that LOST every site: if it was rewired to a member, move its "
        "entry to _MEMBER_SITES in the same commit. That move is the evidence the site "
        "changed hands, and nothing else in this file needs to be touched."
    )


def test_member_reference_sites_match_source(py_census: _PythonCensus) -> None:
    """The other half: where a `Limitation` member (not a string) is used.

    It began as `_assembly/_parentage.py`'s `_MARKER` table and nothing else —
    the "vocabulary without an emitter" drift stated as a fact rather than as
    prose. It grew as `_CENSUS_PY` emptied, so it now reaches every
    Python site that names a marker, across `_assembly/`, `_adapters/`,
    `_interceptors/` and `_semantics/`.
    """
    drift = _diff_sites(py_census.members, _MEMBER_SITES)
    assert not drift, (
        f"Limitation member usage drifted: {drift} (member -> (sites gained, sites lost)).\n"
        "A member that GAINED a site is a rewiring: record the site in "
        "_MEMBER_SITES and delete the free string's entry from _CENSUS_PY, in "
        "the same commit. Those two edits are the whole bookkeeping — "
        "no bound in this file needs to move."
    )


def test_rust_census_matches_source(rust_census: _RustCensus) -> None:
    """Verified independently of §6.5.1's claim, not taken from it.

    The design says `body_cap_exceeded` is the only Rust marker attached to a
    span. The scan is what establishes it, and the declaration guard below is
    what keeps it established.
    """
    drift = _diff_sites(rust_census.markers, _CENSUS_RUST)
    assert not drift, f"Rust limitation census drifted: {drift}"


def test_rust_enum_reference_census_matches_source(rust_census: _RustCensus) -> None:
    """The other Rust channel: a value named through the generated enum.

    The string scan above is blind to it by construction, and the OTLP surface
    uses it — so without this a marker could reach a user's wire with nothing in
    this file recording that it exists.
    """
    drift = _diff_sites(rust_census.enum_markers, _CENSUS_RUST_ENUM)
    assert not drift, (
        f"Rust `Limitation::` reference census drifted: {drift} "
        "(value -> (sites gained, sites lost)).\n"
        "A NEW value referenced from Rust: record it here, and if the site "
        "attaches the marker to a span rather than asserting something about "
        "it, add the value to _RUST_ENUM_EMITTERS too."
    )


def test_the_meta_values_have_no_member() -> None:
    """`_META_VALUES` is subtracted from a join whose remaining elements are
    then looked up with `by_value[s]`, so it is the one list here that can turn
    a loud failure into a silent skip.

    Two halves, and the second is the one that matters: a name in here that DOES
    have a member would hide that member's emitter from the provenance test, and
    a name Rust never mentions is a subtraction from a set it was never in.

    Corroborated against the SCAN rather than against the schema, like
    `test_every_rust_enum_emitter_is_seen_by_the_scanner` and for the same
    reason this whole file reads source text: importing the extension to check
    it would make a stale wheel able to answer.
    """
    values = {m.value for m in Limitation}
    assert not (_META_VALUES & values), "a real member value is being filtered out as meta"
    assert _META_VALUES <= set(_CENSUS_RUST_ENUM), "a name here is referenced from nowhere"


def test_every_rust_enum_emitter_is_seen_by_the_scanner() -> None:
    """`_RUST_ENUM_EMITTERS` is the one hand-declared set here, so the scanner
    has to be able to corroborate every entry in it.

    It cannot confirm that a reference ATTACHES a marker — that needs the module
    tree. It can refuse a name it never found in the source, which is what stops
    the set from drifting into a list of members someone meant to wire up.
    """
    assert _RUST_ENUM_EMITTERS <= set(_CENSUS_RUST_ENUM)


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("OtlpAttributeTruncated", "otlp_attribute_truncated"),
        ("BodyCapExceeded", "body_cap_exceeded"),
        ("TtftUnavailableH2", "ttft_unavailable_h2"),
        ("SessionAborted", "session_aborted"),
    ],
)
def test_a_generated_variant_resolves_to_its_wardex_value(variant, expected):
    """The scanner names a marker the Rust code never spells, so the naming rule
    it undoes has to be pinned on its own.

    `TtftUnavailableH2` is the case that decides it: a digit is not a word
    boundary, and a rule that treated it as one would produce
    `ttft_unavailable_h_2`, silently invent a value no enum has, and report a
    censused marker as a new one.
    """
    assert _wardex_value(variant) == expected
    assert expected in {m.value for m in Limitation}


def test_the_marker_vector_is_declared_only_where_recorded(rust_census: _RustCensus) -> None:
    """The anti-blindness guard for the Rust text scan, keyed on the TYPE.

    Keying it on the identifier `limitations` — as the first version did — makes
    it a restatement of the regexes it is supposed to back up: a vector called
    `lims` in `grpc.rs` would be missed by the mutation regex AND by the guard,
    and the pair would still report a clean census. `Vec<&'static str>` is what
    the channel actually is, so that is what is counted.
    """
    assert set(rust_census.declarations) == set(_RUST_VEC_DECLARATIONS), (
        f"declared in {sorted(rust_census.declarations)}, "
        f"recorded {sorted(_RUST_VEC_DECLARATIONS)}.\n"
        "A `Vec<&str>` has the exact shape of a marker channel. Record it in "
        "_RUST_VEC_DECLARATIONS with a comment saying what it carries — the "
        "mutation regex derives its receiver names from these declarations, so "
        "recording it is also what puts it under the scan."
    )
    assert "limitations" in rust_census.receivers, (
        "the receiver names the mutation regex looks for are derived from the "
        "declarations above; deriving nothing would make the Rust scan vacuous"
    )


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        ("    limitations: Vec<&'static str>,", "limitations"),
        ("    limitations: Vec<&str>,", "limitations"),
        ("    limitations: Vec<String>,", "limitations"),
        ("    limitations : Vec< String >,", "limitations"),
        ("    markers: Vec<String>,", "markers"),
        ("    fn limitations(&self) -> Vec<&'static str> {", "limitations"),
        ("    fn limitations(&self) -> Vec<String> {", "limitations"),
    ],
)
def test_an_owned_marker_vector_is_still_a_marker_vector(declaration, expected):
    """Getting this wrong silences the scan AND its own backstop, together.

    Everything downstream hangs off recognizing the declaration: an unrecognized
    one never enters `receivers`, so the mutation regex — whose receiver names
    are DERIVED from the declarations — never looks at the vector at all. And
    `test_the_marker_vector_is_declared_only_where_recorded`, the guard whose
    job is to notice a channel nobody recorded, reads the same declarations, so
    it reports a clean census in the same instant.

    Two guards going blind on one edit is the coupling that guard's docstring
    says it was designed to avoid. It survived only because every marker vector
    in the tree happens to borrow its strings today — the moment one needs to
    build a marker at runtime it becomes `Vec<String>` and both go quiet.
    """
    assert _rust_declared_names(declaration) == {expected}


@pytest.mark.parametrize(
    "declaration",
    [
        "    data_lines: Vec<String>,",
        "    into_vec: Vec<String>,",
        "    pii_disabled: Vec<String>,",
        "    fn key(&self) -> Vec<String> {",
    ],
)
def test_an_ordinary_string_vector_is_not_mistaken_for_a_marker_channel(declaration):
    """The other half of the same decision, and the reason it is a decision.

    `Vec<String>` is an ordinary Rust type; the tree holds seven of them that
    carry no markers. Accepting every one would force each into
    `_RUST_VEC_DECLARATIONS` as though it were a marker channel — and a table
    whose entries mostly mean nothing teaches the next reader to add the next
    entry without looking, which is how a real channel gets recorded and
    ignored. Every declaration here is a real one from `crates/`.
    """
    assert _rust_declared_names(declaration) == set()


def test_an_owned_marker_is_still_extracted_as_a_literal():
    """`push("x".to_string())` and `push(String::from("x"))` must yield `x`.

    Widening the type regex is worth nothing if the literal inside the owned
    construction is then unreadable — the vector would be scanned, every
    mutation of it would resolve to no literal, and each one would land in
    `unresolved` instead of being censused.
    """
    for expression in ('"body_cap_exceeded".to_string()', 'String::from("body_cap_exceeded")'):
        assert _RUST_STR.findall(expression) == ["body_cap_exceeded"], expression


def test_rust_markers_are_all_literals(rust_census: _RustCensus) -> None:
    """`limitations.push(SOME_CONST)` is a marker the text scan cannot name.

    Same reasoning as `_UNRESOLVED_PY`: a mutation the scanner cannot read has to
    fail, because the alternative is a marker that reaches the wire and never
    reaches this file.
    """
    assert rust_census.unresolved == set(), (
        f"marker vector mutated with a non-literal: {sorted(rust_census.unresolved)}. "
        "Push the string literal directly, or extend the Rust scanner to follow "
        "the constant — do not leave it unreadable."
    )


def test_the_rust_walk_only_skips_build_output(rust_census: _RustCensus) -> None:
    """`target` is a legal module name; a Cargo build directory is not a module.

    Reported rather than silent, because an exclusion that drops a source file
    turns the Rust half of the census into a smaller census that still passes.
    """
    for rel in rust_census.excluded:
        assert "/target/" in rel, f"{rel} was skipped and is not under a target/ directory"
        assert "/src/" not in rel.split("/target/")[0], f"{rel} looks like a source module"
    assert any(
        p.as_posix().endswith("crates/wardex-protocol/src/http1.rs") for p in rust_census.files
    ), "the walk did not reach the one file known to push a marker"


def test_no_marker_is_assembled(py_census: _PythonCensus) -> None:
    """Every marker is a plain literal — zero f-strings, `%`, `.format` or `join`.

    Load-bearing: a computed marker cannot be enumerated by any scanner, so the
    first one has to break the build rather than slip past it.
    """
    assert py_census.assembled == set()


def test_unresolvable_marker_slots_are_exactly_the_recorded_ones(
    py_census: _PythonCensus,
) -> None:
    """Where the scanner looked and could not read the value.

    This is the assertion that makes the six one-line escapes fail. A literal
    bound to a plainly-named local, a module constant, a conditional, a tuple —
    R8 reads all of those. A funnel list (`notes = []; notes.append(...)`) it
    cannot, and neither can any scanner that does not implement dataflow. So the
    slot is recorded as a hole, and a hole that is not on the list fails the
    build even though nothing here can name the string inside it.

    If this fired on you: the question is not "how do I make it pass" but "can
    the expression in that slot ever hold a marker string?" If it cannot — it is
    a container being passed along — add it. If it can, spell the literal out at
    the slot so the census can see it.
    """
    assert py_census.unresolved == set(_UNRESOLVED_PY), (
        f"  new holes:  {sorted(py_census.unresolved - _UNRESOLVED_PY)}\n"
        f"  gone holes: {sorted(_UNRESOLVED_PY - py_census.unresolved)}\n"
        "A NEW hole is a marker-ish slot whose value the scanner cannot follow. "
        "Nothing else in this file will notice a marker hidden there."
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('def add_limitation(self, *, marker="baked_kwonly"): ...', "baked_kwonly"),
        ('def note(self, a, b, marker="baked_positional"): ...', "baked_positional"),
    ],
    ids=("kwonly", "positional"),
)
def test_a_marker_written_into_a_parameter_default_is_censused(source, expected):
    """Read the defaults directly, on source this test owns.

    `test_unresolvable_marker_slots_are_exactly_the_recorded_ones` does go red
    when `_collect_defs` stops reading defaults — but only because reading them
    surfaced one hole, `_emit_tool`'s `markers=()` at `_adapters/_assembler.py`.
    That makes the coverage a side effect of an unrelated production signature:
    the day someone gives `_emit_tool` a required `markers`, the recorded hole
    is removed along with it and nothing is left watching defaults at all. A
    guard that stops biting when unrelated code changes is the thing this file
    was audited for, so the behaviour gets a test that owns its own input.

    The positional case is the alignment too. `ast.arguments.defaults` covers
    the LAST N parameters, so pairing it from the left hands `"..."` to `a` and
    `marker` nothing — the marker is silently dropped, which is worse than not
    reading defaults at all because the scan still reports itself complete.
    """
    census = _PythonCensus({"synthetic.py": ast.parse(source)})
    assert expected in census.markers, (
        "a marker string written as a parameter's default reaches a span every "
        "time the funnel is called bare, and no call-site scan can ever see it"
    )


def test_scanner_is_not_blind(py_census: _PythonCensus, rust_census: _RustCensus) -> None:
    """The vacuous-pass trap: a scanner that finds nothing passes everything.

    Every mapping above is satisfied by an empty result set once the frozen
    table it is compared against is also empty — and `_CENSUS_PY` is empty on
    purpose. So the quantities here are chosen to be INVARIANT under a rewiring
    rather than to be counts a rewiring necessarily drives down: markers and
    members are summed, and `sites` counts a slot the same whether it holds
    `"ws_compressed"` or `Limitation.PAYLOAD_COMPRESSED`.

    Every bound may be RAISED as the SDK grows. Both were lowered exactly once,
    when the free strings were rewired, and each drop is arithmetic that can be
    checked rather than a number that was in the way — see below. If a bound
    fails on you, do the same thing: derive what the number SHOULD be and show
    the derivation, or accept that the scanner went blind.
    """
    surface = len(py_census.markers) + len(py_census.members)
    # 26 -> 24, and the two are the census's own merges landing. Beforehand the
    # scan saw 24 distinct STRINGS plus 2 members; afterwards, 24 distinct
    # MEMBERS. The difference: three pairs collapsed into one member each
    # (async_connect_unavailable+connect_timing_unavailable,
    # ws_compressed+grpc_compressed, ws_parse_failed+grpc_parse_failed) for -3,
    # and SNAPSHOT_TYPE_UNKNOWN gained its first emitter for +1. 26-3+1 = 24.
    # The draft of this test claimed the sum was invariant under the rewiring;
    # it is not, because merging two spellings of one fact is the point of the
    # census.
    assert surface >= 24, (
        f"the scan found {surface} distinct marker values (24 members on "
        "2026-07-29). A drop below this means the scanner stopped "
        "seeing something — the member-by-member, site-by-site equality tests "
        "above are what say WHICH. Do not lower this bound without the "
        "arithmetic."
    )
    # 56 -> 45. A slot is a marker-ish SLOT, not a marker, and the rewiring
    # deleted the accumulator plumbing that made up the difference: `limitations =
    # limitations + ("x",)` counted a slot per rebinding, and eleven of those
    # rebindings became `draft.add_limitation(Limitation.X)` calls with no
    # intermediate. Every marker is still found — that is what the equality
    # tests establish — and only the plumbing between them is gone.
    assert py_census.sites >= 45, (
        f"only {py_census.sites} marker-ish slots resolved to a value (45 on 2026-07-29)."
    )
    assert len(rust_census.markers) >= 1, "the Rust scan found no markers at all"
    assert len(rust_census.files) >= 20, (
        f"only {len(rust_census.files)} .rs files walked; the workspace has more, "
        "so the walker or its exclusion rule is broken"
    )
    # R4, R6 and R9 are the rules that reach the indirect sites; if any derived
    # nothing, four markers vanish and the bounds above are the only thing left
    # standing between that and a green run.
    assert {"flush", "_emit_tool"} <= py_census.marker_functions, (
        "R4 derived no marker-taking helpers — `ws_no_close` and `ws_evicted` "
        "are reachable only through `flush(marker)` and nothing else finds them"
    )
    assert {"_resolve_timing", "build_grpc_fields"} <= {name for name, _ in py_census.producers}, (
        "R6 derived no marker-producing functions; the transport-timing markers are behind it"
    )
    assert "_merge_markers" in py_census.marker_functions, (
        "R9 registered no marker sink; a helper that takes a marker container "
        "without declaring a marker-ish parameter is then invisible"
    )
    # The indirection every rewired site uses: it reaches the vocabulary
    # through `SpanDraft.add_limitation(marker)` / `IntegrityBuilder.limitation`
    # rather than by writing into a tuple. If R4 stopped deriving those two, the
    # scan would go blind on the entire rewired surface at once and every
    # equality table above would agree with it.
    assert {"add_limitation", "limitation"} <= py_census.marker_functions, (
        "R4 derived no draft-side marker sink; that is the path every marker takes"
    )


# ==========================================================================
# The gate: no emitted string outside the vocabulary
# ==========================================================================


def test_every_emitted_marker_has_a_member(
    py_census: _PythonCensus, rust_census: _RustCensus
) -> None:
    """THE assertion: no emitted string is outside the vocabulary entirely.

    This is NOT the same test `SpanDraft.finish()` runs. `finish()` rejects any
    string that is not a `Limitation` VALUE, and the seven aliases below are not
    values — they are pre-rename spellings a member replaces. So
    this test is deliberately the weaker one: it asks whether the census knows
    about the string at all. `test_alias_sites_are_the_rewiring_worklist` carries
    the stronger half, and the two must not be collapsed — folding the aliases
    into the pass path here is what would let a NEW site for one of them look
    clean.
    """
    by_value = {m.value: m for m in Limitation}
    emitted = {**py_census.markers, **rust_census.markers}
    unmapped = {
        marker: sorted(where)
        for marker, where in emitted.items()
        if marker not in by_value and marker not in _ALIASES
    }
    assert unmapped == {}, (
        f"limitation marker(s) with no Limitation member: {unmapped}. "
        "`SpanDraft.finish()` DELETES these spans silently."
    )


def test_alias_sites_are_the_rewiring_worklist(py_census: _PythonCensus) -> None:
    """The strings `SpanDraft.finish()` would reject today, and where.

    Reported as their own list rather than folded into the test above, because
    they are the actual work item: every site here emits a string that is not a
    `Limitation` value, so a site left un-rewired while routed through
    `finish()` drops its span and leaves only a counter.

    The expectation is DERIVED from `_CENSUS_PY` rather than written out, so the
    one edit a rewiring makes — moving an entry to `_MEMBER_SITES` — is the only
    edit it makes. A hardcoded list here would be a third table to keep in step,
    and the first contributor to hit it would reach for the shortest fix.
    """
    remaining = {
        old: sorted(py_census.markers[old]) for old in sorted(_ALIASES) if old in py_census.markers
    }
    expected = {old: sorted(_CENSUS_PY[old]) for old in sorted(_ALIASES) if old in _CENSUS_PY}
    assert remaining == expected, (
        f"the alias worklist is {remaining}, but _CENSUS_PY says {expected}"
    )
    sites = sum(len(where) for where in remaining.values())
    assert sites <= 7, (
        f"{sites} sites still emit a pre-rename string (7 on 2026-07-29): {remaining}.\n"
        "This is a RATCHET — it is at zero and nothing may push it back up. "
        "A new site for one of these strings is a span that `finish()` deletes."
    )
    values = {m.value for m in Limitation}
    for old in _ALIASES:
        assert old not in values, f"{old} became a member; it is supposed to be replaced by one"


def test_the_four_renames_are_recorded(py_census: _PythonCensus) -> None:
    """§6.5.1's four value-name changes, asserted from both ends.

    This used to assert the old string was still what the code emits. Every
    site is rewired now, so the assertion inverts and gets STRONGER: the old
    spelling must appear nowhere the scanner can see, the new value must be
    a member, and the old value must NOT be — a vocabulary that carries both
    names is exactly the drift the census was run to end.
    """
    assert set(_RENAMES) == {
        "ws_evicted",
        "grpc_compressed",
        "grpc_parse_failed",
        "tool_span_unclosed",
    }
    values = {m.value for m in Limitation}
    for old, member in _RENAMES.items():
        assert old not in py_census.markers, f"{old} is still emitted; it should be a member now"
        assert member.value in values
        assert old not in values, f"{old} survived as a member alongside {member.value}"


def test_merges_lose_only_provenance() -> None:
    """The merge rule: markers whose reader would ACT differently stay apart.

    Asserted as the negative, because the merges themselves are already covered
    by the mapping test. The knob each marker names is recorded beside it and was
    read out of the source, not out of the design doc — two of the four cap
    markers turned out to share `max_body_bytes`, and `ws_payload_truncated`'s
    knob is not the one §6.5.1 wrote down. They stay separate on the strength of
    the fact each reports, which is what the rule was always about.
    """
    values = {m.value for m in Limitation}
    for distinct in (
        # body stored short. http1.rs::append_capped, cap from cap_for_content_type
        "body_cap_exceeded",  # max_body_bytes / max_opaque_body_bytes
        # a length-prefixed frame cut in half. grpc.rs::parse_grpc_frames takes no
        # Limits at all, so the knob upstream is the SAME body cap — and on h2,
        # where gRPC lives, http2.rs pushes no body_cap_exceeded, so this marker
        # is the only signal the user gets. Different fact, different replay
        # consequence, therefore a different member
        "grpc_message_truncated",  # max_body_bytes (application/grpc is "meaningful")
        # the 64 KiB content sample filled up. NOT max_ws_frame_bytes
        "ws_payload_truncated",  # ws_sample_bytes
        # The two table bounds. `wardex-limits` calls the second a
        # generalization of the first — same order, same semantics — and they
        # still stay apart, because the rule is not whether two markers mean
        # the same thing: it is whether raising the knob the reader was sent to
        # makes the marker go away. These are separate FIELDS, so it does not.
        "unit_table_full",  # max_entries_per_unit
        "session_entry_table_full",  # max_session_entries
        "stream_buffer_exceeded",  # max_stream_buffer_bytes
        "unit_evicted",  # max_units
        "connection_evicted",  # max_connections
        # NOT semantic_parse_failed: framing vs interpretation, different layers.
        # Also where an oversize frame lands — websocket.rs turns a payload over
        # max_ws_frame_bytes into ParseStep::Error, i.e. a disabled parser
        "frame_parse_failed",  # max_ws_frame_bytes, by way of is_disabled()
        "semantic_parse_failed",
        "output_messages_unmapped_part",  # response replay
        "input_messages_unmapped_part",  # prompt replay
        "ws_no_close",  # NOT adapter_uninstalled: completeness, not lifecycle
        "adapter_uninstalled",
    ):
        assert distinct in values, distinct
    # every merged-away string maps somewhere, and nowhere as itself
    for old, member in _MERGES.items():
        assert old not in values, old
        assert member.value in values


# ==========================================================================
# The exclusions
# ==========================================================================


def test_disabled_reasons_are_a_separate_vocabulary(rust_census: _RustCensus) -> None:
    """Connection-level parser disables never become members.

    Equality, not containment: a NEW disable reason must fail here so that
    whoever added it answers the only question that matters — wire-contract
    marker, or stderr debug state? Getting that wrong in either direction is
    what §6.5.1 calls the two-vocabularies mistake.
    """
    assert frozenset(rust_census.disabled_reasons) == _DISABLED_REASONS
    values = {m.value for m in Limitation}
    for reason in _DISABLED_REASONS - {_TWO_FATES}:
        assert reason not in values, f"{reason} is a disable reason, not a span marker"


def test_stream_buffer_exceeded_has_two_fates(
    py_census: _PythonCensus, rust_census: _RustCensus
) -> None:
    """The trap, asserted from both sides.

    It IS a member, because design §4.6 site-3 makes the MCP-stdio JSON-RPC
    instance reportable on a span that already exists. It is NOT emitted as a
    span marker today, because that wiring has not landed; today's two
    occurrences are both parser latches. Assert only the first half and a future
    reader concludes the HTTP/1 latch should be promoted too; assert only the
    second and someone deletes the member as unused.
    """
    assert _TWO_FATES in {m.value for m in Limitation}
    assert _TWO_FATES not in py_census.markers
    assert _TWO_FATES not in rust_census.markers
    assert rust_census.disabled_reasons[_TWO_FATES] == {
        "crates/wardex-protocol/src/http1.rs",
        "crates/wardex-protocol/src/json_rpc.rs",
    }


def test_retired_vocabulary_stays_retired(
    py_census: _PythonCensus, rust_census: _RustCensus
) -> None:
    """Four strings that exist in prose and nowhere else, kept out deliberately.

    Three are dead (`output_messages_tool_calls_only` removed, and
    `tool_calls_parse_failed`/`tls_inner_only` never emitted at all). The fourth,
    `chat_span_unobserved`, is a live design decision: §13-R5 split it off as a
    future detector rather than adopting it, and this assertion is where that
    decision is preserved once the design doc stops being read.
    """
    values = {m.value for m in Limitation}
    emitted = set(py_census.markers) | set(rust_census.markers)
    for ghost in _NOT_ADOPTED:
        assert ghost not in values, f"{ghost} was adopted without a decision"
        assert ghost not in emitted, f"{ghost} acquired an emitter — re-run the census"


def test_a_wardex_bug_is_not_reported_as_a_shallow_tree() -> None:
    """The one member whose subject is wardex, kept apart from its nearest neighbour.

    `CONTEXT_PROPAGATION_DEGRADED` and `INSTRUMENTATION_DEGRADED` describe the
    same SYMPTOM — a tree shallower or thinner than the run that produced it —
    and opposite CAUSES. The first is declared as a property of the runtime, a
    carrier that legitimately could not inherit the context; its knob is "expect
    shallow trees on this seam". The second is a bug in wardex; its knob is
    "file it". Reusing the first for the second files wardex's own defects under
    the host's threading model, where nobody will ever look for them.

    Reusing `CHILD_SPAN_UNCLOSED` or `ADAPTER_UNINSTALLED` is the other tempting
    shortcut and is refused for a sharper reason: each claims a teardown that
    did not happen, so a reader goes hunting for an unclosed child that does not
    exist.
    """
    from wardex_sdk._adapters._context import AdapterContext
    from wardex_sdk._assembly import EMPTY_AMBIENT, SpanIntent, UnitKey, UnitKind, UnitRegistry

    assert Limitation.INSTRUMENTATION_DEGRADED is not Limitation.CONTEXT_PROPAGATION_DEGRADED

    class _Sink:
        def __init__(self) -> None:
            self.drafts: list = []

        def emit(self, draft, *, agent_semantic: bool) -> bool:
            self.drafts.append(draft)
            return True

    class Broken(UnitRegistry):
        __slots__ = ()

        def open(self, *a, **k):
            raise RuntimeError("wardex is broken at open")

    sink = _Sink()
    ctx = AdapterContext("probe", units=Broken(sink=sink), limits={})
    healthy = UnitRegistry(sink=sink)
    holder = healthy.open(
        UnitKind.SESSION,
        UnitKey("t", "s"),
        ambient=EMPTY_AMBIENT,
        intent=SpanIntent.INVOKE_AGENT,
    )
    ctx._degrade("enter.execute_tool", holder=holder, consequence="one span is missing")

    markers = set(holder.draft.integrity.markers)
    assert Limitation.INSTRUMENTATION_DEGRADED in markers
    assert Limitation.CONTEXT_PROPAGATION_DEGRADED not in markers
