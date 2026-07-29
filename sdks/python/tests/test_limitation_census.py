"""Limitation vocabulary census — design §6.5.1, the hard prerequisite for step 3a.

`assembly/_integrity.Limitation` is a CLOSED vocabulary, and step 3a puts every
span-emit site under `SpanDraft.finish()`, which raises `VocabularyError` on a
marker that is not a member of it — an exception `SpanSink.guard()` swallows.
A marker string that exists in the emitters and not in the enum therefore does
not produce a warning, a partial span or a log line: it deletes the whole span,
and the only trace left is a counter. Merged against the 15-member enum step 0
landed, 3a would have silently deleted every gRPC span, every streaming chat
span, every WebSocket span and every Agent-SDK adapter span.

So this file is not a unit test of the enum. It is the census itself, re-run
from source on every test run, and it is the mechanism that keeps the two
halves from drifting apart again. Two drifts produced the situation it guards:

  vocabulary without an emitter — the 15 members step 0 declared, of which
  migration step 1 has since given exactly one a live emitter
  (`PARENT_UNRESOLVED`, on the orphan-snapshot path in `__init__.py`) and the
  other 14 still reach no span;

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

HOW STEP 3a INTERACTS WITH THIS FILE. 3a replaces a free string with a
`Limitation` member at the same slot. The scanner resolves `Limitation.X`
references as well as string literals, so a migrated site does not disappear
from the scan — it moves. The commit that migrates a site therefore moves its
entry from `_CENSUS_PY` to `_MEMBER_SITES`, and that move is the evidence the
site actually changed hands. Nothing in this file needs its bounds lowered as
3a proceeds; if a test here tells you to lower a number, that is a bug in the
test and not an instruction.

Scope note: everything here reads *source text*. Nothing imports the Rust
extension, so a stale `_wardex_native` wheel cannot make these tests lie, and
`uv sync --reinstall-package wardex-sdk` is not needed to run them.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

from wardex_sdk.assembly import Limitation

_REPO = pathlib.Path(__file__).resolve().parents[3]
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "wardex_sdk"
_RUST_ROOTS = (_REPO / "crates", _REPO / "bindings")


# ==========================================================================
# The census, frozen
# ==========================================================================

_CENSUS_PY: dict[str, frozenset[str]] = {}
"""Every limitation string Python can attach to a span today: **none**.

Step 3a emptied this table, which is what it was for. Before it, 24 free strings
reached `CaptureIntegrity.limitations` from six modules, and seven of them — the
four renames plus the three merges — were not `Limitation` values at all, so
routing their sites through `SpanDraft.finish()` without rewiring them would
have deleted the spans and left a counter. Every one of those sites now names a
member and appears in `_MEMBER_SITES` below; the move IS the migration record.

It stays here, empty, rather than being deleted. An empty expectation is a live
assertion: `test_python_census_matches_source` now says *no Python site may
emit a free-string marker again*, which is the post-3a rule and is stronger
than anything the populated table said. Deleting it would retire that rule
silently.
"""

_CENSUS_RUST: dict[str, frozenset[str]] = {
    "body_cap_exceeded": frozenset({"crates/wardex-protocol/src/http1.rs"}),
}
"""Every limitation string Rust can attach to a span today (1), by site.

`crates/wardex-protocol` builds markers into `Vec<&'static str>`, they cross the
PyO3 boundary in `bindings/python/src/lib.rs`, are read back in
`protocol/_http1.py`, and are merged into the span's markers by the byte seam.
They are invisible to any Python-only scan, which is why the Rust half of this
file exists.
"""

_MEMBER_SITES: dict[str, frozenset[str]] = {
    # --- assembly/ itself ---
    "PARENT_UNRESOLVED": frozenset({"assembly/_parentage.py"}),
    "UNIT_INFERRED_SOLE": frozenset({"assembly/_parentage.py"}),
    "SNAPSHOT_TYPE_UNKNOWN": frozenset({"assembly/_snapshot.py"}),
    # --- transport timing ---
    "CONNECT_TIMING_UNAVAILABLE": frozenset({"interceptors/_socket.py", "interceptors/_ssl.py"}),
    "TTFT_UNAVAILABLE_H2": frozenset({"interceptors/_seam.py"}),
    "TTFT_IPC_APPROXIMATION": frozenset({"adapters/_assembler.py"}),
    "TRANSPORT_TIMING_UNAVAILABLE_SUBPROCESS": frozenset({"adapters/_assembler.py"}),
    # --- caps ---
    "GRPC_MESSAGE_TRUNCATED": frozenset({"interceptors/_seam.py"}),
    "WS_PAYLOAD_TRUNCATED": frozenset({"interceptors/_seam.py", "interceptors/_trackers.py"}),
    "CONNECTION_EVICTED": frozenset({"interceptors/_seam.py"}),
    # --- parsing / interpretation ---
    "FRAME_PARSE_FAILED": frozenset({"interceptors/_seam.py", "interceptors/_trackers.py"}),
    "SEMANTIC_PARSE_FAILED": frozenset({"interceptors/_seam.py"}),
    "PAYLOAD_COMPRESSED": frozenset({"interceptors/_seam.py", "interceptors/_trackers.py"}),
    "TOOL_ARGS_UNPARSED": frozenset({"interceptors/_seam.py"}),
    "OUTPUT_MESSAGES_UNMAPPED_PART": frozenset({"interceptors/_seam.py"}),
    "INPUT_MESSAGES_UNMAPPED_PART": frozenset({"interceptors/_seam.py"}),
    # --- streaming ---
    "REASSEMBLED_FROM_STREAM": frozenset({"interceptors/_seam.py"}),
    "STREAM_USAGE_UNAVAILABLE": frozenset({"interceptors/_seam.py"}),
    "SSE_UNKNOWN_PROVIDER": frozenset({"interceptors/_seam.py"}),
    # --- protocol-specific ---
    "GRPC_WEB_UNSUPPORTED": frozenset({"interceptors/_seam.py"}),
    "GRPC_STATUS_UNAVAILABLE": frozenset({"interceptors/_seam.py"}),
    "WS_NO_CLOSE": frozenset({"interceptors/_socket.py", "interceptors/_ssl.py"}),
    # --- unit / adapter lifecycle ---
    "CHILD_SPAN_UNCLOSED": frozenset({"adapters/_assembler.py"}),
    "SESSION_ABORTED": frozenset({"adapters/_assembler.py"}),
}
"""Every place a `Limitation` MEMBER (rather than a free string) reaches a
marker slot, by site — the other half of the census, and the half that grew.

Step 3a moved 21 entries into this table out of `_CENSUS_PY`, one per rewired
site, and added `SNAPSHOT_TYPE_UNKNOWN` — the one member that gained a NEW
emitter rather than a renamed one, because 3a is also the step that closes
`SnapshotType`.

The site sets are the same files the strings were emitted from, with three
exceptions that are the census's merges landing:

  * `FRAME_PARSE_FAILED` and `PAYLOAD_COMPRESSED` each now list TWO files,
    because `grpc_parse_failed` (`_seam.py`) and `ws_parse_failed`
    (`_trackers.py`) were one fact, and so were `grpc_compressed` and
    `ws_compressed`.
  * `CONNECT_TIMING_UNAVAILABLE` absorbed `async_connect_unavailable`, which
    shared `_ssl.py` with it, so the file set is unchanged.

These are SOURCE sites — the place a member NAME appears in a marker slot — and
that is not the same as the set of markers that reach a span.
`capture_state_snapshot` ships `PARENT_UNRESOLVED` onto real records and does
not appear here, because `__init__.py` names no member: the reference is
`_MARKER`'s and `SnapshotDraft`'s, one call away. `UNIT_INFERRED_SOLE` still
fires for no one — nothing passes `ParentSource.UNIT_SOLE` yet.

`BODY_CAP_EXCEEDED` is absent for a different reason: it is produced in Rust and
crosses the PyO3 boundary as a string, so `_CENSUS_RUST` is where it is
recorded. `interceptors/_seam.py` resolves it with `Limitation.from_wire`, which
names no member and correctly does not appear here.
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
Design §6.5.1 requires step 3b to keep them out of the `Limitation` proto enum:
the two vocabularies have different lifetimes and different consumers, one a wire
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

Each is a wire-value change in its own right. Step 3b already breaks
`wardex.v1` deliberately (§6.7), and `buf`'s `ENUM_VALUE_SAME_NAME` locks a
value name the instant it is declared — so a rename that does not ride 3b never
happens at all.
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
        # step 0's, which the census found an emitter for
        "CHILD_SPAN_UNCLOSED",
        # step 0's, whose emitter step 3a BUILT rather than renamed: closing
        # `SnapshotType` is 3a's, and `SnapshotDraft` attaches this when a
        # caller hands `capture_state_snapshot` a type outside the enum. The one
        # legitimate way this set grows — a member moving from "declared" to
        # "emitted" — as opposed to the migration, which never changes it.
        "SNAPSHOT_TYPE_UNKNOWN",
        # step 0's, whose emitter is the _parentage.py _MARKER table. A source
        # reference and a live caller are not the same thing and this set keeps
        # them apart: since step 1 wired `capture_state_snapshot`,
        # PARENT_UNRESOLVED reaches real records, while UNIT_INFERRED_SOLE is
        # still reference-only (nothing passes `ParentSource.UNIT_SOLE`)
        "PARENT_UNRESOLVED",
        "UNIT_INFERRED_SOLE",
    }
)
"""Which MEMBERS have an emit site today, derived independently below.

Invariant under step 3a: rewiring `"ws_compressed"` to
`Limitation.PAYLOAD_COMPRESSED` changes how a member is reached, never whether
it is reached. So this set is a decision record that survives the migration,
where a count of census entries would not.
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

    Step 3a's whole edit is replacing a literal with one of these at the same
    slot, so the scanner has to see both or a migrated site looks like a deleted
    one — which is how a vacuity bound ends up being lowered.
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
        # public `Client.flush(timeout)`. The scanner keys helpers by bare name,
        # so the public one's argument lands here. Narrowing the key would drop
        # `ws_no_close`, which is the marker hardest to find in the first place.
        ("__init__.py", "Name:timeout"),
        ("_client.py", "Name:timeout"),
        ("_types.py", "Tuple"),
        # `_build_tool(sess, tool, end_ns, failed, markers, error_type)` declares
        # a marker-ish parameter, so R4 registers it; R9 then makes it read-all
        # because a marker container goes in. Its other arguments land here.
        # None of them can hold a marker string — they are a session, a tool
        # record, a timestamp, a bool and an `error.type`.
        ("adapters/_assembler.py", "Name:end_ns"),
        ("adapters/_assembler.py", "Name:error_type"),
        ("adapters/_assembler.py", "Name:failed"),
        ("adapters/_assembler.py", "Name:marker"),
        ("adapters/_assembler.py", "Name:markers"),
        ("adapters/_assembler.py", "Name:sess"),
        ("adapters/_assembler.py", "Name:tool"),
        # every one below is a marker CONTAINER being passed along, or a
        # marker-typed PARAMETER being forwarded, not a marker
        ("assembly/_builder.py", "Attribute:markers"),
        ("assembly/_builder.py", "Call:tuple"),
        ("assembly/_builder.py", "List"),
        ("assembly/_builder.py", "Name:marker"),
        ("assembly/_parentage.py", "BinOp"),
        ("assembly/_parentage.py", "Call:_markers_for"),
        ("assembly/_parentage.py", "Name:marker"),
        ("assembly/_parentage.py", "Tuple"),
        ("assembly/_snapshot.py", "Call:list"),
        ("assembly/_snapshot.py", "List"),
        ("assembly/_snapshot.py", "Name:marker"),
        ("interceptors/_seam.py", "Name:limitations"),
        ("interceptors/_seam.py", "Name:marker"),
        ("interceptors/_seam.py", "Tuple"),
        ("interceptors/_socket.py", "Tuple"),
        ("interceptors/_ssl.py", "Tuple"),
        ("interceptors/_trackers.py", "Attribute:_req_limitations"),
        ("interceptors/_trackers.py", "Attribute:limitations"),
        ("interceptors/_trackers.py", "Call:_merge_markers"),
        ("interceptors/_trackers.py", "Call:list"),
        ("interceptors/_trackers.py", "Call:tuple"),
        ("interceptors/_trackers.py", "Tuple"),
        # `_resolve_markers(raw)` is where a Rust-produced marker STRING becomes
        # a member, and it is now the ONLY such crossing (step 3b moved it here
        # from the byte seam, which is why `interceptors/_seam.py::Name:member`
        # is no longer a hole). It cannot introduce a value: `from_wire` returns
        # a member or None, and the Rust half of this census bounds which
        # members it can return.
        ("protocol/_http1.py", "Call:_resolve_markers"),
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

_RUST_VEC_TYPE = r"Vec\s*<\s*&\s*(?:'static\s+)?str\s*>"
_RUST_BINDING_DECL = re.compile(r"\b([A-Za-z_]\w*)\s*:\s*" + _RUST_VEC_TYPE)
_RUST_FN_DECL = re.compile(r"\bfn\s+([A-Za-z_]\w*)\s*\([^)]*\)\s*->\s*" + _RUST_VEC_TYPE)
_RUST_MUTATORS = r"(?:push|insert|extend|extend_from_slice|append)"
_RUST_DISABLED = re.compile(
    r"(?:Disabled|Fail)\s*\(\s*\"([^\"]*)\"" r"|disabled_reason\s*=\s*Some\s*\(\s*\"([^\"]*)\""
)
_RUST_STR = re.compile(r'"((?:[^"\\]|\\.)*)"')

_RUST_VEC_DECLARATIONS: dict[str, str] = {
    "crates/wardex-protocol/src/http1.rs": "ParsedHttp.limitations — the marker vector itself",
    "bindings/python/src/lib.rs": "the PyO3 getter that hands it to protocol/_http1.py",
}
"""Every declaration of a `Vec<&str>` in `crates/` and `bindings/`, by file.

Equality is asserted. A new one is not necessarily a marker channel — but it has
the exact shape of one, and the cost of being wrong is a marker the Rust scan
never sees and step 3a silently deletes. If yours carries something else, add it
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
        self.disabled_reasons: dict[str, set[str]] = {}
        self.declarations: dict[str, set[str]] = {}
        self.receivers: set[str] = set()
        self.unresolved: set[tuple[str, str]] = set()
        scanned, excluded = _rust_paths()
        self.files = scanned
        self.excluded = frozenset(p.relative_to(_REPO).as_posix() for p in excluded)
        texts = {p.relative_to(_REPO).as_posix(): p.read_text(encoding="utf-8") for p in scanned}

        # pass 1 — where is a marker vector declared, and what is it called?
        for rel, text in texts.items():
            names = {m.group(1) for m in _RUST_BINDING_DECL.finditer(text)}
            names |= {m.group(1) for m in _RUST_FN_DECL.finditer(text)}
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
    # --- step 0 (15), values unchanged by the census ---
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
}


def test_the_vocabulary_is_exactly_these_thirty_seven() -> None:
    """15 from step 0 + 21 from the census + 1 from §5.4, pinned name by name.

    A count alone is not enough: a RENAME keeps the count and is the single most
    expensive mistake available here. Step 3b declares these as proto enum
    values, at which point `buf`'s `ENUM_VALUE_SAME_NAME` (FILE category) locks
    every name and `ENUM_VALUE_NO_DELETE` locks every number. A name that is
    wrong at that moment costs a SECOND deliberate schema break to correct, and
    §6.7's argument for why the first one is free — no backend, no deployed
    envelope bytes — expires the day the first ingest endpoint stores anything.

    So this table is a decision record, not a duplicate of the enum. Changing it
    is meant to be as annoying as changing the wire, because after 3b it IS
    changing the wire.
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
    source = _SRC / "assembly" / "_integrity.py"
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

    Derived from the frozen tables rather than counted, so step 3a moving a site
    from a free string to a member does not change the answer — which member has
    an emitter is invariant under that migration, and a bare count is not.
    """
    by_value = {m.value: m for m in Limitation}
    strings = set(_CENSUS_PY) | set(_CENSUS_RUST)
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
        "This set is invariant under step 3a — migrating a site changes HOW a "
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
    are not `Limitation` values, so under step 3a `SpanDraft.finish()` raises on
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
        "A marker that LOST every site: if step 3a rewired it, move its entry to "
        "_MEMBER_SITES in the same commit. That move is the evidence the site "
        "changed hands, and nothing else in this file needs to be touched."
    )


def test_member_reference_sites_match_source(py_census: _PythonCensus) -> None:
    """The other half: where a `Limitation` member (not a string) is used.

    Today this is `assembly/_parentage.py`'s `_MARKER` table and nothing else,
    which is the "vocabulary without an emitter" drift stated as a fact rather
    than as prose. Step 3a grows this table as it shrinks `_CENSUS_PY`.
    """
    drift = _diff_sites(py_census.members, _MEMBER_SITES)
    assert not drift, (
        f"Limitation member usage drifted: {drift} (member -> (sites gained, sites lost)).\n"
        "A member that GAINED a site is a step 3a migration: record the site in "
        "_MEMBER_SITES and delete the free string's entry from _CENSUS_PY, in "
        "the same commit. Those two edits are the whole migration bookkeeping — "
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


def test_scanner_is_not_blind(py_census: _PythonCensus, rust_census: _RustCensus) -> None:
    """The vacuous-pass trap: a scanner that finds nothing passes everything.

    Every mapping above is satisfied by an empty result set once the frozen
    table it is compared against is also empty — and step 3a empties
    `_CENSUS_PY` on purpose. So the quantities here are chosen to be INVARIANT
    under that migration rather than to be counts that 3a necessarily drives
    down: markers and members are summed, and `sites` counts a slot the same
    whether it holds `"ws_compressed"` or `Limitation.PAYLOAD_COMPRESSED`.

    Every bound may be RAISED as the SDK grows. Both were lowered exactly once,
    by step 3a, and each drop is arithmetic that can be checked rather than a
    number that was in the way — see below. If a bound fails on you, do the same
    thing: derive what the number SHOULD be and show the derivation, or accept
    that the scanner went blind.
    """
    surface = len(py_census.markers) + len(py_census.members)
    # 26 -> 24, and the two are the census's own merges landing. Before 3a the
    # scan saw 24 distinct STRINGS plus 2 members; after it, 24 distinct
    # MEMBERS. The difference: three pairs collapsed into one member each
    # (async_connect_unavailable+connect_timing_unavailable,
    # ws_compressed+grpc_compressed, ws_parse_failed+grpc_parse_failed) for -3,
    # and SNAPSHOT_TYPE_UNKNOWN gained its first emitter for +1. 26-3+1 = 24.
    # The draft of this test claimed the sum was invariant under 3a; it is not,
    # because merging two spellings of one fact is the point of the census.
    assert surface >= 24, (
        f"the scan found {surface} distinct marker values (24 members on "
        "2026-07-29, post-3a). A drop below this means the scanner stopped "
        "seeing something — the member-by-member, site-by-site equality tests "
        "above are what say WHICH. Do not lower this bound without the "
        "arithmetic."
    )
    # 56 -> 45. A slot is a marker-ish SLOT, not a marker, and 3a deleted the
    # accumulator plumbing that made up the difference: `limitations =
    # limitations + ("x",)` counted a slot per rebinding, and eleven of those
    # rebindings became `draft.add_limitation(Limitation.X)` calls with no
    # intermediate. Every marker is still found — that is what the equality
    # tests establish — and only the plumbing between them is gone.
    assert py_census.sites >= 45, (
        f"only {py_census.sites} marker-ish slots resolved to a value (45 on 2026-07-29, post-3a)."
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
    assert {"_resolve_timing", "_build_grpc_fields"} <= {name for name, _ in py_census.producers}, (
        "R6 derived no marker-producing functions; the transport-timing markers are behind it"
    )
    assert "_merge_markers" in py_census.marker_functions, (
        "R9 registered no marker sink; a helper that takes a marker container "
        "without declaring a marker-ish parameter is then invisible"
    )
    # Step 3a's own indirection: every rewired site reaches the vocabulary
    # through `SpanDraft.add_limitation(marker)` / `IntegrityBuilder.limitation`
    # rather than by writing into a tuple. If R4 stopped deriving those two, the
    # scan would go blind on the entire migrated surface at once and every
    # equality table above would agree with it.
    assert {"add_limitation", "limitation"} <= py_census.marker_functions, (
        "R4 derived no draft-side marker sink; after step 3a that is the path every marker takes"
    )


# ==========================================================================
# The gate on step 3a
# ==========================================================================


def test_every_emitted_marker_has_a_member(
    py_census: _PythonCensus, rust_census: _RustCensus
) -> None:
    """THE assertion: no emitted string is outside the vocabulary entirely.

    This is NOT the same test `SpanDraft.finish()` runs. `finish()` rejects any
    string that is not a `Limitation` VALUE, and the seven aliases below are not
    values — they are pre-rename spellings that step 3a is going to replace. So
    this test is deliberately the weaker one: it asks whether the census knows
    about the string at all. `test_alias_sites_are_the_step_3a_worklist` carries
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
        "Under step 3a these spans are DELETED silently."
    )


def test_alias_sites_are_the_step_3a_worklist(py_census: _PythonCensus) -> None:
    """The strings `SpanDraft.finish()` would reject today, and where.

    Reported as their own list rather than folded into the test above, because
    they are the actual work item: every site here emits a string that is not a
    `Limitation` value, so the moment 3a routes that site through `finish()`
    without rewiring it, the span is dropped and only a counter remains.

    The expectation is DERIVED from `_CENSUS_PY` rather than written out, so the
    one edit a migration makes — moving an entry to `_MEMBER_SITES` — is the only
    edit it makes. A hardcoded list here would be a third table to keep in step,
    and the first contributor to hit it would reach for the shortest fix.
    """
    remaining = {
        old: sorted(py_census.markers[old]) for old in sorted(_ALIASES) if old in py_census.markers
    }
    expected = {old: sorted(_CENSUS_PY[old]) for old in sorted(_ALIASES) if old in _CENSUS_PY}
    assert remaining == expected, (
        f"the step 3a alias worklist is {remaining}, but _CENSUS_PY says {expected}"
    )
    sites = sum(len(where) for where in remaining.values())
    assert sites <= 7, (
        f"{sites} sites still emit a pre-rename string (7 on 2026-07-29): {remaining}.\n"
        "This is a RATCHET — step 3a drives it to zero and nothing may push it "
        "back up. A new site for one of these strings is a span that 3a deletes."
    )
    values = {m.value for m in Limitation}
    for old in _ALIASES:
        assert old not in values, f"{old} became a member; it is supposed to be replaced by one"


def test_the_four_renames_are_recorded(py_census: _PythonCensus) -> None:
    """§6.5.1's four value-name changes, asserted from both ends.

    Before step 3a this asserted the old string was still what the code emits.
    3a rewired every site, so the assertion inverts and gets STRONGER: the old
    spelling must now appear nowhere the scanner can see, the new value must be
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
        assert old not in py_census.markers, f"{old} is still emitted; 3a should have rewired it"
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
