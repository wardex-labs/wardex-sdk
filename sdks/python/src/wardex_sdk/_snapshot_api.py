"""The body of `wardex_sdk.capture_state_snapshot`, out of the package root.

The function is public API and stays on `wardex_sdk.__all__`; its BODY lives
here so that the package root imports nothing from `_assembly`. Those eight
names (`resolve_parentage`, `SnapshotDraft`, ...) are the parentage-forging
vocabulary the layering rules exist to keep out of reach, and a root-level
import would have left every one of them reachable as `wardex_sdk.<name>` —
one attribute access from being pinned by a host that was never promised them.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping

from . import _hub
from ._assembly import (
    EMPTY_AMBIENT,
    Evidence,
    ParentSource,
    SnapshotDraft,
    guard,
    latch_ambient,
    parent_is_closed_unit,
    resolve_parentage,
)
from ._enums import SnapshotType
from ._types import InputRef, ToolDefinitionSet

__all__ = ["capture_state_snapshot"]


def capture_state_snapshot(
    *,
    snapshot_type: SnapshotType | str = SnapshotType.TURN_START,
    turn_index: int = 0,
    conversation_state: bytes = b"",
    input_refs: Iterable[InputRef | tuple[str, str]] = (),
    attributes: Mapping[str, str | int | float | bool] | None = None,
    tool_definitions: ToolDefinitionSet | None = None,
) -> None:
    """Record a point-in-time state snapshot against the ambient span.

    `snapshot_type` takes the `SnapshotType` enum, and — as THE deliberate
    exception to the enums-only rule every config field follows — a bare
    string alongside it. Degradation-not-validation is this call's semantics:
    an unrecognized string is not refused, it is recorded as UNSPECIFIED with
    `Limitation.SNAPSHOT_TYPE_UNKNOWN`, because a snapshot with a fuzzy label
    is worth more than a snapshot deleted over one.
    """
    client = _hub.get_client()
    if client is None:
        return
    # A snapshot always EXPECTS a span to hang off — it describes the state of
    # one. So the no-parent case is `unresolved` (I4), not `trace_root`: the two
    # must stay distinguishable downstream, and until this call went through the
    # core it was neither, because the snapshot was dropped where it stood.
    ambient = latch_ambient()
    if parent_is_closed_unit(ambient.span_context):
        # A parent whose unit has already CLOSED is refused here for the same
        # reason the byte seams refuse it (design §10.3): its span has shipped,
        # and a snapshot hung off it describes the state of a run that had
        # already ended. Collapsing to `EMPTY_AMBIENT` takes the conversation
        # and the tracestate down with it, because those came off the dead
        # unit's fork too.
        #
        # This does NOT call `resolve_observed`, and the difference is the
        # no-parent answer rather than an oversight: that function returns a
        # TRACE ROOT when nothing was latched, and a snapshot's whole invariant
        # below is that a missing parent is `unresolved` (I4). A refused corpse
        # therefore lands on the SAME branch as "no parent at all", which is the
        # answer this call site already documents.
        ambient = EMPTY_AMBIENT
    parentage = resolve_parentage(
        ambient,
        Evidence(ParentSource.CONTEXTVAR)
        if ambient.span_context is not None
        else Evidence(ParentSource.UNRESOLVED),
    )
    snapshot = None
    # `finish()` validates, and I6 forbids a validation failure reaching the
    # host: `capture_state_snapshot` is called from the user's own code.
    with guard("capture_state_snapshot", debug=bool(client.config.debug)):
        # `snapshot_type` is coerced to `SnapshotType` inside the draft (enum
        # and string alike), where an unrecognized value degrades to
        # UNSPECIFIED *with* Limitation.SNAPSHOT_TYPE_UNKNOWN
        # instead of being silently flattened by `codec.rs`'s `map_snap`.
        draft = SnapshotDraft(parentage, snapshot_type=snapshot_type, turn_index=turn_index)
        draft.set_conversation_state(conversation_state)
        draft.set_tool_definitions(tool_definitions)
        for ref in input_refs:
            draft.add_input_ref(
                ref if isinstance(ref, InputRef) else InputRef(key=ref[0], content_hash=ref[1])
            )
        draft.set_extras(attributes)
        snapshot = draft.finish(time.time_ns())
    if snapshot is not None:
        client.capture_snapshot(snapshot)
