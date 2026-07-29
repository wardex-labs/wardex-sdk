"""State snapshots, on the same core as spans — design §4.5.

The sixth emit site, and the one every proposal forgot. `capture_state_snapshot`
is a published API that writes a record naming a span, with a trace id, a span
id and an attribute bag — which is to say it makes every decision a span makes
and made all of them somewhere else. Before migration step 1 it made one of them
by returning silently when no parent existed, so a snapshot taken outside a span
was not degraded, marked or counted: it did not happen.

This module finishes the job step 1 started. `SnapshotDraft` is to
`InternalStateSnapshot` what `SpanDraft` is to `InternalSpan`, and it absorbs
the ad-hoc path `__init__.py` was carrying inline — the orphan marker, the
`wardex.limitations` attribute key it rides on, and the caller-key collision
rule that keeps that key from shipping twice.

Two things are closed here that were open:

  * `snapshot_type` was a free `str` with a `"turn_start"` default and a
    `map_snap` in `codec.rs` that silently mapped anything unrecognized to
    `UNSPECIFIED`. It is coerced to `SnapshotType` at exactly one place now, and
    an unrecognized value degrades WITH `Limitation.SNAPSHOT_TYPE_UNKNOWN`
    attached rather than disappearing into the codec. The public signature still
    takes a `str`, so nothing a caller writes today stops working.

  * the marker carrier. `InternalStateSnapshot` has no `limitations` field —
    giving it one is a `state.proto` change and 3a is additive on the wire — so
    markers ride the opaque kv under `wardex.limitations`, exactly as step 1
    left them, but now via a draft that owns the key instead of a module
    constant that had to explain itself to the census scanner.
"""

from __future__ import annotations

from collections.abc import Mapping

from .._enums import SnapshotType
from .._types import InputRef, InternalStateSnapshot, SpanId, ToolDefinitionSet
from ._integrity import Limitation
from ._parentage import Parentage
from ._vocab import VocabularyError

_Scalar = str | int | float | bool

_SNAPSHOT_TYPES: dict[str, SnapshotType] = {t.value: t for t in SnapshotType}
"""Wire value -> member. A table rather than `SnapshotType(value)` in a `try`,
for the reason `Limitation.from_wire` gives: `assembly/` is held to zero silent
swallows (C-S4)."""

NO_SPAN = SpanId(b"\x00" * 8)
"""OTel's invalid-span id, for a record that names no span.

Not a span id this module produced — I1 is about *deciding* parentage, and this
decides nothing. Minting a random id instead would put a dangling reference on
the wire that is byte-indistinguishable from a real one, leaving a wardex-private
kv nobody else parses as the only signal that it points at nothing.
"""

SNAPSHOT_INTEGRITY_KEY = "wardex.limitations"
"""The attribute key snapshot markers ride on until `state.proto` grows a field.

Deliberately NOT named `LIMITATIONS_KEY`: `tests/test_limitation_census.py`
treats any NAME matching `marker|limitation` as a marker SLOT, and would read
this key string as an emitted marker value with no `Limitation` member — a
build failure whose only honest fix would be renaming this constant anyway.
"""


def coerce_snapshot_type(value: SnapshotType | str | None) -> tuple[str, bool]:
    """`(wire value, was_recognized)`.

    An unrecognized type becomes `""`, which `codec.rs`'s `map_snap` renders as
    `SNAPSHOT_TYPE_UNSPECIFIED` — the same wire outcome as before, except that
    the caller now also gets `Limitation.SNAPSHOT_TYPE_UNKNOWN` on the record
    saying wardex did not understand what it was given.
    """
    if isinstance(value, SnapshotType):
        return value.value, True
    known = _SNAPSHOT_TYPES.get(value) if isinstance(value, str) else None
    if known is not None:
        return known.value, True
    return "", False


class SnapshotDraft:
    """A state snapshot under construction. The only constructor of
    `InternalStateSnapshot` outside `_types.py`."""

    __slots__ = (
        "_attrs",
        "_conversation_state",
        "_input_refs",
        "_markers",
        "_parentage",
        "_snapshot_type",
        "_tool_definitions",
        "_turn_index",
    )

    def __init__(
        self,
        parentage: Parentage,
        *,
        snapshot_type: SnapshotType | str | None = SnapshotType.TURN_START,
        turn_index: int = 0,
    ) -> None:
        self._parentage = parentage
        wire, recognized = coerce_snapshot_type(snapshot_type)
        self._snapshot_type = wire
        self._turn_index = turn_index
        self._conversation_state = b""
        self._tool_definitions: ToolDefinitionSet | None = None
        self._input_refs: list[InputRef] = []
        self._attrs: list[tuple[str, _Scalar]] = []
        # The edge's own markers travel with the record: a snapshot ALWAYS
        # expects a span to describe, so "no parent" is `unresolved`, not
        # `trace_root`, and I4 requires that to be visible on the record rather
        # than inferable from an all-zero span id.
        self._markers: list[Limitation] = list(parentage.limitations)
        if not recognized:
            self.add_limitation(Limitation.SNAPSHOT_TYPE_UNKNOWN)

    def set_conversation_state(self, data: bytes) -> None:
        self._conversation_state = data

    def set_tool_definitions(self, defs: ToolDefinitionSet | None) -> None:
        self._tool_definitions = defs

    def add_input_ref(self, ref: InputRef) -> None:
        self._input_refs.append(ref)

    def add_limitation(self, marker: Limitation) -> None:
        # Assignment, not `.append(marker)` — see `IntegrityBuilder.limitation`
        # for why the census scanner cares.
        if marker not in self._markers:
            self._markers = [*self._markers, marker]

    def set_extra(self, key: str, value: _Scalar) -> None:
        """Caller attributes, minus the key this draft owns.

        `attributes` is a `repeated KeyValue` (`state.proto`) and nothing
        between here and the wire de-duplicates, so appending next to
        `wardex.limitations` would ship the key twice; a consumer folding the
        list into a map keeps one, and if it keeps the caller's, the marker that
        makes this record honest is the half that disappears.
        """
        if key == SNAPSHOT_INTEGRITY_KEY:
            return
        self._attrs.append((key, value))

    def set_extras(self, attributes: Mapping[str, _Scalar] | None) -> None:
        if not attributes:
            return
        for key, value in attributes.items():
            self.set_extra(key, value)

    def finish(self, timestamp_ns: int) -> InternalStateSnapshot:
        bad = [m for m in self._markers if not isinstance(m, Limitation)]
        if bad:
            raise VocabularyError(f"limitation(s) outside the Limitation enum: {bad!r}")
        # No namespace check on the keys, and that is deliberate rather than an
        # omission: `capture_state_snapshot(attributes=...)` is a published API
        # that has always taken an arbitrary mapping, so rejecting a key would
        # DELETE a user's record to enforce a namespace wardex has not given
        # them a way to declare. `SpanDraft` exempts its MANUAL mode for the
        # same reason. The one key this draft owns is taken back in `set_extra`.
        attrs = tuple(self._attrs)
        if self._markers:
            attrs = attrs + ((SNAPSHOT_INTEGRITY_KEY, ",".join(m.value for m in self._markers)),)
        parent = self._parentage.parent_span_id
        return InternalStateSnapshot(
            trace_id=self._parentage.trace_id,
            # The snapshot names an EXISTING span, so its span_id is the parent,
            # not a freshly minted child.
            span_id=parent if parent is not None else NO_SPAN,
            timestamp_ns=timestamp_ns,
            snapshot_type=self._snapshot_type,
            turn_index=self._turn_index,
            conversation_state=self._conversation_state,
            tool_definitions=self._tool_definitions,
            attributes=attrs,
            input_refs=tuple(self._input_refs),
        )


__all__ = ["NO_SPAN", "SNAPSHOT_INTEGRITY_KEY", "SnapshotDraft", "coerce_snapshot_type"]
