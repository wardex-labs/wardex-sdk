"""Span assembly core — the SDK surface adapters and interceptors build on.

This package is the one public boundary below `wardex_sdk` itself: `__all__`
here is the surface adapters and interceptors are written against, every module
inside stays underscore-private, and the layering is one-way.

WHILE THE SDK IS BETA, `__all__` HERE CARRIES NO SEMVER GUARANTEE. It said it
did, and the claim was retracted rather than quietly broken, because it is about
to be false: several of these names hand out a PARENT — `Ambient`, `Evidence`,
`ParentSource`, `latch_ambient`, `UnitRegistry` among them — and a framework
adapter holding them can assemble a causal edge by hand, at confidence 1.0, from
whatever it likes. That is the one thing this SDK claims never happens, and
closing it means those names stop being exported here. Nothing outside this
repository can be affected today: there is no way to register a third-party
adapter (`AdapterName` is a closed enum, `AdapterInterface` is exported from
nowhere), so the surface has no users to break. Retracting the promise while
that is still true is the honest order; discovering it after someone depends on
it is not.

`assembly/` imports the leaf vocabulary
(`_types`, `_enums`, `_limits`) and the scope layer (`_hub`, `_scope`,
`context/`) and nothing above it — never `interceptors/`, `adapters/`,
`protocol/` or `semantics/` (design §3.1). `tests/test_import_graph.py`
enforces that with an AST scan rather than trusting this docstring.

Why a package rather than a helper module: extraction that leaves the bypass
reachable is not extraction. Adapters cannot import `InternalSpan`, `TraceId`,
`SpanId`, `SpanContext`, `Client` or `_hub` at all (I5), so the only way for an
adapter to obtain parentage is to ask this package for it — the shortcut has no
name to call.

What lives here, and the duplication each module exists to remove:

* `_parentage` + `_diag` + the `Limitation` enum are the base. All six
  parentage sites resolve through `resolve_parentage` / `child_of`, so "how
  did I know this was the parent" has one answer and one vocabulary.
* `_policy` holds the capture gate, so it has one implementation and the byte
  seams compose with it instead of overriding it.
* `_vocab`, `_builder` and `_snapshot` give those same six sites one span
  CONSTRUCTOR and one closed vocabulary — `InternalSpan(...)` appears nowhere
  outside `_types.py` and `_builder.py`, and `guard()` has callers outside
  this package because every draft is built inside one (`finish()` throws on a
  vocabulary breach and I6 forbids that reaching the host).
* `_patchset` is the SDK's one monkeypatch mechanism, replacing six
  hand-rolled dictionaries with an uninstall that is identity-checked, LIFO
  and individually guarded — it is what gives `Limitation.PATCH_SUPERSEDED` an
  emitter.
* `_units` is where a framework identifier is a LOOKUP KEY rather than a
  parent: `UnitRegistry.resolve()` holds the whole of I2 in one function — the
  entry point for a caller that has an id to offer, which the Agent SDK adapter
  does not, since it carries the session in context and reaches the registry
  through `current()` / `sole_live()` / `Unit.child()` instead. And
  `UNIT_EVICTED` / `CHILD_SPAN_UNCLOSED` are the emitters that put an evicted
  entry's own span on the wire (I10) instead of dropping it unmarked. Only the
  three bounded tables whose entries OWN a span can be marked that way; evicting
  a lookup alias or a de-duplication key emits nothing, because there is no span
  to mark, and is recorded in the `alias_table_full` / `claim_table_full`
  counters instead.

`_emit` — the one sink `tests/test_import_graph.py` reserves the name for — is
not here yet. `__all__` grows as modules land and does not shrink.
"""

from ._builder import NULL_DRAFT, IntegrityBuilder, SpanDraft
from ._diag import Counters, counters, guard, report_once
from ._integrity import Limitation
from ._parentage import (
    AMBIENT,
    EMPTY_AMBIENT,
    Ambient,
    Evidence,
    Parentage,
    ParentSource,
    child_of,
    degraded_run,
    in_degraded_run,
    latch_ambient,
    resolve_observed,
    resolve_parentage,
)
from ._patchset import PatchSet
from ._policy import Prefilter, capture_mode_of, should_capture
from ._snapshot import SnapshotDraft
from ._units import PinToken, SpanSink, Unit, UnitKey, UnitKind, UnitRegistry
from ._vocab import (
    Block,
    LinkReason,
    SnapshotType,
    SpanIntent,
    TransportLabel,
    VocabularyError,
    is_declared_extra_key,
    vocabulary_name,
)

__all__ = [
    "AMBIENT",
    "Ambient",
    "Block",
    "Counters",
    "EMPTY_AMBIENT",
    "Evidence",
    "IntegrityBuilder",
    "Limitation",
    "LinkReason",
    "NULL_DRAFT",
    "ParentSource",
    "Parentage",
    "PatchSet",
    "PinToken",
    "Prefilter",
    "SnapshotDraft",
    "SnapshotType",
    "SpanDraft",
    "SpanIntent",
    "SpanSink",
    "TransportLabel",
    "Unit",
    "UnitKey",
    "UnitKind",
    "UnitRegistry",
    "VocabularyError",
    "capture_mode_of",
    "child_of",
    "counters",
    "degraded_run",
    "guard",
    "in_degraded_run",
    "is_declared_extra_key",
    "latch_ambient",
    "report_once",
    "resolve_observed",
    "resolve_parentage",
    "should_capture",
    "vocabulary_name",
]
