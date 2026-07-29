"""Span assembly core — the SDK surface adapters and interceptors build on.

This package is the one public boundary below `wardex_sdk` itself: `__all__`
here is a semver-stable contract, every module inside stays underscore-private,
and the layering is one-way. `assembly/` imports the leaf vocabulary
(`_types`, `_enums`, `_limits`) and the scope layer (`_hub`, `_scope`,
`context/`) and nothing above it — never `interceptors/`, `adapters/`,
`protocol/` or `semantics/` (design §3.1). `tests/test_import_graph.py`
enforces that with an AST scan rather than trusting this docstring.

Why a package rather than a helper module: extraction that leaves the bypass
reachable is not extraction. Adapters cannot import `InternalSpan`, `TraceId`,
`SpanId`, `SpanContext`, `Client` or `_hub` at all (I5), so the only way for an
adapter to obtain parentage is to ask this package for it — the shortcut has no
name to call.

Migration status (design §11): step 0 landed `_parentage`, `_diag` and the
`Limitation` enum; step 1 wired the six parentage sites onto `resolve_parentage`
/ `child_of`; step 2 landed `_policy`, so the capture gate has one
implementation and the byte seams compose with it instead of overriding it;
step 3a landed `_vocab`, `_builder` and `_snapshot`, so the same six sites now
also share one span CONSTRUCTOR and one closed vocabulary — `InternalSpan(...)`
appears nowhere outside `_types.py` and `_builder.py`, and `guard()` finally has
callers outside this package (every draft is built inside one, because
`finish()` throws on a vocabulary breach and I6 forbids that reaching the host);
step 5 landed `_patchset`, so the six hand-rolled monkeypatch dictionaries are
one mechanism whose uninstall is identity-checked, LIFO and individually
guarded — and `Limitation.PATCH_SUPERSEDED` has an emitter for the first time.
`_units` and `_emit` arrive in later steps; `__all__` grows with them and does
not shrink.
"""

from ._builder import IntegrityBuilder, SpanDraft
from ._diag import Counters, counters, guard
from ._integrity import Limitation
from ._parentage import (
    AMBIENT,
    EMPTY_AMBIENT,
    Ambient,
    Evidence,
    Parentage,
    ParentSource,
    child_of,
    latch_ambient,
    resolve_parentage,
)
from ._patchset import PatchSet
from ._policy import Prefilter, capture_mode_of, should_capture
from ._snapshot import SnapshotDraft
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
    "ParentSource",
    "Parentage",
    "PatchSet",
    "Prefilter",
    "SnapshotDraft",
    "SnapshotType",
    "SpanDraft",
    "SpanIntent",
    "TransportLabel",
    "VocabularyError",
    "capture_mode_of",
    "child_of",
    "counters",
    "guard",
    "is_declared_extra_key",
    "latch_ambient",
    "resolve_parentage",
    "should_capture",
    "vocabulary_name",
]
