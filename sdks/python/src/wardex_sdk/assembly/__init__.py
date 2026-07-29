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
/ `child_of`, so `_parentage` is live on every span the SDK emits. `_diag.guard`
still has no caller outside this package — `interceptors/` and `adapters/` adopt
it with the seam decomposition. `_units`, `_builder`, `_vocab`, `_policy`,
`_emit`, `_snapshot` and `_patchset` arrive in later steps; `__all__` grows with
them and does not shrink.
"""

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

__all__ = [
    "AMBIENT",
    "Ambient",
    "Counters",
    "EMPTY_AMBIENT",
    "Evidence",
    "Limitation",
    "ParentSource",
    "Parentage",
    "child_of",
    "counters",
    "guard",
    "latch_ambient",
    "resolve_parentage",
]
