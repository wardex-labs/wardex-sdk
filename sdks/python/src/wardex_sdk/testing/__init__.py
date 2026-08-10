"""Conformance machinery for framework adapters — shipped, not test-only.

An adapter's obligations are a product claim, not a detail of this repository's
test layout: the causal tree comes from in-process context propagation, every
span site declares whether it may begin a trace, an install is exactly undone,
and a shutdown mid-run ships the run instead of dropping it. Anyone writing an
adapter needs to be able to check those, so the suite ships in the wheel rather
than living in `tests/`.

It has no third-party dependency and imports no test framework: the checks are
plain `assert` statements, so they read the same under pytest, under unittest
and in a script.

    from wardex_sdk.testing import AdapterConformanceSuite, AdapterSubject

See `conformance.py` for what makes this a gate rather than a checklist.
"""

from .conformance import AdapterConformanceSuite
from .harness import (
    AdapterSubject,
    LiveAdapter,
    RecordingClient,
    SpanNode,
    StalledRun,
    clean_state,
    collapse_onto_root,
    exactly_one,
    installed_adapter,
    never_installed,
    parent_name_of,
    read_spans,
)

__all__ = [
    "AdapterConformanceSuite",
    "AdapterSubject",
    "LiveAdapter",
    "RecordingClient",
    "SpanNode",
    "StalledRun",
    "clean_state",
    "collapse_onto_root",
    "exactly_one",
    "installed_adapter",
    "never_installed",
    "parent_name_of",
    "read_spans",
]
