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

The other half is for applications rather than adapters: `assert_no_capture_gaps`
fails when wardex missed something it should have recorded, at the moment the
caller chooses (see `strict.py`). `pytest --wardex-strict` makes that check
around every test through `pytest_plugin.py`, which is the one module here that
imports a test framework. pytest reaches it through `_wardex_pytestplugin`, a
module outside this package, and only when the flag is given.
"""

from .conformance import AdapterConformanceSuite
from .harness import (
    AdapterSubject,
    LiveAdapter,
    RecordingClient,
    RecordingTransport,
    SpanNode,
    StalledRun,
    UsageSnapshot,
    clean_state,
    collapse_onto_root,
    exactly_one,
    installed_adapter,
    never_installed,
    parent_name_of,
    read_spans,
    read_usage,
)
from .strict import CaptureGap, CaptureGapError, assert_no_capture_gaps, capture_gaps

__all__ = [
    "AdapterConformanceSuite",
    "AdapterSubject",
    "CaptureGap",
    "CaptureGapError",
    "LiveAdapter",
    "RecordingClient",
    "RecordingTransport",
    "SpanNode",
    "StalledRun",
    "UsageSnapshot",
    "assert_no_capture_gaps",
    "capture_gaps",
    "clean_state",
    "collapse_onto_root",
    "exactly_one",
    "installed_adapter",
    "never_installed",
    "parent_name_of",
    "read_spans",
    "read_usage",
]
