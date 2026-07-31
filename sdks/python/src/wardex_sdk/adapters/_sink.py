"""The `assembly.SpanSink` an adapter hands to its unit registry.

Its own module because it is the one piece of the Anthropic assembler that is
not about Anthropic. Design §4.4 puts this line — and the capture-mode gate that
belongs beside it — in `assembly/_emit.py`, so that one place decides whether a
span ships instead of each caller deciding for itself; `tests/test_import_graph.py`
reserves that name and asserts no module under `assembly/` reaches the sink
today. Standing alone here is what lets it move there whole rather than being
disentangled from an adapter's construction first.
"""

from __future__ import annotations

from typing import Any

from ..assembly import SpanDraft


class _ClientSink:
    """Materializes a draft and hands the span to the client.

    Materializing HERE rather than at each call site is what lets the registry
    own two-phase spans end to end: `UnitRegistry.close()` stamps the status and
    the end instant under its lock and hands the draft over AFTER releasing it
    (I11), and this is the last step. `finish()` may raise `VocabularyError`;
    the registry calls every sink inside `guard()`, so a breach is a counted,
    debug-logged deletion rather than an exception in the host's own hook
    callback.

    A class rather than a closure over the hook wiring, for the same reason it
    is a module: a named object is something the capture-mode gate can be added
    to, where a lambda would have to be taken apart first.
    """

    __slots__ = ("_client",)

    def __init__(self, client: Any) -> None:
        self._client = client

    def emit(self, draft: SpanDraft, *, agent_semantic: bool) -> bool:
        if self._client is None:
            return False
        self._client.capture_span(draft.finish())
        return True
