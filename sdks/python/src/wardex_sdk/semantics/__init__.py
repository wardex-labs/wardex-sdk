"""Protocol semantics — parsed wire data mapped onto span vocabulary.

This package answers one question: what did those bytes MEAN. The byte seam
(`interceptors/_seam.py`) owns the mechanics — which socket, which tracker,
which transaction, when to consume the timing record — and every question that
begins "what does this parsed body mean in gen_ai / rpc / websocket terms" is
answered here instead, by a pure function over a parser result.

Layering (design §3.1). `semantics/` is a sibling of `interceptors/` and
`adapters/`, above `assembly/` and the leaf vocabulary, and it imports NEITHER
sibling: it reads `protocol/` (the Rust parser bindings) and writes
`assembly/`'s closed vocabularies, and that is its whole dependency surface. It
therefore never sees a `_Txn`, a `_ConnectionState` or a `Client` as a type —
`build_grpc_fields` takes its transaction as a structural `Any` rather than
naming the seam's record, because naming it would be the upward import this
boundary exists to forbid.

These are pure in the sense that matters — no seam state, no connection, no
client, no I/O — which is what lets them be tested without a socket. They are
NOT free of error handling: `build_grpc_fields` carries a bare `except
Exception:` around frame parsing, and it moved here with the function. Saying
otherwise would be worse than the swallow itself, because it is the sentence a
reviewer would trust instead of looking. It is counted, not waved through —
`tests/test_import_graph.py` scopes its silent-swallow ratchet over this package
and budgets `semantics/_grpc.py` at exactly one. Converting it to
`assembly._diag.guard` is a behaviour change — the swallow starts counting and,
in debug, starts printing — so it belongs with the sweep that converts every
remaining bare `except` in the SDK, not with a move that only relocated this
one.

`__all__` is the four mappings the byte seam asks for, and deliberately nothing
else:

  `build_gen_ai` / `has_core_semantics` — the gen_ai pair, exported together
  because the second is the gate on the first. A parse result carrying neither a
  response model nor a token count is not agent traffic, and the capture policy
  asks that question BEFORE any attribute is built; splitting them would let a
  caller build attributes for something the policy would have dropped.

  `build_grpc_fields` — the gRPC branch, the one with enough protocol logic to
  be worth testing on its own.

  `ws_close_name` — the WebSocket close code to `error.type` mapping.

`_OPERATION_MAP` and `_PROVIDER_MAP` stay module-private on purpose. They are
lookup tables WITH a fallback: `dict.get(value, value)` passes an unenumerated
provider straight through, which is how a provider wardex has never heard of
still gets a span instead of a `KeyError`. A caller holding the bare table gets
the lookup and none of the fallback, so the mapping is the export and the table
is not.
"""

from ._genai import build_gen_ai, has_core_semantics
from ._grpc import build_grpc_fields
from ._ws import ws_close_name

__all__ = [
    "build_gen_ai",
    "build_grpc_fields",
    "has_core_semantics",
    "ws_close_name",
]
