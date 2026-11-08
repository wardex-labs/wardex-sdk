"""Framework-reported usage on a bridge CHAT unit: kept aside, promoted only when no wire saw it.

A bridge adapter turns a framework's own `chat` span into a CHAT unit, and the framework span
carries token usage. So does the wire CHAT span the byte seams record from the same model call, and
that span nests UNDER the bridge's unit by ordinary context propagation. Shipping both numbers in
the typed usage block counts one call twice: every sum over a trace doubles. The wire span is the
authority (it read the provider's own response bytes); the framework's number is a report about
them.

So the adapter writes the framework's usage as DEMOTED extras under its own namespace
(`wardex.<adapter>.otel.gen_ai.usage.*`, the value kept, the name moved) and declares the unit a
*framework-reported usage holder* here. What happens at the unit's close depends on one fact this
module records as it happens: was a transaction to a model-call endpoint SEALED under the unit?

* Yes: the demoted extras stay demoted. The wire span below carries the usage.
* No (a provider the interceptors do not see, an endpoint the parser does not read as a model call
  — Bedrock InvokeModel, a custom model's own API —, a local test model, a call that failed before
  any byte left): the demoted numbers are the only ones there are, so they are PROMOTED into the
  typed block and the pairs removed — one number, one home — and the span says
  `Limitation.NO_WIRE_EVIDENCE`, which is literally its definition: assembled from framework
  callbacks, no interceptor read the bytes as a model call.

Why "a model-call transaction was sealed" rather than "a request was issued" or "a wire CHAT was
parsed". Issued is too early: a request to an endpoint the parser does not read ships as a plain
HTTP span with no usage, and the numbers would stay demoted with nowhere typed to live. Parsed is
too late: the parse is asynchronous and the close is not, so the chat span's shape would depend on
queue timing. The seal (`_interceptors/_seam.py::_seal`) sits between them: it runs synchronously
on the thread that read the response's last byte, BEFORE the client hands that byte to the app and
so before the framework's chat span can end, and it asks the same endpoint table the parser is
gated on (`classify_path`), which is a lookup, not a parse. The parent it witnesses is the one the
tracker latched at issue (`_bridge_latch.py`), the parent the wire span is built under.

What follows from witnessing at the seal, stated so nobody mistakes it for a defect fixed later:

* A request that got no complete response (timeout, cancellation, a stream the app abandoned) is
  not witnessed before the unit closes: the framework's numbers for that call are promoted with
  `NO_WIRE_EVIDENCE`. An abandoned stream may still be sealed later, at connection close, as a
  partial span; usage arrives in a stream's last chunk, so that span normally carries none.
* A 4xx/5xx from a model-call endpoint IS witnessed: the wire span exists and carries the failure.
* A WebSocket session is one span, sealed at connection close, and never goes through `_seal`: a
  chat over a WebSocket transport is never witnessed and its framework numbers are promoted. The
  WebSocket span carries no `gen_ai` block and no usage, so nothing is counted twice.
* A sealed transaction the capture policy then refuses keeps the usage demoted with no wire span
  under it. Under the default mode it cannot happen here: the latched parent is a live local unit,
  which the policy admits.

Why a side table keyed by span id and not a field on `Unit`: the flag is one bridge's concern and
`Unit` is shared by every adapter; the table is empty — and every check here a single falsy read —
in a process where no bridge holds anything. Framework-neutral on purpose: nothing here names a
framework, so the next bridge reuses it with no core edit.

Imported by `_units` (the close path calls `settle_usage`), so this module must never import
`_units` back.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from ._builder import NULL_DRAFT
from ._diag import counters, guard
from ._integrity import Limitation
from ._vocab import SpanIntent

if TYPE_CHECKING:
    from .._types import SpanContext, SpanId
    from ._builder import SpanDraft

#: The usage keys a holder's demoted extras may be promoted from, after the holder's prefix, and
#: the typed field each one fills. CLOSED. The dotted spellings are OpenTelemetry's GenAI semconv
#: (what pydantic-ai emits); the underscore spellings are the ones the Agent SDK CLI bridge reads,
#: listed so that bridge needs no core edit either. `gen_ai.usage.details.*` is deliberately absent:
#: it has no typed field and stays a demoted extra.
_PROMOTED: dict[str, str] = {
    "gen_ai.usage.input_tokens": "input_tokens",
    "gen_ai.usage.output_tokens": "output_tokens",
    "gen_ai.usage.cache_read.input_tokens": "cache_read_input_tokens",
    "gen_ai.usage.cache_read_input_tokens": "cache_read_input_tokens",
    "gen_ai.usage.cache_creation.input_tokens": "cache_creation_input_tokens",
    "gen_ai.usage.cache_creation_input_tokens": "cache_creation_input_tokens",
    "gen_ai.usage.reasoning.output_tokens": "reasoning_output_tokens",
}


class _Holder:
    __slots__ = ("issued", "prefix")

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        #: A plain bool, set and never counted up: several model calls sealed under one unit (a
        #: retry, say) cannot double anything.
        self.issued = False


#: Holders, keyed by the unit draft's own `context.span_id`. Every unit close passes
#: `settle_usage`, which pops its entry; the one path that drops units without closing them is the
#: registry's fork reset, which is why this table has one of its own.
_HOLDERS: dict[SpanId, _Holder] = {}


def _at_fork_reinit() -> None:
    """Fork-child reset: the child's registries drop their inherited units without emitting them,
    so nothing would ever settle the parent's `_HOLDERS` entries in the child."""
    _HOLDERS.clear()


def hold_framework_usage(draft: SpanDraft, *, prefix: str) -> bool:
    """Declare `draft` — a UNIT's own draft, CHAT intent — a framework-reported usage holder.

    Its `<prefix>gen_ai.usage.*` extras are promoted into the typed block at close unless a
    model-call transaction was sealed under the unit meanwhile. Refused (False, counted) for a null
    draft, a non-CHAT intent, or a prefix not of the form `wardex.<something>.` — the prefix is
    what keeps a holder's promotion inside wardex's own extras namespace.

    Only a unit's own draft (`RunHandle.draft`): a `child_draft` never passes the unit close path,
    so its entry would never be settled and would sit here until the process ends or forks.
    """
    if (
        draft is NULL_DRAFT
        or getattr(draft, "_intent", None) is not SpanIntent.CHAT
        or not (prefix.startswith("wardex.") and prefix.endswith(".") and len(prefix) > 8)
    ):
        counters.bump("assembly._usage_hold.hold_refused")
        return False
    _HOLDERS[draft.context.span_id] = _Holder(prefix)
    return True


def witness_request(parent: SpanContext | None) -> None:
    """A transaction to a model-call endpoint was sealed with `parent` as its latched parent.

    Called by the byte seam's seal (`_interceptors/_seam.py::_seal`), synchronously on the thread
    that completed the transaction, for a path the endpoint table classifies `llm_call` — never
    from the parse, which is asynchronous. Two module-global reads and nothing else when no bridge
    holds anything: this runs once per sealed model call on every seam.
    """
    if _HOLDERS and parent is not None:
        holder = _HOLDERS.get(parent.span_id)
        if holder is not None:
            holder.issued = True


def settle_usage(draft: SpanDraft) -> SpanDraft:
    """Decide a holder's usage at its close. Every other draft passes through untouched.

    Called from `UnitRegistry`'s finalize step under its lock; does not emit. The witness store
    happens-before this read: the seal runs before the client returns the response's last byte,
    while the framework's chat span is open, and the unit closes from that span's end.
    """
    if not _HOLDERS:
        return draft
    holder = _HOLDERS.pop(draft.context.span_id, None)
    if holder is None:
        return draft
    with guard("assembly._usage_hold.settle"):  # a fault costs the promotion, never the span
        if holder.issued:
            # The wire span under this unit carries the usage; it is the authority.
            counters.bump("assembly._usage_hold.usage_kept_demoted")
        else:
            _promote(draft, holder.prefix)
            draft.add_limitation(Limitation.NO_WIRE_EVIDENCE)
            counters.bump("assembly._usage_hold.usage_promoted")
    return draft


def _promote(draft: SpanDraft, prefix: str) -> None:
    """Move the demoted usage pairs into the typed block, and remove exactly the pairs moved.

    A value that is not a plain int (a bool is an int to Python, and never a token count), or a
    typed field the block already filled, is left as a demoted extra and counted: the block is
    never overwritten, because a number already there came from something that read it first.
    """
    block = draft._gen_ai
    fields: dict[str, int] = {}
    moved: set[str] = set()
    for key, value in draft._extra:
        field = _PROMOTED.get(key[len(prefix) :]) if key.startswith(prefix) else None
        if field is None:
            continue
        if (
            block is None
            or not isinstance(value, int)
            or isinstance(value, bool)
            or field in fields
            or getattr(block, field) is not None
        ):
            counters.bump("assembly._usage_hold.promote_skipped")
            continue
        fields[field] = value
        moved.add(key)
    if block is None or not fields:
        return
    # `dataclasses.replace` is not a construction site (the usage-construction census says why);
    # `set_gen_ai` runs the inclusive-totals check on the result.
    draft.set_gen_ai(replace(block, **fields))
    draft._extra[:] = [pair for pair in draft._extra if pair[0] not in moved]


__all__ = ["hold_framework_usage", "settle_usage", "witness_request"]
