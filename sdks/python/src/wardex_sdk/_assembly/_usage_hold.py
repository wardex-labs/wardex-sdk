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
module records as it happens: was a wire request issued under the unit?

* Yes: the demoted extras stay demoted. The wire span below carries the usage.
* No (a provider the interceptors do not see, a local test model, a call that failed before any
  byte left): the demoted numbers are the only ones there are, so they are PROMOTED into the typed
  block and the pairs removed — one number, one home — and the span says
  `Limitation.NO_WIRE_EVIDENCE`, which is literally its definition: assembled from framework
  callbacks, no interceptor saw the bytes.

Why "a request was issued" rather than "a wire CHAT was parsed": the parse is asynchronous and the
close is not. Waiting for the parser would make the chat span's shape depend on queue timing. The
witness is written synchronously where the seams LATCH the request's parent (`_bridge_latch.py`),
so the tree and the usage are decided by the same read and cannot disagree.

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
        #: A plain bool, set and never counted up: two seams witnessing one request (the h2
        #: issuer and the writer-scope fallback) cannot double anything.
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

    Its `<prefix>gen_ai.usage.*` extras are promoted into the typed block at close unless a wire
    request was issued under the unit meanwhile. Refused (False, counted) for a null draft, a
    non-CHAT intent, or a prefix not of the form `wardex.<something>.` — the prefix is what keeps a
    holder's promotion inside wardex's own extras namespace.

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
    """A request is being issued with `parent` as its latched parent. Called by the seams' latch,
    and by the h2 writer fallback for a stream whose issuer was not proven (its proven streams were
    witnessed by their own issuer's latch).

    Two module-global reads and nothing else when no bridge holds anything — this runs once per
    outgoing request on every seam.
    """
    if _HOLDERS and parent is not None:
        holder = _HOLDERS.get(parent.span_id)
        if holder is not None:
            holder.issued = True


def settle_usage(draft: SpanDraft) -> SpanDraft:
    """Decide a holder's usage at its close. Every other draft passes through untouched.

    Called from `UnitRegistry`'s finalize step under its lock; does not emit. The witness store
    happens-before this read: the request is issued while the framework's chat span is open, and
    the unit closes from that span's end.
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
