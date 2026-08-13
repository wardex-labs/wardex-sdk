"""The single constructor of `InternalSpan` — design §4.3, invariant I5.

Every span the SDK emits is built here. Not "should be" — `tests/test_import_graph.py`
asserts that `InternalSpan(...)` appears in exactly one module outside `_types.py`,
as a hard rule rather than a budget, from the commit that lands this file.

**Why a draft with typed setters and not a `build(...)` free function.** The
rejected alternative (design §13-D2) was a single function taking every field.
It diagnoses the god-parameter problem in the gate it replaces and then rebuilds
the same shape at 22 arguments — where "which of these are mandatory together"
is a comment rather than a check, and where every new field is a new positional
question at eleven call sites. A draft answers that structurally: a site sets
what it observed, `finish()` decides whether what it observed is a legal span.

**What one constructor buys, concretely.** Before this file the tree held two
`execute_tool` shapes — the adapter's, with `capture_sources`, `capture_integrity`
and `correlation`, and `_tracing.span()`'s, with none of the three — because
they were built in two places by two people. It held a span literally named
`"chat None"`, produced by interpolating a model the stream never reported. And
it held `request_body_captured=bool(tool.input_data)`, which reports a tool
called with `{}` as a capture FAILURE on the field a dashboard reads to decide
whether a replay is trustworthy. None of the three is reachable from here:
`capture_sources` is always set, the name is built by the grammar, and
`IntegrityBuilder` has no way to spell the old meaning of `*_captured`.

`finish()` raises `VocabularyError`, and a raised `VocabularyError` DELETES the
span — `guard()` swallows it and only a counter is left. That is the cost of a
closed vocabulary, it is paid deliberately, and it is why §6.5.1's census had to
close `Limitation` at 37 members before any emit site was routed through here.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .._enums import CaptureSource, SpanKind, StatusCode
from .._types import (
    AgentAttributes,
    CallSite,
    CaptureIntegrity,
    ConversationContext,
    CorrelationInfo,
    EmbeddingsAttributes,
    EvaluationAttributes,
    GenAIAttributes,
    InternalSpan,
    InternalSpanEvent,
    InternalSpanLink,
    RetrievalAttributes,
    SpanContext,
    ToolAttributes,
    TransportAttributes,
)
from ._integrity import Limitation
from ._parentage import EMPTY_AMBIENT, Evidence, Parentage, ParentSource, resolve_parentage
from ._vocab import (
    Block,
    LinkReason,
    SpanIntent,
    TransportLabel,
    VocabularyError,
    is_declared_extra_key,
    transport_name,
    vocabulary_name,
)

_Scalar = str | int | float | bool

#: OTel semconv's sanctioned `error.type` fallback — "the instrumentation does
#: not define a value for this". It is what a MANUAL span gets when the host
#: marked it ERROR without naming a type; see `_manual_error_type`.
OTEL_ERROR_TYPE_OTHER = "_OTHER"

#: The name a MANUAL span gets when the host passed an empty one.
_MANUAL_NAME_FALLBACK = "span"


def _manual_span_name(name: str) -> str:
    """The host's name, or a placeholder if the host passed an empty one.

    `wardex.span(label or "")` is a real shape — a config value that was not
    set, a framework label that was missing — and `_check_name`'s empty-name
    refusal would DELETE that span, while the two things the caller wanted from
    it, its timing and its parentage, do not depend on the name at all. The
    refusal stays hard for the two modes wardex names itself, where an empty
    name means the grammar produced nothing.
    """
    return name or _MANUAL_NAME_FALLBACK


class IntegrityBuilder:
    """`CaptureIntegrity`, built so the wrong meaning cannot be expressed.

    `*_captured` means **"capture was attempted and it succeeded"**. It does NOT
    mean "the payload is non-empty" — and the difference is not academic: the
    adapter shipped `request_body_captured=bool(tool.input_data)`, so every tool
    invoked with `{}` reported a failed capture on the field the dashboard uses
    to judge replay confidence. The API below has no argument that would let a
    caller say that again: emptiness is not one of its inputs.

    `build()` returns `None` until something is actually recorded, so a span
    that has nothing to say about its own integrity says nothing rather than
    shipping a record of seven `False`s that reads as "we tried everything and
    failed at all of it".
    """

    __slots__ = (
        "_dropped_chunks",
        "_markers",
        "_redacted",
        "_req_body",
        "_req_headers",
        "_resp_body",
        "_resp_headers",
        "_touched",
        "_truncated",
    )

    def __init__(self) -> None:
        self._req_headers = False
        self._req_body = False
        self._resp_headers = False
        self._resp_body = False
        self._redacted = False
        self._truncated = False
        self._dropped_chunks = 0
        self._markers: list[Limitation] = []
        self._touched = False

    def request_headers(self, *, attempted: bool, ok: bool = True) -> IntegrityBuilder:
        self._touched = True
        self._req_headers = attempted and ok
        return self

    def request_body(self, *, attempted: bool, ok: bool = True) -> IntegrityBuilder:
        self._touched = True
        self._req_body = attempted and ok
        return self

    def response_headers(self, *, attempted: bool, ok: bool = True) -> IntegrityBuilder:
        self._touched = True
        self._resp_headers = attempted and ok
        return self

    def response_body(self, *, attempted: bool, ok: bool = True) -> IntegrityBuilder:
        self._touched = True
        self._resp_body = attempted and ok
        return self

    def truncated(self, value: bool) -> IntegrityBuilder:
        self._touched = True
        self._truncated = bool(value)
        return self

    def redacted(self, value: bool) -> IntegrityBuilder:
        self._touched = True
        self._redacted = bool(value)
        return self

    def dropped_chunks(self, count: int) -> IntegrityBuilder:
        self._touched = True
        self._dropped_chunks = int(count)
        return self

    def limitation(self, marker: Limitation) -> IntegrityBuilder:
        """Attach a marker. Idempotent and order-preserving.

        Typed, so "a limitation not in the `Limitation` enum" is normally
        unrepresentable rather than merely rejected. `finish()` re-checks
        anyway: this object is reachable from `_interceptors/` and `_adapters/`,
        where a `str` sneaking through a duck-typed hop is exactly the failure
        the check exists for.

        Rebuilt rather than `.append(marker)`, and that is not style. The census
        scanner keys marker-taking helpers by BARE NAME, so a `.append(marker)`
        here would register the name `append` as a marker sink and make the
        scanner read every argument of every `.append(...)` call in the SDK —
        turning `_Txn(method="GET", ...)` into seven marker strings with no
        `Limitation` member. An assignment to a marker-ish name is the shape the
        scanner reads exactly, and only, as this slot.
        """
        self._touched = True
        if marker not in self._markers:
            self._markers = [*self._markers, marker]
        return self

    @property
    def markers(self) -> tuple[Limitation, ...]:
        return tuple(self._markers)

    def build(self) -> CaptureIntegrity | None:
        if not self._touched:
            return None
        return CaptureIntegrity(
            request_headers_captured=self._req_headers,
            request_body_captured=self._req_body,
            response_headers_captured=self._resp_headers,
            response_body_captured=self._resp_body,
            redacted=self._redacted,
            truncated=self._truncated,
            dropped_chunk_count=self._dropped_chunks,
            # Members, all the way to the encoder. Unwrapping to `.value` here
            # was a placeholder while the wire field was still
            # `repeated string`; now that it is `repeated Limitation`, keeping
            # the member means the type is checkable at every hop instead of
            # only at the two ends.
            limitations=tuple(self._markers),
        )


class SpanDraft:
    """A span under construction, and the only thing that becomes an `InternalSpan`.

    Built FROM a `Parentage`, so a draft cannot exist without a parentage
    decision having been made by `_parentage.py` (I1). `child_context()` is
    called exactly once, here, for the reason that method's docstring gives:
    calling it twice anchors children to a span id that is never emitted, and
    nothing downstream can detect it.

    Three naming modes, chosen at construction and never afterwards — see
    `_vocab.py` for why there are three. The plain constructor is the
    VOCABULARY mode of design §4.3; `transport()` and `manual()` have to be
    named, so a site cannot drift out of the closed grammar quietly.
    """

    __slots__ = (
        "_agent",
        "_call_site",
        "_conversation",
        "_correlation",
        "_cost_usd",
        "_embeddings",
        "_emitted",
        "_end_ns",
        "_evaluation",
        "_events",
        "_extra",
        "_gen_ai",
        "_integrity",
        "_intent",
        "_kind",
        "_label",
        "_links",
        "_manual_name",
        "_operation_label",
        "_parentage",
        "_retrieval",
        "_server_address",
        "_server_port",
        "_sources",
        "_start_ns",
        "_status",
        "_status_message",
        "_error_type",
        "_input_data",
        "_output_data",
        "_subject",
        "_tool",
        "_transport",
        "_workflow_name",
        "context",
    )

    def __init__(
        self,
        parentage: Parentage,
        *,
        intent: SpanIntent | None,
        subject: str | None = None,
        source: CaptureSource,
        start_ns: int,
        kind: SpanKind | None = None,
    ) -> None:
        self.context: SpanContext = parentage.child_context()
        self._parentage = parentage
        self._intent = intent
        self._subject = subject
        self._label: TransportLabel | None = None
        self._manual_name: str | None = None
        self._kind = kind if kind is not None else (intent.default_kind if intent else None)
        self._start_ns = start_ns
        self._end_ns: int | None = None
        self._sources: list[CaptureSource] = [source]
        self._integrity = IntegrityBuilder()

        self._gen_ai: GenAIAttributes | None = None
        self._agent: AgentAttributes | None = None
        self._tool: ToolAttributes | None = None
        self._retrieval: RetrievalAttributes | None = None
        self._embeddings: EmbeddingsAttributes | None = None
        self._evaluation: EvaluationAttributes | None = None
        self._transport: TransportAttributes | None = None
        self._conversation: ConversationContext | None = parentage.conversation
        self._correlation: CorrelationInfo | None = parentage.correlation
        self._call_site: CallSite | None = None
        self._cost_usd: float | None = None
        self._workflow_name: str | None = None
        self._server_address: str | None = None
        self._server_port: int | None = None
        self._operation_label: str | None = None

        self._status = StatusCode.UNSET
        self._status_message = ""
        self._error_type: str | None = None
        self._input_data = b""
        self._output_data = b""
        self._extra: list[tuple[str, _Scalar]] = []
        self._links: list[InternalSpanLink] = []
        self._events: list[InternalSpanEvent] = []
        self._emitted = False

        # THE EDGE'S OWN MARKERS, carried here and not by each caller. A draft is
        # built FROM a parentage, so an interpreted edge — `unit_sole`,
        # `unresolved`, a recorded `correlation_conflict` — arrives already
        # knowing what it is; leaving the caller to copy them across is how a
        # span ships a confidence below 1.0 with an EMPTY limitation list, which
        # is half of I4 missing and the half a dashboard renders. Two of the six
        # parentage sites remembered to do it and the rest did not.
        for inherited in parentage.limitations:
            self._integrity.limitation(inherited)

    # -- the two non-vocabulary modes ------------------------------------

    @classmethod
    def transport(
        cls,
        parentage: Parentage,
        *,
        label: TransportLabel,
        subject: str | None,
        source: CaptureSource,
        start_ns: int,
        kind: SpanKind = SpanKind.CLIENT,
    ) -> SpanDraft:
        """A byte-seam observation wardex did not interpret as an operation.

        `HTTP POST /v1/messages` is not `chat` — the seam knows a request
        happened, and on the branches where it also parses LLM semantics it
        attaches a `gen_ai` block, but the NAME still reports what was observed
        rather than what it was guessed to mean. §6.2's twelve intents have no
        member for "uninterpreted traffic", and inventing one would be the
        protocol-branching that section rules out.
        """
        draft = cls(
            parentage, intent=None, subject=subject, source=source, start_ns=start_ns, kind=kind
        )
        draft._label = label
        return draft

    @classmethod
    def manual(
        cls,
        parentage: Parentage,
        *,
        name: str,
        kind: SpanKind,
        start_ns: int,
        source: CaptureSource = CaptureSource.MANUAL,
    ) -> SpanDraft:
        """A span the HOST named, via the published `wardex.span()` API.

        The name is free-form because the API says it is. What this mode does
        NOT do is exempt the span from anything else: it goes through the same
        `finish()`, gets `capture_sources` like every other span, and gains the
        `correlation` the `_tracing` path used to omit.
        """
        draft = cls(
            parentage, intent=None, subject=None, source=source, start_ns=start_ns, kind=kind
        )
        draft._manual_name = _manual_span_name(name)
        return draft

    def rename(self, name: str) -> None:
        """Re-name a MANUAL span. The host named it once; it may name it again.

        Restricted to MANUAL mode for the same reason `relabel()` is restricted
        to TRANSPORT: a VOCABULARY span's name IS its intent, and letting it be
        overwritten after the fact would make `_check_name`'s grammar assertion
        decorative.
        """
        if self._manual_name is None:
            raise VocabularyError("rename() is only meaningful for a MANUAL-mode draft")
        self._manual_name = _manual_span_name(name)

    def relabel(self, label: TransportLabel, subject: str | None) -> None:
        """Re-classify a TRANSPORT observation once the protocol is known.

        The byte seam cannot know a request is gRPC until it has parsed the
        frames, which happens after the draft exists. Restricted to TRANSPORT
        mode on purpose: a VOCABULARY span whose intent could change after the
        fact would make the required-block check meaningless, and a MANUAL span
        is named by the host.
        """
        if self._label is None:
            raise VocabularyError("relabel() is only meaningful for a TRANSPORT-mode draft")
        self._label = label
        self._subject = subject

    # -- typed setters ---------------------------------------------------

    def set_gen_ai(self, attrs: GenAIAttributes) -> None:
        self._gen_ai = attrs

    def set_agent(self, attrs: AgentAttributes) -> None:
        self._agent = attrs

    def set_tool(self, attrs: ToolAttributes) -> None:
        self._tool = attrs

    def set_retrieval(self, attrs: RetrievalAttributes) -> None:
        self._retrieval = attrs

    def set_embeddings(self, attrs: EmbeddingsAttributes) -> None:
        self._embeddings = attrs

    def set_evaluation(self, attrs: EvaluationAttributes) -> None:
        self._evaluation = attrs

    def set_transport(self, attrs: TransportAttributes) -> None:
        self._transport = attrs

    def set_conversation(self, conv: ConversationContext | None) -> None:
        self._conversation = conv

    def set_call_site(self, call_site: CallSite | None) -> None:
        self._call_site = call_site

    def set_workflow_name(self, name: str | None) -> None:
        self._workflow_name = name

    def set_cost_usd(self, cost: float | None) -> None:
        self._cost_usd = cost

    def set_server(self, address: str | None, port: int | None) -> None:
        self._server_address = address
        self._server_port = port

    def set_io(
        self,
        *,
        input_data: bytes = b"",
        output_data: bytes = b"",
        input_attempted: bool = True,
        output_attempted: bool = True,
    ) -> None:
        """Record the task I/O AND what capturing it was worth.

        The two are one call because they were two decisions in two places and
        they disagreed. `attempted` is what the site knows and emptiness is not:
        a tool called with `{}` attempted successfully and captured `b"{}"`.
        """
        self._input_data = input_data
        self._output_data = output_data
        self._integrity.request_body(attempted=input_attempted)
        self._integrity.response_body(attempted=output_attempted)

    def set_end_ns(self, end_ns: int) -> None:
        """Stamp the end instant on a TWO-PHASE span.

        A unit's span is opened by one framework callback and closed by another,
        and it is materialized later still — by the sink, which is the only
        caller of `finish()`. Without a slot for the end instant it would have to
        travel beside the draft in every table that holds one, which is exactly
        the two-object shape the adapter's subagent entry had before the draft
        itself became the thing held. `finish(end_ns=...)` still wins when given,
        so a one-phase site is unaffected.
        """
        self._end_ns = end_ns

    def set_start_ns(self, start_ns: int) -> None:
        """Replace the start instant on a TWO-PHASE span — `set_end_ns`'s mirror.

        For the one caller shaped like this: the OTel bridge holds an
        assembler-built draft PENDING at session close and replaces its
        IPC-approximated start with the instant the CLI measured inside its own
        process. A CORRECTION, never the first value — construction still
        requires `start_ns`, so a draft cannot exist without one — and
        `finish()`'s precedence rules are untouched.
        """
        self._start_ns = start_ns

    def set_status(self, code: StatusCode, message: str = "") -> None:
        self._status = code
        self._status_message = message

    def set_error(self, error_type: str | None, message: str = "") -> None:
        """Declare the failure. `finish()` refuses `status=ERROR` without one.

        This method exists because the adapter shipped `is_error=true` spans
        with no `error.type`, while the payload that carries the reason was
        sitting in the hook input the whole time.
        """
        self._error_type = error_type
        if message:
            self._status_message = message

    def add_limitation(self, marker: Limitation) -> None:
        self._integrity.limitation(marker)

    @property
    def integrity(self) -> IntegrityBuilder:
        """The integrity record under construction, for sites that need the
        full `attempted/ok` vocabulary rather than `set_io`'s common case."""
        return self._integrity

    def add_link(self, ctx: SpanContext, reason: LinkReason) -> None:
        """A CAUSAL edge — never a containment edge (§6.3).

        Nesting a handoff chain is what makes every duration in a flame graph
        wrong; the receiving agent is a SIBLING with a `HANDOFF_FROM` link.
        """
        # The MEMBER, not `reason.value`. This is the only production site that
        # builds an `InternalSpanLink`, so unwrapping here meant every link the
        # SDK emits carried a string where the type says `LinkReason` — and
        # `link.reason is LinkReason.HANDOFF_FROM`, the check a renderer needs
        # to draw a sibling instead of nesting, was always False.
        self._links.append(
            InternalSpanLink(trace_id=ctx.trace_id, span_id=ctx.span_id, reason=reason)
        )

    def add_event(self, name: str, ts_ns: int, **attrs: _Scalar) -> None:
        """§6.5 tier 2: a point-in-time fact inside the span."""
        self._events.append(
            InternalSpanEvent(
                name=name, timestamp_ns=ts_ns, attributes=tuple(sorted(attrs.items()))
            )
        )

    def add_source(self, source: CaptureSource) -> None:
        if source not in self._sources:
            self._sources.append(source)

    def set_operation_label(self, operation: Any) -> None:
        """Record `gen_ai.operation.name` for a MANUAL span.

        A label, not an intent: the host said "call this an `execute_tool`" and
        wardex has no `ToolAttributes` to check it against, because the public
        decorator signature makes that argument optional. Closing that gap means
        changing a published API, which is not this constructor's to make.
        """
        if operation is None:
            self._operation_label = None
            return
        self._operation_label = getattr(operation, "value", operation)

    def set_extra(self, key: str, value: _Scalar) -> None:
        """Write a namespaced attribute. `finish()` rejects an undeclared key.

        MANUAL spans are exempt: `Span.set_attribute` is a published API
        that has always taken any key, and rejecting one now would delete a
        user's span to enforce a namespace wardex has not yet given them a way
        to declare (§6.5 tier 1's `FRAMEWORK_EXTRAS` does not exist yet).
        """
        self._extra.append((key, value))

    def replace_correlation(self, correlation: CorrelationInfo | None) -> None:
        """Override the edge's own `CorrelationInfo`, or withhold it entirely.

        DEBT, and it is here so the debt is countable. Two adapter cases need
        it, and both disappear once every adapter edge is opened by the unit
        registry against a context a pin delivered, so `resolve()` can supply
        evidence for that edge instead of the adapter guessing:

          * `None` — the edge was picked by a heuristic outside the parentage
            core (`_resolve_subagent_anchor`'s fallback, `_session_for_hook`'s
            sole-live-session guess). `Parentage.correlation` would report it as
            `unit_active` at confidence 1.0 with no marker, which is I4's exact
            prohibition: a claim wardex cannot back is worse than no claim.

          * a replacement — the tool span's own `CorrelationInfo`, carrying the
            framework's call id as a hint and the trust gap between the hook and
            stream paths as confidence, with `strategy=None` because this path
            cannot yet say how the parent was derived. It goes away with the
            same ingestion move, together with the two assertions that pin it;
            dropping the override here would republish the base edge's
            `unit_active`/1.0, which is the claim the first bullet forbids.
        """
        self._correlation = correlation

    def merge_correlation(self, extra: CorrelationInfo) -> None:
        """Fold framework-id hints into the resolved edge.

        Confidence is the MINIMUM of the two: a hint can only LOWER trust, never
        raise it, for the same reason `Evidence.confidence` can only lower the
        table default. The strategy stays the one the parentage core decided —
        a framework id is a lookup key, never a parentage source (I2).
        """
        base = self._correlation
        if base is None:
            self._correlation = extra
            return
        self._correlation = CorrelationInfo(
            operation_id=(
                extra.operation_id if extra.operation_id is not None else base.operation_id
            ),
            request_id=extra.request_id if extra.request_id is not None else base.request_id,
            attempt_id=extra.attempt_id if extra.attempt_id is not None else base.attempt_id,
            active_span_id_at_capture=base.active_span_id_at_capture,
            confidence=min(base.confidence, extra.confidence),
            strategy=base.strategy,
        )

    # -- materialize -----------------------------------------------------

    @property
    def name(self) -> str:
        if self._manual_name is not None:
            return self._manual_name
        if self._label is not None:
            return transport_name(self._label, self._subject)
        if self._intent is not None:
            return vocabulary_name(self._intent, self._subject)
        return ""

    def claim_emit(self) -> bool:
        """True exactly once per draft: may this draft be handed to a sink?

        The state a draft was missing. A draft's `context` — and therefore its
        span id — is decided at construction, and `finish()` re-materializes
        from it happily, so a second emit of the same draft puts TWO spans on
        the wire under ONE span id. Downstream that is not a duplicate, it is a
        contradiction: the second copy keeps the `CHILD_SPAN_UNCLOSED` a
        force-close added while reporting `status=OK` and a later end, and a
        backend doing last-write-wins shows a span that says both "completed
        successfully" and "was killed by someone else's teardown". Nothing after
        this point can undo it.

        Deliberately NOT folded into `finish()`. `finish()` is a pure
        materializer that tests and future readers call to inspect a draft;
        one-shotting it would make inspection destructive and would put the
        count at a place that cannot tell "materialized twice" from "emitted
        twice". The caller latches here and materializes there.
        """
        if self._emitted:
            return False
        self._emitted = True
        return True

    def finish(self, end_ns: int | None = None) -> InternalSpan:
        """Validate and materialize. Raises `VocabularyError` on a vocabulary breach.

        Never reaches the host: every caller builds inside `_diag.guard()`, so a
        breach is a counted, debug-logged, DELETED span (I6). That is the whole
        trade — a closed vocabulary is worth a class of span disappearing only
        because the vocabulary was censused shut first (§6.5.1).
        """
        end = end_ns if end_ns is not None else self._end_ns
        if end is None:
            raise VocabularyError("a span must be finished with an end timestamp")

        self._normalize_manual_error()

        name = self.name
        self._check_name(name)
        self._check_structure()
        self._check_status()
        self._check_conversation()
        self._check_extra()
        self._check_trace()

        gen_ai = self._gen_ai
        extra = list(self._extra)
        # The two branches that used to decide `gen_ai.operation.name`
        # separately, folded into one decision. An intent with a gen_ai block
        # MIRRORS into `GenAIAttributes.operation` (the codec flattens that key
        # out of the block); an intent without one stamps the key directly.
        # Writing both would put the key on the wire twice and force a
        # precedence rule nobody has — the same double-carry that keeps
        # `OperationName` out of `Span` as a typed field (§6.6, V11).
        if self._intent is not None:
            if gen_ai is not None:
                gen_ai = replace(gen_ai, operation=self._intent.operation)
            else:
                extra.insert(0, ("gen_ai.operation.name", self._intent.value))
        elif self._operation_label is not None and gen_ai is None:
            extra.insert(0, ("gen_ai.operation.name", self._operation_label))

        markers = self._integrity.markers
        bad = [m for m in markers if not isinstance(m, Limitation)]
        if bad:
            raise VocabularyError(f"limitation(s) outside the Limitation enum: {bad!r}")

        return InternalSpan(
            context=self.context,
            parent_span_id=self._parentage.parent_span_id,
            name=name,
            kind=self._kind if self._kind is not None else SpanKind.INTERNAL,
            start_time_ns=self._start_ns,
            end_time_ns=end,
            status=self._status,
            status_message=self._status_message,
            gen_ai=gen_ai,
            agent=self._agent,
            tool=self._tool,
            transport=self._transport,
            retrieval=self._retrieval,
            embeddings=self._embeddings,
            evaluation=self._evaluation,
            cost_usd=self._cost_usd,
            conversation=self._conversation,
            call_site=self._call_site,
            error_type=self._error_type,
            server_address=self._server_address,
            server_port=self._server_port,
            workflow_name=self._workflow_name,
            # Always set. The `_tracing.span()` path shipped spans with an empty
            # tuple here, which is why the tree held two structurally different
            # execute_tool shapes (§6.4).
            capture_sources=tuple(self._sources),
            capture_integrity=self._integrity.build(),
            correlation=self._correlation,
            input_data=self._input_data,
            output_data=self._output_data,
            extra=tuple(extra),
            events=tuple(self._events),
            links=tuple(self._links),
        )

    # -- normalization ---------------------------------------------------

    def _normalize_manual_error(self) -> None:
        """Keep a host-marked failure instead of deleting it.

        `Span.set_status(StatusCode.ERROR)` is published API and
        `StatusCode.ERROR` is its only non-OK member, so the ordinary way a host
        reports a failed operation would otherwise hit `_check_status` and
        delete the record of exactly the operation the host most wanted
        recorded. `Span.set_error` exists so a type CAN be given; this is
        what happens when it was not. `_OTHER` is OTel semconv's own fallback —
        "the instrumentation has no classification for this" — which is a true
        statement about a span whose failure only the host observed.

        MANUAL only. The two modes wardex names itself always know what failed
        (a gRPC status, an HTTP status, a JSON-RPC code, a WS close code), so
        for them the hard refusal in `_check_status` stays the check that makes
        an untyped failure impossible to reintroduce.
        """
        if self._manual_name is None:
            return
        if self._status is StatusCode.ERROR and not self._error_type:
            self._error_type = OTEL_ERROR_TYPE_OTHER

    # -- checks ----------------------------------------------------------

    def _check_name(self, name: str) -> None:
        if not name:
            raise VocabularyError("a span must have a name")
        if self._intent is None:
            return
        if name != vocabulary_name(self._intent, self._subject):
            raise VocabularyError(f"{name!r} does not follow the §6.1 grammar")
        if name.endswith(" None") or name == "None":
            raise VocabularyError(f"{name!r} interpolated a missing subject")

    def _check_structure(self) -> None:
        intent = self._intent
        if intent is None:
            return
        present = {
            Block.GEN_AI: self._gen_ai,
            Block.AGENT: self._agent,
            Block.TOOL: self._tool,
            Block.RETRIEVAL: self._retrieval,
            Block.EMBEDDINGS: self._embeddings,
            Block.EVALUATION: self._evaluation,
            Block.WORKFLOW_NAME: self._workflow_name,
        }
        missing = [b.value for b in intent.required_blocks if present[b] is None]
        if missing:
            raise VocabularyError(f"{intent.value} requires {missing}")
        keys = {k for k, _ in self._extra}
        missing_keys = [k for k in intent.required_extra_keys if k not in keys]
        if missing_keys:
            raise VocabularyError(f"{intent.value} requires extra key(s) {missing_keys}")

    def _check_status(self) -> None:
        if self._status is StatusCode.ERROR and not self._error_type:
            raise VocabularyError("status=ERROR requires an error_type (design §12 item 5)")

    def _check_conversation(self) -> None:
        conv = self._conversation
        if conv is None:
            return
        if not conv.conversation_id:
            # §6.3: an empty conversation id collides across every session in
            # any store that keys on it. If the framework does not supply one,
            # wardex issues one — it does not ship "".
            raise VocabularyError("conversation_id may not be empty (design §6.3)")

    def _check_extra(self) -> None:
        if self._manual_name is not None:
            return  # published API; see set_extra
        bad = sorted({k for k, _ in self._extra if not is_declared_extra_key(k)})
        if bad:
            raise VocabularyError(f"undeclared extra key(s): {bad} (design §6.5 tier 1)")

    def _check_trace(self) -> None:
        if self.context.trace_id != self._parentage.trace_id:
            raise VocabularyError("the draft's context left its parentage's trace")


#: The anchor a null draft answers `.context` with. Minted ONCE at import,
#: through the sanctioned factory — this adds no `TraceId.generate()` site, and
#: it names a span that is never emitted, which is exactly what a null anchor
#: means. A caller that reads it and hangs a child off it gets a subtree in a
#: trace nothing else is in, which is a visible loss rather than a subtree
#: silently grafted onto a live span.
_NULL_CONTEXT = resolve_parentage(EMPTY_AMBIENT, Evidence(ParentSource.UNRESOLVED)).child_context()


class _NullDraft:
    """Every verb reachable from a DEGRADED handle, as a total no-op.

    Exists so that "wardex failed here" costs the span and never the host. An
    adapter that has been handed a degraded scope keeps describing it — that is
    the whole point of not making the adapter branch — and every one of those
    calls has to land somewhere that cannot fail.

    NOT a `SpanDraft`, and deliberately not built like one. Constructing a real
    draft is `resolve_parentage()` + `SpanDraft(...)`, which is precisely the
    code whose failure produced the degraded handle in the first place: a lazy
    property would raise inside the host's `with` body and an eager one would
    raise out of `__enter__`, so the host would never run at all. `__slots__ =
    ()` and a module singleton leave nothing here that can fail.

    NO `finish()`, and that absence is structural rather than an oversight: a
    null draft may never reach a sink, and the only way to make that true by
    construction is for the materializer not to exist. `claim_emit()` answers
    False for the same reason — a sink that asks "may I emit this" gets an
    answer, not an `AttributeError`.
    """

    __slots__ = ()

    # -- SpanDraft, verb for verb ---------------------------------------
    def rename(self, name: str) -> None: ...
    def relabel(self, label: TransportLabel, subject: str | None) -> None: ...
    def set_gen_ai(self, attrs: GenAIAttributes) -> None: ...
    def set_agent(self, attrs: AgentAttributes) -> None: ...
    def set_tool(self, attrs: ToolAttributes) -> None: ...
    def set_retrieval(self, attrs: RetrievalAttributes) -> None: ...
    def set_embeddings(self, attrs: EmbeddingsAttributes) -> None: ...
    def set_evaluation(self, attrs: EvaluationAttributes) -> None: ...
    def set_transport(self, attrs: TransportAttributes) -> None: ...
    def set_conversation(self, conv: ConversationContext | None) -> None: ...
    def set_call_site(self, call_site: CallSite | None) -> None: ...
    def set_workflow_name(self, name: str | None) -> None: ...
    def set_cost_usd(self, cost: float | None) -> None: ...
    def set_server(self, address: str | None, port: int | None) -> None: ...
    def set_io(self, **kw: Any) -> None: ...
    def set_end_ns(self, end_ns: int) -> None: ...
    def set_start_ns(self, start_ns: int) -> None: ...
    def set_status(self, code: StatusCode, message: str = "") -> None: ...
    def set_error(self, error_type: str | None, message: str = "") -> None: ...
    def set_operation_label(self, operation: Any) -> None: ...
    def set_extra(self, key: str, value: _Scalar) -> None: ...
    def add_limitation(self, marker: Limitation) -> None: ...
    def add_link(self, ctx: SpanContext, reason: LinkReason) -> None: ...
    def add_event(self, name: str, ts_ns: int, **attrs: _Scalar) -> None: ...
    def add_source(self, source: CaptureSource) -> None: ...
    def replace_correlation(self, correlation: CorrelationInfo | None) -> None: ...
    def merge_correlation(self, extra: CorrelationInfo) -> None: ...

    # -- IntegrityBuilder, chained the way the real one chains -----------
    def request_headers(self, *, attempted: bool, ok: bool = True) -> _NullDraft:
        return self

    def request_body(self, *, attempted: bool, ok: bool = True) -> _NullDraft:
        return self

    def response_headers(self, *, attempted: bool, ok: bool = True) -> _NullDraft:
        return self

    def response_body(self, *, attempted: bool, ok: bool = True) -> _NullDraft:
        return self

    def truncated(self, value: bool) -> _NullDraft:
        return self

    def redacted(self, value: bool) -> _NullDraft:
        return self

    def dropped_chunks(self, count: int) -> _NullDraft:
        return self

    def limitation(self, marker: Limitation) -> _NullDraft:
        return self

    def build(self) -> CaptureIntegrity | None:
        return None

    @property
    def markers(self) -> tuple[Limitation, ...]:
        return ()

    # -- the three readers a caller may still reach ----------------------
    @property
    def integrity(self) -> _NullDraft:
        return self

    @property
    def context(self) -> SpanContext:
        return _NULL_CONTEXT

    @property
    def name(self) -> str:
        return ""

    def claim_emit(self) -> bool:
        return False


#: The one instance. Identity is load-bearing: `Scope.close_child` short-circuits
#: on `draft is NULL_DRAFT`, so a degraded child draft cannot be handed to a
#: registry that would try to close a span that was never opened.
NULL_DRAFT = _NullDraft()


__all__ = ["NULL_DRAFT", "OTEL_ERROR_TYPE_OTHER", "IntegrityBuilder", "SpanDraft"]
