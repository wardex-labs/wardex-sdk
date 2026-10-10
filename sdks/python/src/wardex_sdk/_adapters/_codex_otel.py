"""What one `codex exec` run's own OTel traces say about its model calls.

A pure function over the spans the loopback receiver filed for one run: no I/O,
no clock, no wardex state. The adapter turns the answer into spans; this module
only reads.

The shape it reads was measured on `codex-cli 0.160.0` (Responses API over a
WebSocket, ChatGPT sign-in), and Codex documents none of it — span names and
attributes are its internals, not a contract. So every rule below is stated as
the observation it rests on, and a run whose spans match none of them is
reported as RECOGNIZED NOTHING rather than guessed at:

* One model call is one ``try_run_sampling_request`` span. A retried request
  is a second one under the same ``run_sampling_request``, which is right: it
  was a second request. The span carries ``model``.
* That call's usage rides a descendant ``handle_responses`` span as
  ``gen_ai.usage.input_tokens`` (cached part inside it, the OpenAI
  convention), ``gen_ai.usage.cache_read.input_tokens``,
  ``gen_ai.usage.cache_write.input_tokens``, ``gen_ai.usage.output_tokens``
  and ``codex.usage.reasoning_output_tokens``. Measured: the calls' usages sum
  exactly to the ``turn.completed`` usage the ``--json`` stream reports.
* The warm-up request Codex sends at session start is a
  ``model_client.stream_responses_websocket`` span with
  ``websocket.warmup = true`` under ``startup_prewarm``, never under a
  sampling request. Its token counts appear in Codex's LOG events only, which
  the bridge does not enable (they carry the account's e-mail address), so
  the warm-up is reported as having happened and how long it took — never as
  a call with usage.
* The model provider is ``model_client.websocket_connection``'s ``provider``
  (``"OpenAI"`` under ChatGPT sign-in). Absent, the provider stays unknown.
* The Codex version is the resource's ``service.version``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

#: The one Codex release this reading was measured against.
VERIFIED_VERSION = "0.160.0"

_CALL = "try_run_sampling_request"
_USAGE = "handle_responses"
_CLIENT = "model_client.stream_responses_websocket"
_CONNECTION = "model_client.websocket_connection"


@dataclass(frozen=True, slots=True)
class CodexCall:
    """One model request, as Codex measured it inside its own process."""

    start_ns: int
    end_ns: int
    model: str | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    #: OTLP status code of the call's span; 2 is ERROR.
    status_code: int = 0


@dataclass(frozen=True, slots=True)
class CodexWarmup:
    start_ns: int
    end_ns: int


@dataclass(frozen=True, slots=True)
class CodexBridgeView:
    calls: tuple[CodexCall, ...] = ()
    warmups: tuple[CodexWarmup, ...] = ()
    version: str | None = None
    #: The provider as Codex names it (`"OpenAI"`), or None.
    provider: str | None = None
    #: How many spans arrived at all — the census `recognized` is judged by.
    seen: int = 0

    @property
    def recognized(self) -> bool:
        """Whether anything here is a model call. False with `seen > 0` is the
        schema-drift signal: telemetry arrived and meant nothing to this reader."""
        return bool(self.calls)


def _int(value: Any) -> int | None:
    """An OTLP int attribute; Codex's JSON exporter spells int64 as a string."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _ns(raw: Mapping[str, Any], key: str) -> int:
    return _int(raw.get(key)) or 0


def _attrs(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    attrs = raw.get("attributes")
    return attrs if isinstance(attrs, Mapping) else {}


def classify(spans: Iterable[Mapping[str, Any]], resource: Mapping[str, Any]) -> CodexBridgeView:
    spans = [s for s in spans if isinstance(s, Mapping)]
    children: dict[str, list[Mapping[str, Any]]] = {}
    for span in spans:
        parent = span.get("parent_span_id") or ""
        children.setdefault(str(parent), []).append(span)

    def descendants(span: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
        stack = list(children.get(str(span.get("span_id") or ""), ()))
        seen: set[str] = set()
        while stack:
            node = stack.pop()
            key = str(node.get("span_id") or "")
            if key in seen:  # a malformed export may cycle; reading must still end
                continue
            seen.add(key)
            yield node
            stack.extend(children.get(key, ()))

    calls: list[CodexCall] = []
    for span in spans:
        if span.get("name") != _CALL:
            continue
        attrs = _attrs(span)
        model = attrs.get("model")
        usage: Mapping[str, Any] = {}
        for node in descendants(span):
            node_attrs = _attrs(node)
            if node.get("name") == _USAGE and "gen_ai.usage.input_tokens" in node_attrs:
                usage = node_attrs
            elif node.get("name") == _CLIENT and model is None:
                model = node_attrs.get("model")
        status = span.get("status")
        calls.append(
            CodexCall(
                start_ns=_ns(span, "start_time_unix_nano"),
                end_ns=_ns(span, "end_time_unix_nano"),
                model=model if isinstance(model, str) and model else None,
                input_tokens=_int(usage.get("gen_ai.usage.input_tokens")),
                output_tokens=_int(usage.get("gen_ai.usage.output_tokens")),
                cache_read_tokens=_int(usage.get("gen_ai.usage.cache_read.input_tokens")),
                cache_creation_tokens=_int(usage.get("gen_ai.usage.cache_write.input_tokens")),
                reasoning_output_tokens=_int(usage.get("codex.usage.reasoning_output_tokens")),
                status_code=(_int(status.get("code")) or 0) if isinstance(status, Mapping) else 0,
            )
        )
    calls.sort(key=lambda c: c.start_ns)

    warmups: list[CodexWarmup] = []
    for span in spans:
        if span.get("name") != _CLIENT or _attrs(span).get("websocket.warmup") is not True:
            continue
        # The client span itself is opened and closed around the send; the
        # request's own duration is its child's.
        start, end = _ns(span, "start_time_unix_nano"), _ns(span, "end_time_unix_nano")
        for node in descendants(span):
            start = min(start, _ns(node, "start_time_unix_nano"))
            end = max(end, _ns(node, "end_time_unix_nano"))
        warmups.append(CodexWarmup(start_ns=start, end_ns=end))
    warmups.sort(key=lambda w: w.start_ns)

    provider = None
    for span in spans:
        if span.get("name") == _CONNECTION:
            value = _attrs(span).get("provider")
            if isinstance(value, str) and value:
                provider = value
                break

    version = resource.get("service.version") if isinstance(resource, Mapping) else None
    return CodexBridgeView(
        calls=tuple(calls),
        warmups=tuple(warmups),
        version=version if isinstance(version, str) and version else None,
        provider=provider,
        seen=len(spans),
    )
