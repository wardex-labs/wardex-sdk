"""What the byte seams decide from a request head, before any body is stored or parsed.

The byte seams ask the capture question at more than one moment (the comment below says when and
why). This module holds what a request head alone can answer, and the account a refusal leaves:

* `keeps_bodies`: could the policy use a request issued right now on this connection, whatever its
  response says? Everything `_should_capture` reads except the LLM-semantic claim. When the answer
  is no, the HTTP/1 parser stores a body only while the exchange can still be a model call.
* `named_in`: is this peer one the user named in `intercept_hosts`?
* `could_be_model_call`: could a parse find a model call in an exchange, whatever the policy?
* `count_refusal`: a refused call to a model API wardex does not parse yet is counted and said,
  never silent; under debug every refusal is said with why.
* `_accepted_prefix`: the part of a send buffer the kernel took, without copying the rest.
"""

from __future__ import annotations

import functools
from typing import Any

from .. import _hub, _wardex_native
from .._assembly import (
    DropReason,
    Prefilter,
    capture_mode_of,
    counters,
    debug_refused_call,
    in_degraded_run,
    lose,
    parent_is_closed_unit,
    should_capture,
)
from .._enums import CaptureMode

# --- Where "do we capture this?" is asked, and why at more than one moment ---
#
# The question has two halves, split by WHEN each can be answered rather than by what it means.
# Getting that split wrong is expensive in one direction and WRONG in the other, so it is written
# down here.
#
# INVARIANT — settled for the life of the connection, so the seam's `_capture_possible` answers them
# in the patch wrapper, before the send/recv buffer is copied and before a byte reaches a tracker:
#
#   * no client on this seam. Both emit paths already returned on a `None` client, so nothing behind
#     one was ever going to become a span — but the bodies were still accumulated in a tracker that
#     is kept for the life of the connection, which is memory held for an output that cannot exist.
#   * the self-exclusion flag. The exporter's own POST, which must never be re-captured whatever
#     else is true.
#   * a connection the sniff-latch already classified "ignore". `_gate` is latched once, from the
#     first bytes, and never re-decided — so a TLS-backed Redis, Postgres or Kafka connection is
#     settled for its whole life, and every call on one was still materializing its buffer
#     (`bytes(data)`, a real copy whenever the caller passes a memoryview or a bytearray, which is
#     the shape the asyncio and httpx paths use) to re-derive a verdict that was reached on the
#     first write.
#
# CONTEXT-DEPENDENT — an answer about ONE transaction, and only ever correct for the instant it was
# asked, so they live in the module `_should_capture` on the response side: the shared policy's
# ambient-span clause and the LLM-semantic claim `_parse_semantics` earns. An agent unit can
# ACTIVATE after a connection was opened — a pooled keep-alive socket outlives any single run — so
# a per-connection early-out on "no agent unit is ambient right now" would silently drop every
# later request on that socket. That is data loss, and the parse it would save is the one the gate
# needs.
#
# And a THIRD moment since the parse moved off the caller's thread: the gate now usually runs on
# the finalize WORKER, over a `_PendingTxn` sealed at the response instant. What keeps its answer
# honest there is the seal — the prefilter verdict, the mode, and `copy_context()` all travel with
# the job — so the gate reads the response instant's facts wherever and whenever it actually
# executes.
#
# And a FOURTH, first in time: everything the gate reads but the LLM-semantic claim is known when
# the request goes out, and a parse needs bodies only where the exchange can still be a model call
# (a request to an LLM endpoint path; a reply that declares an event stream or whose first bytes
# open one, with the JSON request it answers held until its reply says which). So `keeps_bodies`
# asks the rest per request, before its bytes reach the parser, and an exchange it cannot use has
# its bodies framed but never stored (`Http1Parser.set_body_policy`); `_seam._seal` asks again
# (`could_be_model_call`) and refuses there, before a parse is queued. Per request, never per
# connection, for the reason above.
#
# NOT here, deliberately: "the transport is a NoOpTransport". `init()` resolves a missing
# `transport=` argument to exactly that, so it is the out-of-the-box configuration rather than a
# statement that capture is off, and reading it as one would turn `wardex.init(intercept=True)`
# into a silent no-op.


def _accepted_prefix(data: Any, n: int) -> Any:
    """The first `n` bytes of a send buffer, without materializing the rest.

    `send`/`write` may report a SHORT write, and the caller then keeps the tail and calls again with
    the whole remainder: asyncio's plaintext writer does, with `send(self._buffer)` on 3.10/3.11
    and, from 3.12, `send` for a `write()`'s first attempt and `sendmsg` for the rest
    (`_socket._accepted_prefix_vectored`). `bytes(data)[:n]` copies that entire remainder before
    throwing away everything the kernel refused, so a multi-megabyte body costs a copy per call:
    quadratic in body size, and on the branch the gate deliberately leaves OPEN — a local plaintext
    model server is HTTP, so it latches "http" and pays this on every partial write.

    Returns something `bytes()` accepts rather than `bytes`, so that an exact `bytes` argument
    written in full stays the same object and costs nothing at all; the caller materializes once,
    immediately.

    The three arms are an isinstance chain rather than a `try`, because a handler here would be a
    silent swallow on a path that has a correct answer without one: anything that is not one of the
    three buffer types the socket API actually takes falls through to a plain `bytes(data)[:n]`.
    """
    if isinstance(data, bytes):
        return data[:n]
    if isinstance(data, (bytearray, memoryview)):
        # `.cast("B")` so `n` is read as bytes for an itemsize > 1 view too,
        # which is what the caller's `n` counts.
        return memoryview(data).cast("B")[:n]
    return bytes(data)[:n]


def keeps_bodies(prefilter: Prefilter, client: Any) -> bool:
    """Could the capture policy use a request issued now, under the connection's `prefilter`, by
    `client`'s mode, whatever its response turns out to be? Read per send on the host's thread,
    before the bytes reach the parser, because a request's head and body can arrive in one write:
    a few attribute reads and a ContextVar. Nothing here raises on its inputs (the policy is a pure
    function); anything that still escapes meets the patch wrapper that called the seam."""
    if prefilter is not Prefilter.DEFER:
        return prefilter is Prefilter.ALLOW
    mode = capture_mode_of(client)
    if mode is CaptureMode.ALL:
        return True
    parent = _hub.get_current_scope().active_span_context
    return should_capture(
        mode,
        parent=parent,
        agent_semantic=False,
        degraded=in_degraded_run(),
        parent_closed=parent is not None and parent_is_closed_unit(parent),
    )


def could_be_model_call(treatment: str | None, txn: Any) -> bool:
    """Could a parse find a model call in this exchange? Its target is an LLM endpoint (`treatment`,
    from `classify_path`), or its reply declares an event stream or its stored first bytes open one:
    a stream's grammar names its API on any path. The last is the question an HTTP/1 parser under
    "model_calls_only" answered to keep the reply, asked again over the same bytes; an HTTP/2
    reply is stored whole, so it is first asked here. Reads at most the first KiB of the reply,
    and nothing of one whose body was not stored."""
    if treatment == "llm_call" or txn.event_stream:
        return True
    body = txn.response_body
    return bool(body) and _wardex_native.protocol.opens_event_stream(body)


def named_in(allow: set[str], hostname: str | None, address: str, port: int) -> bool:
    """Is this peer one the user named (`allow`, case-folded): by the name the client connected
    with (a TLS server name, which is how a gateway on its own domain is known) or by its address,
    with or without the port?"""
    for name in (hostname, address):
        if name:
            name = name.lower()
            if name in allow or f"{name}:{port}" in allow:
                return True
    return False


#: Model APIs whose exchanges the semantic parser does not read, named from the request head alone:
#: the host, or a path shape only that API uses. Under `capture_mode=AGENT` an exchange outside any
#: span ships only when it parses as a model call, so a call to one of these is refused like
#: ordinary traffic; this table is what lets that refusal be counted and said rather than silent.
#: A row starting with a dot or a dash also matches any host ending in it.
_UNREAD_MODEL_HOSTS: tuple[tuple[str, str], ...] = (
    ("generativelanguage.googleapis.com", "the Google Gemini API"),
    ("aiplatform.googleapis.com", "Google Vertex AI"),
    ("-aiplatform.googleapis.com", "Google Vertex AI"),
    (".aiplatform.googleapis.com", "Google Vertex AI"),
    ("api.cohere.com", "the Cohere API"),
    ("api.cohere.ai", "the Cohere API"),
    ("api.mistral.ai", "the Mistral API"),
)
#: The last path segment of Gemini's and Vertex AI's generation methods (`models/{m}:{method}`).
_GEMINI_METHODS = (":generateContent", ":streamGenerateContent")
#: Bedrock Runtime's InvokeModel operations on a model (`/model/{id}/{operation}`): each model
#: family's own body shape. Converse and ConverseStream are read (the endpoint table's rows).
_BEDROCK_OPERATIONS = frozenset({"invoke", "invoke-with-response-stream"})
#: Ollama's native endpoints, at the end of the path (a proxy may prefix them). A path rather than
#: a host: Ollama runs wherever its user put it.
_OLLAMA_PATHS = ("/api/chat", "/api/generate")


@functools.lru_cache(maxsize=256)
def _unread_model_host(host: str) -> str | None:
    """The unread model API `host` belongs to, or None. Cached: refusal is ordinary traffic's fate,
    so this runs per request, and a process talks to few hosts."""
    host = host.lower()
    for name, label in _UNREAD_MODEL_HOSTS:
        if host == name or (name[0] in ".-" and host.endswith(name)):
            return label
    if host.startswith("bedrock-runtime.") and host.endswith(".amazonaws.com"):
        return "Amazon Bedrock"
    return None


def unread_model_api(host: str, path: str) -> str | None:
    """Which unread model API (`_UNREAD_MODEL_HOSTS`) this request head names, or None."""
    api = _unread_model_host(host)
    if api is not None:
        return api
    if path.endswith(_OLLAMA_PATHS):
        return "the Ollama API"
    head, _, last = path.rpartition("/")
    if last.endswith(_GEMINI_METHODS):
        return "the Google Gemini API"
    if last in _BEDROCK_OPERATIONS and head.rpartition("/")[0].endswith("/model"):
        return "Amazon Bedrock"
    return None


def count_refusal(treatment: str | None, host: str, path: str, debug: bool) -> None:
    """The capture mode refused this exchange. A Conversations-API path (provider state, no model)
    is counted. A head that names a model API wardex does not parse yet was a model call refused
    only for not being recognised: recorded as `mode_refused` and said once per API, with what to
    do. Any other refusal is ordinary traffic outside agent work and stays uncounted, as is a path
    the endpoint table reads (`llm_call`, Bedrock's Converse on its own host among them): that API
    is parsed, and this exchange was simply not a model call. Under debug, every refusal is said by
    host and path with why, rate-limited (`debug_refused_call`)."""
    unread = treatment not in ("provider_state", "llm_call")
    api = unread_model_api(host, path) if unread else None
    if debug:
        what = api and f"a call to {api}, which wardex does not parse yet"
        if treatment == "provider_state":
            what = "provider state, not a model call"
        debug_refused_call(
            host,
            path,
            f"{what or 'not a model call wardex reads'}, issued outside any open wardex span: "
            "capture_mode=AGENT keeps such traffic only inside a span",
        )
    if treatment == "provider_state":
        counters.bump("interceptors.seam.provider_state_dropped")
    if api is None:
        return
    lose(
        DropReason.MODE_REFUSED,
        where="interceptors.admission.unread_model_api",
        line=(
            f"a call to {api} outside any wardex span was not recorded: wardex does not parse "
            "that API's requests and replies yet, so capture_mode=AGENT (the default) refuses it "
            "like any other traffic outside agent work. Run the call inside a span "
            "(with wardex.span(...)) to record it as an HTTP span, or pass "
            "capture_mode=CaptureMode.ALL to init() to record every exchange. Each one is "
            "counted under mode_refused in wardex.diagnostics()."
        ),
        key=f"interceptors.admission.unread_model_api.{api}",
    )
